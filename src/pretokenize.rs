use anyhow::Result;
use tokenizers::Tokenizer as HFTokenizer;
use std::io::Cursor;
use tree_sitter::{Parser, Language};

/// Trait for pre-tokenizing different data modalities into byte segments
/// that can be processed by the BLT entropy model.
pub trait ModalityPreTokenizer {
    /// Pre-tokenize input bytes into semantically meaningful segments
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>>;
    
    /// Get the modality name for logging/debugging
    fn modality(&self) -> &str;
}

/// A segment of bytes with optional metadata
#[derive(Debug, Clone)]
pub struct ByteSegment {
    /// The raw bytes for this segment
    pub bytes: Vec<u8>,
    
    /// Optional semantic label (e.g., "code_function", "image_patch", "audio_frame")
    pub label: Option<String>,
    
    /// Optional metadata for this segment
    pub metadata: Option<SegmentMetadata>,
}

#[derive(Debug, Clone)]
pub struct SegmentMetadata {
    /// Start position in original data
    pub start_offset: usize,
    
    /// End position in original data
    pub end_offset: usize,
    
    /// Confidence score for this segmentation (0.0 to 1.0)
    pub confidence: f32,
    
    /// Additional modality-specific metadata
    pub extra: Option<serde_json::Value>,
}

/// Auto-detect content type and return appropriate PreTokenizerType
pub fn detect_modality(data: &[u8]) -> PreTokenizerType {
    if data.len() < 4 {
        return PreTokenizerType::TextRaw;
    }

    // Magic bytes check
    if data.starts_with(b"\xFF\xD8") { // JPEG
        return PreTokenizerType::Image { patch_size: 16, stride: 16 };
    } else if data.starts_with(b"\x89PNG") { // PNG
        return PreTokenizerType::Image { patch_size: 16, stride: 16 };
    } else if data.starts_with(b"RIFF") && data.len() > 12 && &data[8..12] == b"WAVE" { // WAV
        return PreTokenizerType::Audio { frame_size: 160, sample_rate: 16000 };
    } else if data.starts_with(b"ID3") || (data.starts_with(b"\xFF\xFB") || data.starts_with(b"\xFF\xF3") || data.starts_with(b"\xFF\xF2")) { // MP3 (ID3 or sync)
        // MP3 decoding needs more deps, defaulting to Audio with standard assumption or failing later
        return PreTokenizerType::Audio { frame_size: 160, sample_rate: 44100 };
    }

    // Heuristics for code
    if data.starts_with(b"#!/") || data.windows(7).any(|w| w == b"import ") || data.windows(3).any(|w| w == b"fn ") {
         // Ideally we'd detect language, but defaulting to Python or Rust based on hints
         if data.windows(4).any(|w| w == b"def ") {
             return PreTokenizerType::Code { language: "python".to_string() };
         }
         if data.windows(3).any(|w| w == b"fn ") {
             return PreTokenizerType::Code { language: "rust".to_string() };
         }
    }
    
    PreTokenizerType::TextRaw
}

/// Text pre-tokenizer using Hugging Face tokenizers
pub struct TextPreTokenizer {
    tokenizer: HFTokenizer,
}

impl TextPreTokenizer {
    pub fn new(tokenizer: HFTokenizer) -> Self {
        Self { tokenizer }
    }
    
    /// Create from a tokenizer file
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer = HFTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        Ok(Self { tokenizer })
    }
    
    /// Create a simple whitespace-based tokenizer as fallback
    pub fn new_simple() -> Result<Self> {
        use tokenizers::{models::bpe::BPE, Tokenizer};
        use tokenizers::pre_tokenizers::whitespace::Whitespace;
        
        let mut tokenizer = Tokenizer::new(BPE::default());
        tokenizer.with_pre_tokenizer(Some(Whitespace {}));     
        Ok(Self { tokenizer })
    }
}

impl ModalityPreTokenizer for TextPreTokenizer {
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>> {
        let text = String::from_utf8_lossy(data);
        
        let encoding = self.tokenizer.encode(text.as_ref(), false)
            .map_err(|e| anyhow::anyhow!("Encoding error: {}", e))?;
        let tokens = encoding.get_tokens();
        let offsets = encoding.get_offsets();
        
        let segments = tokens.iter()
            .zip(offsets.iter())
            .map(|(token, (start, end))| {
                ByteSegment {
                    bytes: data[*start..*end].to_vec(),
                    label: Some("text_token".to_string()),
                    metadata: Some(SegmentMetadata {
                        start_offset: *start,
                        end_offset: *end,
                        confidence: 1.0,
                        extra: Some(serde_json::json!({
                            "token": token,
                        })),
                    }),
                }
            })
            .collect();
        
        Ok(segments)
    }
    
    fn modality(&self) -> &str {
        "text"
    }
}

/// Raw UTF-8 pre-tokenizer (True BLT style)
pub struct RawTextPreTokenizer;

impl ModalityPreTokenizer for RawTextPreTokenizer {
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>> {
        // Return char-level segments for granular processing
        let text = String::from_utf8_lossy(data);
        let mut segments = Vec::new();
        let mut byte_offset = 0;

        for char in text.chars() {
            let char_len = char.len_utf8();
            // Ensure we don't go out of bounds if String::from_utf8_lossy replaced chars
            if byte_offset + char_len > data.len() {
                break;
            }
            
            let char_bytes = data[byte_offset..byte_offset + char_len].to_vec();
            
            segments.push(ByteSegment {
                bytes: char_bytes,
                label: Some("char".to_string()),
                metadata: Some(SegmentMetadata {
                    start_offset: byte_offset,
                    end_offset: byte_offset + char_len,
                    confidence: 1.0,
                    extra: Some(serde_json::json!({
                        "char": char.to_string()
                    })),
                }),
            });
            byte_offset += char_len;
        }
        
        Ok(segments)
    }

    fn modality(&self) -> &str {
        "text_raw"
    }
}

/// Image pre-tokenizer using patch-based segmentation on RAW PIXELS
/// Enhanced with adaptive entropy-based merging
pub struct ImagePreTokenizer {
    patch_size: usize,
    stride: usize,
}

impl ImagePreTokenizer {
    pub fn new(patch_size: usize, stride: usize) -> Self {
        Self { patch_size, stride }
    }
    
    fn compute_patch_entropy(bytes: &[u8]) -> f32 {
        let mut counts = [0usize; 256];
        for &b in bytes {
            counts[b as usize] += 1;
        }
        let total = bytes.len() as f32;
        if total == 0.0 { return 0.0; }
        
        let mut entropy = 0.0;
        for &count in &counts {
            if count > 0 {
                let p = count as f32 / total;
                entropy -= p * p.log2();
            }
        }
        entropy
    }
}

impl ModalityPreTokenizer for ImagePreTokenizer {
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>> {
        // Decode image from bytes (supports PNG, JPEG, etc.)
        let img = image::load_from_memory(data)
            .map_err(|e| anyhow::anyhow!("Failed to decode image: {}", e))?;
        
        // Convert to RGB8
        let rgb = img.to_rgb8();
        let (width, height) = rgb.dimensions();
        
        let mut initial_patches = Vec::new();
        
        // Step 1: Extract standard grid patches
        for y in (0..height).step_by(self.stride) {
            for x in (0..width).step_by(self.stride) {
                let mut patch_bytes = Vec::with_capacity(self.patch_size * self.patch_size * 3);
                for py in 0..self.patch_size as u32 {
                    for px in 0..self.patch_size as u32 {
                        let pixel = if x + px < width && y + py < height {
                            *rgb.get_pixel(x + px, y + py)
                        } else {
                            image::Rgb([0, 0, 0])
                        };
                        patch_bytes.extend_from_slice(&pixel.0);
                    }
                }
                
                let entropy = Self::compute_patch_entropy(&patch_bytes);
                
                initial_patches.push(ByteSegment {
                    bytes: patch_bytes,
                    label: Some("image_patch".to_string()),
                    metadata: Some(SegmentMetadata {
                        start_offset: (y * width + x) as usize * 3, 
                        end_offset: ((y + self.patch_size as u32) * width + x + self.patch_size as u32) as usize * 3,
                        confidence: 1.0,
                        extra: Some(serde_json::json!({
                            "x": x, "y": y, "width": width, "height": height, "local_entropy": entropy
                        })),
                    }),
                });
            }
        }

        // Step 2: Adaptive Merging
        if initial_patches.is_empty() { return Ok(vec![]); }
        
        let mut merged = vec![initial_patches[0].clone()];
        
        for patch in initial_patches.into_iter().skip(1) {
            let last_meta = merged.last().unwrap().metadata.as_ref().unwrap().extra.as_ref().unwrap();
            let last_entropy = last_meta["local_entropy"].as_f64().unwrap_or(0.0) as f32;
            
            let this_meta = patch.metadata.as_ref().unwrap().extra.as_ref().unwrap();
            let this_entropy = this_meta["local_entropy"].as_f64().unwrap_or(0.0) as f32;
            
            // Merge threshold: low entropy implies uniform region
            if (last_entropy + this_entropy) / 2.0 < 1.5 { 
                let last_seg = merged.last_mut().unwrap();
                last_seg.bytes.extend(&patch.bytes);
                // Update metadata to reflect merged region
                if let Some(meta) = &mut last_seg.metadata {
                    meta.end_offset = patch.metadata.as_ref().unwrap().end_offset;
                }
                // We keep the label as "image_patch" but it's now a "super-patch"
            } else {
                merged.push(patch);
            }
        }
        
        Ok(merged)
    }
    
    fn modality(&self) -> &str {
        "image"
    }
}

/// Audio pre-tokenizer using frame-based segmentation on RAW PCM
pub struct AudioPreTokenizer {
    frame_size: usize,
    sample_rate: u32,
}

impl AudioPreTokenizer {
    pub fn new(frame_size: usize, sample_rate: u32) -> Self {
        Self { frame_size, sample_rate }
    }
}

impl ModalityPreTokenizer for AudioPreTokenizer {
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>> {
        // Try to decode as WAV first
        let cursor = Cursor::new(data);
        let mut reader = hound::WavReader::new(cursor)
            .map_err(|e| anyhow::anyhow!("Failed to read WAV header: {}", e))?;
            
        let spec = reader.spec();
        
        let samples: Vec<i16> = reader.samples::<i16>()
            .filter_map(|s| s.ok())
            .collect();
            
        // Convert back to bytes (little endian)
        let raw_bytes: Vec<u8> = samples.iter()
            .flat_map(|&s| s.to_le_bytes().to_vec())
            .collect();
            
        // Now chunk into frames
        let bytes_per_frame = self.frame_size * 2; // 2 bytes per sample
        let mut segments = Vec::new();
        let mut offset = 0;
        
        while offset + bytes_per_frame <= raw_bytes.len() {
            segments.push(ByteSegment {
                bytes: raw_bytes[offset..offset + bytes_per_frame].to_vec(),
                label: Some("audio_frame".to_string()),
                metadata: Some(SegmentMetadata {
                    start_offset: offset,
                    end_offset: offset + bytes_per_frame,
                    confidence: 1.0,
                    extra: Some(serde_json::json!({
                        "frame_index": segments.len(),
                        "sample_rate": spec.sample_rate,
                        "channels": spec.channels
                    })),
                }),
            });
            offset += bytes_per_frame;
        }
        
        Ok(segments)
    }
    
    fn modality(&self) -> &str {
        "audio"
    }
}

/// Code pre-tokenizer using Tree-Sitter for AST-aware segmentation
pub struct CodePreTokenizer {
    language: String,
}

impl CodePreTokenizer {
    pub fn new(language: String) -> Self {
        Self { language }
    }
}

impl ModalityPreTokenizer for CodePreTokenizer {
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>> {
        let mut parser = Parser::new();
        
        let lang: Language = match self.language.as_str() {
            "rust" => tree_sitter_rust::LANGUAGE.into(),
            "python" => tree_sitter_python::LANGUAGE.into(),
            // Fallback or error for unsupported langs
            _ => return Err(anyhow::anyhow!("Unsupported language for AST parsing")),
        };
        
        parser.set_language(&lang)?;
        
        // Tree-sitter expects string-like input usually, but works on bytes
        let tree = parser.parse(data, None)
            .ok_or_else(|| anyhow::anyhow!("Failed to parse code"))?;
            
        let root = tree.root_node();
        let mut segments = Vec::new();
        let mut cursor = root.walk();
        
        // Traverse the tree
        let mut reached_leaf = false;
        while !reached_leaf {
            let node = cursor.node();
            let kind = node.kind();
            
            // Filter for "interesting" nodes that represent semantic units
            // Rust: function_item, struct_item, impl_item, module
            // Python: function_definition, class_definition, import_statement
            let is_semantic_unit = match self.language.as_str() {
                "rust" => matches!(kind, "function_item" | "struct_item" | "impl_item" | "mod_item"),
                "python" => matches!(kind, "function_definition" | "class_definition" | "import_statement"),
                _ => node.is_named(), // Fallback
            };
            
            if is_semantic_unit {
                let start = node.start_byte();
                let end = node.end_byte();
                if end <= data.len() {
                    segments.push(ByteSegment {
                        bytes: data[start..end].to_vec(),
                        label: Some(kind.to_string()),
                        metadata: Some(SegmentMetadata {
                            start_offset: start,
                            end_offset: end,
                            confidence: 1.0,
                            extra: Some(serde_json::json!({
                                "node_type": kind,
                                "is_named": node.is_named()
                            })),
                        }),
                    });
                }
                // Skip children of this node as we captured the whole unit
                if !cursor.goto_next_sibling() {
                    if !cursor.goto_parent() { break; }
                    while !cursor.goto_next_sibling() {
                        if !cursor.goto_parent() { reached_leaf = true; break; }
                    }
                }
            } else {
                // Not a unit, descend
                if !cursor.goto_first_child() {
                    if !cursor.goto_next_sibling() {
                         if !cursor.goto_parent() { break; }
                         while !cursor.goto_next_sibling() {
                             if !cursor.goto_parent() { reached_leaf = true; break; }
                         }
                    }
                }
            }
        }
        
        // If no top-level structure found, fallback
        if segments.is_empty() {
             segments.push(ByteSegment {
                bytes: data.to_vec(),
                label: Some("code_snippet".to_string()),
                metadata: Some(SegmentMetadata {
                    start_offset: 0,
                    end_offset: data.len(),
                    confidence: 0.5,
                    extra: None,
                }),
            });
        }
        
        Ok(segments)
    }
    
    fn modality(&self) -> &str {
        "code_ast"
    }
}

/// Pre-tokenizer factory for creating the appropriate pre-tokenizer
pub enum PreTokenizerType {
    TextSimple,
    TextRaw, // NEW: True BLT style
    TextFromFile { path: String },
    Image { patch_size: usize, stride: usize },
    Audio { frame_size: usize, sample_rate: u32 },
    Code { language: String },
}

impl PreTokenizerType {
    pub fn create(&self) -> Result<Box<dyn ModalityPreTokenizer>> {
        match self {
            PreTokenizerType::TextSimple => {
                Ok(Box::new(TextPreTokenizer::new_simple()?))
            }
            PreTokenizerType::TextRaw => {
                Ok(Box::new(RawTextPreTokenizer))
            }
            PreTokenizerType::TextFromFile { path } => {
                Ok(Box::new(TextPreTokenizer::from_file(path)?))
            }
            PreTokenizerType::Image { patch_size, stride } => {
                Ok(Box::new(ImagePreTokenizer::new(*patch_size, *stride)))
            }
            PreTokenizerType::Audio { frame_size, sample_rate } => {
                Ok(Box::new(AudioPreTokenizer::new(*frame_size, *sample_rate)))
            }
            PreTokenizerType::Code { language } => {
                Ok(Box::new(CodePreTokenizer::new(language.clone())))
            }
        }
    }
}
