use anyhow::Result;
use tokenizers::Tokenizer as HFTokenizer;
use std::io::Cursor;
use tree_sitter::{Parser, Language};
use serde::{Serialize, Deserialize};
use tempfile::NamedTempFile;
use std::io::Write;

// Symphonia imports for audio/video
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;
use symphonia::core::formats::FormatOptions;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::CODEC_TYPE_NULL;

// PDF imports
// use pdf::file::File as PdfFile;

// Goblin imports  
use goblin::elf::Elf;

/// Trait for pre-tokenizing different data modalities into byte segments
/// that can be processed by the BLT entropy model.
pub trait ModalityPreTokenizer {
    /// Pre-tokenize input bytes into semantically meaningful segments
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>>;
    
    /// Get the modality name for logging/debugging
    fn modality(&self) -> &str;
}

/// A segment of bytes with optional metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByteSegment {
    /// The raw bytes for this segment (skipped in serialization to keep metadata lightweight)
    #[serde(skip)]
    pub bytes: Vec<u8>,
    
    /// Optional semantic label (e.g., "code_function", "image_patch", "audio_frame")
    pub label: Option<String>,
    
    /// Optional metadata for this segment
    pub metadata: Option<SegmentMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    } else if data.starts_with(b"%PDF-") { // PDF
        return PreTokenizerType::Pdf { extract_text: true };
    } else if data.len() > 8 && &data[4..8] == b"ftyp" { // MP4/Video
        return PreTokenizerType::Video { frame_rate: 30 };
    } else if data.starts_with(b"RIFF") && data.len() > 12 && &data[8..12] == b"WAVE" { // WAV
        return PreTokenizerType::Audio { frame_size: 160, sample_rate: 16000 };
    } else if data.starts_with(b"ID3") || (data.starts_with(b"\xFF\xFB") || data.starts_with(b"\xFF\xF3") || data.starts_with(b"\xFF\xF2")) { // MP3 (ID3 or sync)
        return PreTokenizerType::Audio { frame_size: 160, sample_rate: 44100 };
    } else if data.starts_with(&[0x7F, b'E', b'L', b'F']) { // ELF Binary
        return PreTokenizerType::Binary;
    } else if data.starts_with(b"PK\x03\x04") { // ZIP / Jar / Docx
         return PreTokenizerType::Binary; 
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

/// Audio pre-tokenizer using Symphonia for pure-Rust decoding (WAV/MP3/OGG/etc.)
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
        // Use Symphonia to decode
        let mss = MediaSourceStream::new(Box::new(Cursor::new(data.to_vec())), Default::default());
        
        // Probe the media source
        let probed = symphonia::default::get_probe().format(
            &Hint::new(),
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default()
        )?;
        let mut format = probed.format;
        
        // Find the first audio track
        let track = format.tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| anyhow::anyhow!("No supported audio track"))?;
            
        // Use the default decoder for the track
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &Default::default())?;
        let track_id = track.id;
        
        let mut segments = Vec::new();
        let mut offset = 0;
        
        // Decode loop
        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(symphonia::core::errors::Error::IoError(_)) => break, // End of stream
                Err(e) => return Err(anyhow::anyhow!("Error reading packet: {}", e)),
            };
            
            if packet.track_id() != track_id { continue; }
            
            match decoder.decode(&packet) {
                Ok(decoded) => {
                    // Convert to 16-bit interleaved PCM
                    let spec = *decoded.spec();
                    let duration = decoded.capacity() as u64;
                    let mut sample_buf = SampleBuffer::<i16>::new(duration, spec);
                    sample_buf.copy_interleaved_ref(decoded);
                    
                    let samples = sample_buf.samples();
                    // Convert i16 samples to bytes
                    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
                    
                    // Chunk into frame_size
                    let bytes_per_frame = self.frame_size * 2; // i16 = 2 bytes
                    for chunk in bytes.chunks(bytes_per_frame) {
                        if chunk.len() == bytes_per_frame { // Only full frames? Or allow partial?
                            segments.push(ByteSegment {
                                bytes: chunk.to_vec(),
                                label: Some("audio_frame".to_string()),
                                metadata: Some(SegmentMetadata {
                                    start_offset: offset,
                                    end_offset: offset + chunk.len(),
                                    confidence: 1.0,
                                    extra: Some(serde_json::json!({
                                        "frame_index": segments.len(),
                                        "sample_rate": spec.rate,
                                        "channels": spec.channels.count()
                                    })),
                                }),
                            });
                            offset += chunk.len();
                        }
                    }
                },
                Err(_) => continue, // Skip decode errors
            }
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

/// PDF Pre-Tokenizer using pure-Rust `pdf` crate
pub struct PdfPreTokenizer {
    extract_text: bool,
}

impl ModalityPreTokenizer for PdfPreTokenizer {
    fn pre_tokenize(&self, _data: &[u8]) -> Result<Vec<ByteSegment>> {
        // Requires 'pdf' crate.
        // Note: 'pdf' 0.9 API for from_data might differ or require specific generics.
        // Placeholder for now to ensure build stability.
        // let file = PdfFile::from_data(data).map_err(|e| anyhow::anyhow!("Failed to parse PDF: {:?}", e))?;
        Err(anyhow::anyhow!("PDF support implementation pending API verification"))
        
        /* 
        // Original logic stub:
        let mut segments = Vec::new();
        for (i, page) in file.pages().enumerate() {
             // ...
        }
        Ok(segments)
        */
    }
    
    fn modality(&self) -> &str {
        "pdf"
    }
}

/// Video Pre-Tokenizer using FFmpeg via video-rs
/// 
/// FFmpeg is required and will be automatically installed during build if missing.
/// Supports all major video codecs: H.264, H.265/HEVC, VP8, VP9, AV1, MPEG-4, MPEG-2, and more.
pub struct VideoPreTokenizer {
    frame_rate: u32,
}

impl VideoPreTokenizer {
    pub fn new(frame_rate: u32) -> Self {
        Self {
            frame_rate,
        }
    }
    
    fn extract_with_ffmpeg(&self, data: &[u8]) -> Result<Vec<ByteSegment>> {
        // Write to temp file for FFmpeg
        let mut temp = NamedTempFile::new()?;
        temp.write_all(data)?;
        let path = temp.into_temp_path();
        
        video_rs::init().map_err(|e| anyhow::anyhow!("Video init failed: {}", e))?;
        let source = video_rs::Location::File(path.to_path_buf());
        let mut decoder = video_rs::Decoder::new(&source).map_err(|e| anyhow::anyhow!("Decoder creation failed: {}", e))?;
        
        let mut segments = Vec::new();
        let mut frame_count = 0;
        
        // Use decode_raw_iter per compiler suggestion
        for frame_result in decoder.decode_raw_iter() {
            if let Ok(frame) = frame_result {
                let (width, height) = (frame.width(), frame.height());
                // frame.timestamp() usually returns Option<i64> (PTS)
                // We'll use frame index if timestamp is tricky without timebase
                let ts: f64 = frame.timestamp().unwrap_or(0) as f64;
                
                // TODO: Convert frame data to RGB bytes efficiently
                // For now, we create a segment with empty bytes but rich metadata
                // In a real training pipeline, we'd flatten the pixels here
                let raw_bytes = Vec::new(); 
                
                segments.push(ByteSegment {
                    bytes: raw_bytes,
                    label: Some("video_frame".to_string()),
                    metadata: Some(SegmentMetadata {
                        start_offset: 0, // Frames are generated, not sliced directly from input bytes
                        end_offset: 0,
                        confidence: 1.0,
                        extra: Some(serde_json::json!({
                            "timestamp": ts,
                            "width": width,
                            "height": height,
                            "frame_idx": frame_count,
                            "source": "ffmpeg_video_rs"
                        }))
                    })
                });
                frame_count += 1;
            }
        }
        
        // Also extract audio if possible (using pure Rust symphonia as fallback/parallel?)
        // The user said "use FFmpeg exclusively". FFmpeg can extract audio too.
        // But video-rs focuses on video frames.
        // For now, we just return video frames.
        
        Ok(segments)
    }
}

impl ModalityPreTokenizer for VideoPreTokenizer {
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>> {
        // FFmpeg is required - build.rs ensures it's installed
        // Extract video frames using FFmpeg
        self.extract_with_ffmpeg(data)
    }
    
    fn modality(&self) -> &str {
        "video"
    }
}

/// Binary/ELF Pre-Tokenizer using `goblin`
pub struct BinaryPreTokenizer;

impl ModalityPreTokenizer for BinaryPreTokenizer {
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>> {
        // Parse ELF
        let elf = Elf::parse(data)
            .map_err(|e| anyhow::anyhow!("Failed to parse ELF: {}", e))?;
            
        let mut segments = Vec::new();
        
        for section in elf.section_headers {
            let start = section.sh_offset as usize;
            let end = start + section.sh_size as usize;
            
            if end <= data.len() {
                // Get section name from strtab
                let name = elf.shdr_strtab.get_at(section.sh_name).unwrap_or("unknown");
                
                segments.push(ByteSegment {
                    bytes: data[start..end].to_vec(),
                    label: Some(format!("elf_section_{}", name)),
                    metadata: Some(SegmentMetadata {
                        start_offset: start,
                        end_offset: end,
                        confidence: 1.0,
                        extra: Some(serde_json::json!({"type": section.sh_type})),
                    }),
                });
            }
        }
        
        Ok(segments)
    }
    
    fn modality(&self) -> &str {
        "binary_elf"
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
    Pdf { extract_text: bool },
    Video { frame_rate: u32 },
    Binary,
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
            PreTokenizerType::Pdf { extract_text } => {
                Ok(Box::new(PdfPreTokenizer { extract_text: *extract_text }))
            }
            PreTokenizerType::Video { frame_rate } => {
                Ok(Box::new(VideoPreTokenizer::new(*frame_rate)))
            }
            PreTokenizerType::Binary => {
                Ok(Box::new(BinaryPreTokenizer))
            }
        }
    }
}