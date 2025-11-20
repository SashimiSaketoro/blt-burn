use anyhow::Result;
use tokenizers::Tokenizer as HFTokenizer;

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

/// Image pre-tokenizer using patch-based segmentation
pub struct ImagePreTokenizer {
    patch_size: usize,
    stride: usize,
}

impl ImagePreTokenizer {
    pub fn new(patch_size: usize, stride: usize) -> Self {
        Self { patch_size, stride }
    }
}

impl ModalityPreTokenizer for ImagePreTokenizer {
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>> {
        // WARNING: This is processing raw compressed bytes for JPEG/PNG
        // The entropy model will see these as high-entropy noise
        // In production, you should:
        // 1. Detect format (JPEG/PNG/etc)
        // 2. Decode to raw pixels
        // 3. Then create patches from pixel data
        
        // For now, we tag compressed data appropriately
        let is_likely_compressed = data.len() > 4 && (
            (data[0] == 0xFF && data[1] == 0xD8) || // JPEG
            (data[0] == 0x89 && data[1] == 0x50)    // PNG
        );
        
        let mut segments = Vec::new();
        let mut offset = 0;
        
        while offset + self.patch_size <= data.len() {
            segments.push(ByteSegment {
                bytes: data[offset..offset + self.patch_size].to_vec(),
                label: Some(if is_likely_compressed { "compressed_image_patch" } else { "image_patch" }.to_string()),
                metadata: Some(SegmentMetadata {
                    start_offset: offset,
                    end_offset: offset + self.patch_size,
                    confidence: if is_likely_compressed { 0.1 } else { 1.0 }, // Low confidence for compressed
                    extra: Some(serde_json::json!({
                        "patch_index": segments.len(),
                        "is_compressed": is_likely_compressed,
                        "warning": if is_likely_compressed { 
                            serde_json::Value::String("High entropy expected - compressed image format".to_string())
                        } else { 
                            serde_json::Value::Null
                        }
                    })),
                }),
            });
            offset += self.stride;
        }
        
        Ok(segments)
    }
    
    fn modality(&self) -> &str {
        "image"
    }
}

/// Audio pre-tokenizer using frame-based segmentation
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
        // Frame-based segmentation
        let mut segments = Vec::new();
        let mut offset = 0;
        
        while offset + self.frame_size <= data.len() {
            segments.push(ByteSegment {
                bytes: data[offset..offset + self.frame_size].to_vec(),
                label: Some("audio_frame".to_string()),
                metadata: Some(SegmentMetadata {
                    start_offset: offset,
                    end_offset: offset + self.frame_size,
                    confidence: 1.0,
                    extra: Some(serde_json::json!({
                        "frame_index": segments.len(),
                        "sample_rate": self.sample_rate,
                    })),
                }),
            });
            offset += self.frame_size;
        }
        
        Ok(segments)
    }
    
    fn modality(&self) -> &str {
        "audio"
    }
}

/// Code pre-tokenizer (placeholder - would use tree-sitter for AST-aware segmentation)
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
        // TODO: Integrate tree-sitter for AST-aware segmentation
        // For now, use simple line-based segmentation
        let text = String::from_utf8_lossy(data);
        let segments = text.lines()
            .enumerate()
            .map(|(i, line)| {
                ByteSegment {
                    bytes: line.as_bytes().to_vec(),
                    label: Some(format!("code_line_{}", self.language)),
                    metadata: Some(SegmentMetadata {
                        start_offset: 0, // TODO: track actual offsets
                        end_offset: line.len(),
                        confidence: 0.8,
                        extra: Some(serde_json::json!({
                            "line_number": i + 1,
                            "language": &self.language,
                        })),
                    }),
                }
            })
            .collect();
        
        Ok(segments)
    }
    
    fn modality(&self) -> &str {
        "code"
    }
}

/// Pre-tokenizer factory for creating the appropriate pre-tokenizer
pub enum PreTokenizerType {
    TextSimple,
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
