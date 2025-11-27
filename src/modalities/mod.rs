//! Modality-specific pre-tokenization modules.
//!
//! This module provides a clean separation of concerns for different data modalities,
//! enabling multi-view learning for byte-level cross-modal association.
//!
//! # Architecture
//!
//! Each modality implements the `ModalityPreTokenizer` trait:
//! - **text**: Raw UTF-8 bytes (BLT style) or HuggingFace tokenizer
//! - **image**: Patch-based with adaptive entropy merging
//! - **audio**: Frame-based using Symphonia (pure Rust)
//! - **code**: AST-aware using Tree-sitter
//! - **document**: PDF with optional multi-view (raw + text + image)
//! - **video**: FFmpeg-based (behind feature flag)
//! - **binary**: ELF/PE section extraction using Goblin
//!
//! # Multi-View Mode
//!
//! When `multiview` is enabled for compound modalities (PDF, video), the pretokenizer
//! emits **multiple ByteSegments** representing different views of the same source:
//!
//! ```text
//! PDF Document
//! ├── View 1: pdf_raw (original binary)
//! ├── View 2: pdf_page_N_text (extracted text per page)
//! └── View 3: pdf_page_N_image (rendered image per page)
//! ```
//!
//! These views are connected via `same_source` hyperedges in the sidecar,
//! which can be used by downstream optimizers (e.g., THRML/EBM) to keep
//! related views close in the learned geometry.

pub mod audio;
pub mod binary;
pub mod code;
pub mod document;
pub mod image;
pub mod text;
pub mod video;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Trait for pre-tokenizing different data modalities into byte segments
/// that can be processed by the BLT entropy model.
pub trait ModalityPreTokenizer: Send + Sync {
    /// Pre-tokenize input bytes into semantically meaningful segments.
    ///
    /// Each segment represents a logical unit (token, patch, frame, etc.)
    /// that will become a leaf node in the hypergraph.
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>>;

    /// Get the modality name for logging/debugging.
    fn modality(&self) -> &str;

    /// Whether this modality supports multi-view extraction.
    ///
    /// If true, `pre_tokenize_multiview` may return additional segments
    /// representing alternative views of the same source.
    fn supports_multiview(&self) -> bool {
        false
    }

    /// Extract multiple views of the same source.
    ///
    /// Returns a map of view_name -> segments, where each view represents
    /// a different perspective (raw, text, image, etc.).
    ///
    /// Default implementation returns a single "default" view.
    fn pre_tokenize_multiview(
        &self,
        data: &[u8],
    ) -> Result<std::collections::HashMap<String, Vec<ByteSegment>>> {
        let segments = self.pre_tokenize(data)?;
        let mut views = std::collections::HashMap::new();
        views.insert("default".to_string(), segments);
        Ok(views)
    }
}

/// A segment of bytes with optional metadata.
///
/// Represents a logical unit extracted from a modality:
/// - Text: character, token, or word
/// - Image: patch of pixels
/// - Audio: frame of samples
/// - Code: AST node (function, class, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByteSegment {
    /// The raw bytes for this segment.
    /// Skipped in serialization to keep metadata lightweight.
    #[serde(skip)]
    pub bytes: Vec<u8>,

    /// Optional semantic label (e.g., "code_function", "image_patch", "audio_frame").
    pub label: Option<String>,

    /// Optional metadata for this segment.
    pub metadata: Option<SegmentMetadata>,
}

/// Metadata attached to a ByteSegment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMetadata {
    /// Start position in original data (byte offset).
    pub start_offset: usize,

    /// End position in original data (byte offset).
    pub end_offset: usize,

    /// Confidence score for this segmentation (0.0 to 1.0).
    pub confidence: f32,

    /// Additional modality-specific metadata (JSON value).
    pub extra: Option<serde_json::Value>,
}

/// Configuration for PDF processing modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdfMode {
    /// Extract only raw PDF bytes (for structural learning).
    RawOnly,
    /// Extract only text content (default, most efficient).
    TextOnly,
    /// Render pages as images (for layout-aware processing).
    ImageOnly,
    /// Full multi-view: raw + text + images.
    MultiView,
}

impl Default for PdfMode {
    fn default() -> Self {
        PdfMode::TextOnly
    }
}

impl std::str::FromStr for PdfMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "raw_only" | "raw" => Ok(PdfMode::RawOnly),
            "text_only" | "text" => Ok(PdfMode::TextOnly),
            "image_only" | "image" => Ok(PdfMode::ImageOnly),
            "multiview" | "multi_view" | "all" => Ok(PdfMode::MultiView),
            _ => Err(format!(
                "Unknown PDF mode: {}. Expected: raw_only, text_only, image_only, or multiview",
                s
            )),
        }
    }
}

/// Auto-detect content type from leading bytes/file signatures and return an appropriate `PreTokenizerType`.
pub fn detect_modality(data: &[u8]) -> PreTokenizerType {
    if data.len() < 4 {
        return PreTokenizerType::TextRaw;
    }

    // Magic bytes check
    if data.starts_with(b"\xFF\xD8") {
        // JPEG
        return PreTokenizerType::Image {
            patch_size: 16,
            stride: 16,
        };
    } else if data.starts_with(b"\x89PNG") {
        // PNG
        return PreTokenizerType::Image {
            patch_size: 16,
            stride: 16,
        };
    } else if data.starts_with(b"%PDF-") {
        // PDF
        return PreTokenizerType::Pdf {
            mode: PdfMode::default(),
        };
    } else if data.len() > 8 && &data[4..8] == b"ftyp" {
        // MP4/Video
        return PreTokenizerType::Video { frame_rate: 30 };
    } else if data.starts_with(b"RIFF") && data.len() > 12 && &data[8..12] == b"WAVE" {
        // WAV
        return PreTokenizerType::Audio {
            frame_size: 160,
            sample_rate: 16000,
        };
    } else if data.starts_with(b"ID3")
        || (data.starts_with(b"\xFF\xFB")
            || data.starts_with(b"\xFF\xF3")
            || data.starts_with(b"\xFF\xF2"))
    {
        // MP3 (ID3 or sync)
        return PreTokenizerType::Audio {
            frame_size: 160,
            sample_rate: 44100,
        };
    } else if data.starts_with(&[0x7F, b'E', b'L', b'F']) {
        // ELF Binary
        return PreTokenizerType::Binary;
    } else if data.starts_with(b"PK\x03\x04") {
        // ZIP / Jar / Docx
        return PreTokenizerType::Binary;
    }

    // Heuristics for code
    if data.starts_with(b"#!/")
        || data.windows(7).any(|w| w == b"import ")
        || data.windows(3).any(|w| w == b"fn ")
    {
        if data.windows(4).any(|w| w == b"def ") {
            return PreTokenizerType::Code {
                language: "python".to_string(),
            };
        }
        if data.windows(3).any(|w| w == b"fn ") {
            return PreTokenizerType::Code {
                language: "rust".to_string(),
            };
        }
    }

    PreTokenizerType::TextRaw
}

/// Pre-tokenizer type enum for factory pattern.
#[derive(Debug, Clone)]
pub enum PreTokenizerType {
    /// HuggingFace tokenizer-based text pre-tokenization.
    TextSimple,
    /// Raw UTF-8 bytes (true BLT style).
    TextRaw,
    /// Load HuggingFace tokenizer from file.
    TextFromFile { path: String },
    /// Image patch extraction with adaptive entropy merging.
    Image { patch_size: usize, stride: usize },
    /// Audio frame extraction using Symphonia.
    Audio { frame_size: usize, sample_rate: u32 },
    /// AST-aware code segmentation using Tree-sitter.
    Code { language: String },
    /// PDF document processing with configurable mode.
    Pdf { mode: PdfMode },
    /// Video frame extraction (requires FFmpeg, behind feature flag).
    Video { frame_rate: u32 },
    /// Binary/ELF section extraction using Goblin.
    Binary,
}

impl PreTokenizerType {
    /// Create the appropriate pre-tokenizer for this type.
    pub fn create(&self) -> Result<Box<dyn ModalityPreTokenizer>> {
        match self {
            PreTokenizerType::TextSimple => Ok(Box::new(text::TextPreTokenizer::new_simple()?)),
            PreTokenizerType::TextRaw => Ok(Box::new(text::RawTextPreTokenizer)),
            PreTokenizerType::TextFromFile { path } => {
                Ok(Box::new(text::TextPreTokenizer::from_file(path)?))
            }
            PreTokenizerType::Image { patch_size, stride } => {
                Ok(Box::new(image::ImagePreTokenizer::new(*patch_size, *stride)))
            }
            PreTokenizerType::Audio {
                frame_size,
                sample_rate,
            } => Ok(Box::new(audio::AudioPreTokenizer::new(
                *frame_size,
                *sample_rate,
            ))),
            PreTokenizerType::Code { language } => {
                Ok(Box::new(code::CodePreTokenizer::new(language.clone())))
            }
            PreTokenizerType::Pdf { mode } => Ok(Box::new(document::PdfPreTokenizer::new(*mode))),
            PreTokenizerType::Video { frame_rate } => {
                Ok(Box::new(video::VideoPreTokenizer::new(*frame_rate)))
            }
            PreTokenizerType::Binary => Ok(Box::new(binary::BinaryPreTokenizer)),
        }
    }
}

// Re-export commonly used types for convenience
pub use audio::AudioPreTokenizer;
pub use binary::BinaryPreTokenizer;
pub use code::CodePreTokenizer;
pub use document::PdfPreTokenizer;
pub use image::ImagePreTokenizer;
pub use text::{RawTextPreTokenizer, TextPreTokenizer};
pub use video::VideoPreTokenizer;

