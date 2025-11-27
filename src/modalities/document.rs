//! PDF document pre-tokenization with multi-view support.
//!
//! Uses `pdfium-render` for text and image extraction.
//!
//! # Multi-View Mode
//!
//! When `PdfMode::MultiView` is enabled, the pre-tokenizer emits multiple
//! segments representing different views of the same document:
//!
//! ```text
//! PDF Document (source_id: abc123)
//! ├── pdf_raw          - Original binary (structural patterns)
//! ├── pdf_page_1_text  - Extracted text (semantic content)
//! ├── pdf_page_1_image - Rendered image (visual layout)
//! └── ...
//! ```
//!
//! These views share a `source_id` in their metadata, enabling the hypergraph
//! to create `same_source` edges between them for cross-modal learning.
//!
//! # Requirements
//!
//! - Requires the pdfium library to be installed on the system
//! - Set `PDFIUM_DYNAMIC_LIB_PATH` environment variable if not in system path

use anyhow::Result;
use pdfium_render::prelude::*;
use std::collections::HashMap;
use std::env;
use uuid::Uuid;

use super::{ByteSegment, ModalityPreTokenizer, PdfMode, SegmentMetadata};

/// PDF Pre-Tokenizer using `pdfium-render` for high-fidelity extraction.
///
/// Supports multiple extraction modes via `PdfMode`:
/// - `RawOnly`: Just the original binary (for structural learning)
/// - `TextOnly`: Extracted text only (default, most efficient)
/// - `ImageOnly`: Rendered images only (for layout learning)
/// - `MultiView`: All three views (raw, text, images) for cross-modal learning
pub struct PdfPreTokenizer {
    mode: PdfMode,
    /// Target width for rendered images (height scales proportionally).
    render_width: u16,
}

impl PdfPreTokenizer {
    /// Create a new PDF pre-tokenizer with the specified mode.
    pub fn new(mode: PdfMode) -> Self {
        Self {
            mode,
            render_width: 1024,
        }
    }

    /// Set the target width for rendered images.
    pub fn with_render_width(mut self, width: u16) -> Self {
        self.render_width = width;
        self
    }

    /// Initialize pdfium bindings from system or environment path.
    fn init_pdfium() -> Result<Pdfium> {
        let bindings = if let Ok(path) = env::var("PDFIUM_DYNAMIC_LIB_PATH") {
            Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path(&path))
                .or_else(|_| Pdfium::bind_to_system_library())
        } else {
            Pdfium::bind_to_system_library()
        }
        .map_err(|e| anyhow::anyhow!("Failed to bind to Pdfium library: {e}"))?;

        Ok(Pdfium::new(bindings))
    }

    /// Extract raw PDF bytes as a single segment.
    fn extract_raw(&self, data: &[u8], source_id: &str, page_count: usize) -> ByteSegment {
        ByteSegment {
            bytes: data.to_vec(),
            label: Some("pdf_raw".to_string()),
            metadata: Some(SegmentMetadata {
                start_offset: 0,
                end_offset: data.len(),
                confidence: 1.0,
                extra: Some(serde_json::json!({
                    "modality": "pdf_binary",
                    "page_count": page_count,
                    "source_type": "raw",
                    "source_id": source_id,
                    "view": "raw"
                })),
            }),
        }
    }

    /// Extract text from a single page.
    fn extract_page_text(
        &self,
        page: &PdfPage,
        page_num: usize,
        source_id: &str,
    ) -> Option<ByteSegment> {
        match page.text() {
            Ok(text_page) => {
                let text = text_page.all();
                if text.trim().is_empty() {
                    return None;
                }

                let text_bytes = text.into_bytes();
                Some(ByteSegment {
                    bytes: text_bytes.clone(),
                    label: Some(format!("pdf_page_{}_text", page_num)),
                    metadata: Some(SegmentMetadata {
                        start_offset: 0,
                        end_offset: text_bytes.len(),
                        confidence: 1.0,
                        extra: Some(serde_json::json!({
                            "page": page_num,
                            "modality": "pdf_text",
                            "source_type": "extracted_text",
                            "source_id": source_id,
                            "view": "text"
                        })),
                    }),
                })
            }
            Err(e) => {
                eprintln!(
                    "Warning: Failed to extract text from PDF page {}: {}",
                    page_num, e
                );
                None
            }
        }
    }

    /// Render a page as an image.
    fn extract_page_image(
        &self,
        page: &PdfPage,
        page_num: usize,
        source_id: &str,
    ) -> Option<ByteSegment> {
        let render_config = PdfRenderConfig::new().set_target_width(self.render_width as i32);

        match page.render_with_config(&render_config) {
            Ok(bitmap) => {
                let width = bitmap.width();
                let height = bitmap.height();
                let format = format!("{:?}", bitmap.format());
                let raw_bytes = bitmap.as_raw_bytes().to_vec();

                Some(ByteSegment {
                    bytes: raw_bytes,
                    label: Some(format!("pdf_page_{}_image", page_num)),
                    metadata: Some(SegmentMetadata {
                        start_offset: 0,
                        end_offset: 0, // Images don't have byte offsets in source
                        confidence: 1.0,
                        extra: Some(serde_json::json!({
                            "page": page_num,
                            "modality": "pdf_image",
                            "source_type": "rendered_image",
                            "source_id": source_id,
                            "view": "image",
                            "format": format,
                            "dimensions": {
                                "width": width,
                                "height": height
                            }
                        })),
                    }),
                })
            }
            Err(e) => {
                eprintln!(
                    "Warning: Failed to render PDF page {}: {}",
                    page_num, e
                );
                None
            }
        }
    }
}

impl ModalityPreTokenizer for PdfPreTokenizer {
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>> {
        let pdfium = Self::init_pdfium()?;
        let doc = pdfium
            .load_pdf_from_byte_slice(data, None)
            .map_err(|e| anyhow::anyhow!("Failed to parse PDF: {e}"))?;

        let mut segments = Vec::new();
        let source_id = Uuid::new_v4().to_string();
        let page_count = doc.pages().len() as usize;

        // Determine what to extract based on mode
        let (extract_raw, extract_text, extract_images) = match self.mode {
            PdfMode::RawOnly => (true, false, false),
            PdfMode::TextOnly => (false, true, false),
            PdfMode::ImageOnly => (false, false, true),
            PdfMode::MultiView => (true, true, true),
        };

        // Extract raw bytes if requested
        if extract_raw {
            segments.push(self.extract_raw(data, &source_id, page_count));
        }

        // Process each page
        for (index, page) in doc.pages().iter().enumerate() {
            let page_num = index + 1;

            if extract_text {
                if let Some(text_segment) = self.extract_page_text(&page, page_num, &source_id) {
                    segments.push(text_segment);
                }
            }

            if extract_images {
                if let Some(image_segment) = self.extract_page_image(&page, page_num, &source_id) {
                    segments.push(image_segment);
                }
            }
        }

        Ok(segments)
    }

    fn modality(&self) -> &str {
        "pdf"
    }

    fn supports_multiview(&self) -> bool {
        true
    }

    fn pre_tokenize_multiview(
        &self,
        data: &[u8],
    ) -> Result<HashMap<String, Vec<ByteSegment>>> {
        let pdfium = Self::init_pdfium()?;
        let doc = pdfium
            .load_pdf_from_byte_slice(data, None)
            .map_err(|e| anyhow::anyhow!("Failed to parse PDF: {e}"))?;

        let source_id = Uuid::new_v4().to_string();
        let page_count = doc.pages().len() as usize;

        let mut views: HashMap<String, Vec<ByteSegment>> = HashMap::new();

        // Always include raw view
        views.insert(
            "raw".to_string(),
            vec![self.extract_raw(data, &source_id, page_count)],
        );

        // Collect text segments
        let mut text_segments = Vec::new();
        let mut image_segments = Vec::new();

        for (index, page) in doc.pages().iter().enumerate() {
            let page_num = index + 1;

            if let Some(text_segment) = self.extract_page_text(&page, page_num, &source_id) {
                text_segments.push(text_segment);
            }

            if let Some(image_segment) = self.extract_page_image(&page, page_num, &source_id) {
                image_segments.push(image_segment);
            }
        }

        if !text_segments.is_empty() {
            views.insert("text".to_string(), text_segments);
        }

        if !image_segments.is_empty() {
            views.insert("image".to_string(), image_segments);
        }

        Ok(views)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdf_mode_from_str() {
        assert_eq!("raw_only".parse::<PdfMode>().unwrap(), PdfMode::RawOnly);
        assert_eq!("text".parse::<PdfMode>().unwrap(), PdfMode::TextOnly);
        assert_eq!("multiview".parse::<PdfMode>().unwrap(), PdfMode::MultiView);
        assert!("invalid".parse::<PdfMode>().is_err());
    }

    #[test]
    fn test_pdf_mode_default() {
        assert_eq!(PdfMode::default(), PdfMode::TextOnly);
    }

    #[test]
    fn test_multiview_support() {
        let pretok = PdfPreTokenizer::new(PdfMode::MultiView);
        assert!(pretok.supports_multiview());
        assert_eq!(pretok.modality(), "pdf");
    }
}

