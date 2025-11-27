//! Text pre-tokenization modules.
//!
//! Provides two modes:
//! - **RawTextPreTokenizer**: True BLT-style character-level segmentation
//! - **TextPreTokenizer**: HuggingFace tokenizer-based segmentation

use anyhow::Result;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::{models::bpe::BPE, Tokenizer as HFTokenizer};

use super::{ByteSegment, ModalityPreTokenizer, SegmentMetadata};

/// Text pre-tokenizer using HuggingFace tokenizers.
///
/// Segments text into subword tokens using a trained tokenizer model.
pub struct TextPreTokenizer {
    tokenizer: HFTokenizer,
}

impl TextPreTokenizer {
    /// Create from an existing HuggingFace tokenizer instance.
    pub fn new(tokenizer: HFTokenizer) -> Self {
        Self { tokenizer }
    }

    /// Load a tokenizer from a file path.
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer = HFTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        Ok(Self { tokenizer })
    }

    /// Create a simple whitespace-based tokenizer as fallback.
    pub fn new_simple() -> Result<Self> {
        let mut tokenizer = HFTokenizer::new(BPE::default());
        tokenizer.with_pre_tokenizer(Some(Whitespace {}));
        Ok(Self { tokenizer })
    }
}

impl ModalityPreTokenizer for TextPreTokenizer {
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>> {
        let text = String::from_utf8_lossy(data);

        let encoding = self
            .tokenizer
            .encode(text.as_ref(), false)
            .map_err(|e| anyhow::anyhow!("Encoding error: {}", e))?;
        let tokens = encoding.get_tokens();
        let offsets = encoding.get_offsets();

        let segments = tokens
            .iter()
            .zip(offsets.iter())
            .filter_map(|(token, (start, end))| {
                // Bounds check to handle edge cases
                if *start < data.len() && *end <= data.len() && start < end {
                    Some(ByteSegment {
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
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(segments)
    }

    fn modality(&self) -> &str {
        "text"
    }
}

/// Raw UTF-8 pre-tokenizer (True BLT style).
///
/// Segments text at the character level, preserving exact byte boundaries
/// for UTF-8 encoded characters. This is the canonical BLT approach where
/// each character becomes a separate token for the entropy model.
pub struct RawTextPreTokenizer;

impl ModalityPreTokenizer for RawTextPreTokenizer {
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>> {
        let text = String::from_utf8_lossy(data);
        let mut segments = Vec::new();
        let mut byte_offset = 0;

        for ch in text.chars() {
            let char_len = ch.len_utf8();

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
                        "char": ch.to_string()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_text_ascii() {
        let pretok = RawTextPreTokenizer;
        let data = b"Hello";
        let segments = pretok.pre_tokenize(data).unwrap();

        assert_eq!(segments.len(), 5);
        assert_eq!(segments[0].bytes, b"H");
        assert_eq!(segments[4].bytes, b"o");
    }

    #[test]
    fn test_raw_text_utf8() {
        let pretok = RawTextPreTokenizer;
        let data = "こんにちは".as_bytes(); // 5 characters, 15 bytes
        let segments = pretok.pre_tokenize(data).unwrap();

        assert_eq!(segments.len(), 5);
        // Each hiragana character is 3 bytes in UTF-8
        assert_eq!(segments[0].bytes.len(), 3);
    }
}

