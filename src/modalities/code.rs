//! Code pre-tokenization using Tree-sitter for AST-aware segmentation.
//!
//! Parses source code into an Abstract Syntax Tree and extracts
//! semantic units (functions, classes, imports) as segments.
//!
//! # Supported Languages
//!
//! - Rust (via tree-sitter-rust)
//! - Python (via tree-sitter-python)
//!
//! Additional languages can be added by including the appropriate
//! tree-sitter grammar crate.

use anyhow::Result;
use tree_sitter::{Language, Parser};

use super::{ByteSegment, ModalityPreTokenizer, SegmentMetadata};

/// Code pre-tokenizer using Tree-sitter for AST-aware segmentation.
///
/// Extracts semantic units (functions, classes, modules) as segments,
/// preserving code structure for the entropy model.
pub struct CodePreTokenizer {
    /// Programming language identifier (e.g., "rust", "python").
    language: String,
}

impl CodePreTokenizer {
    /// Create a new code pre-tokenizer for the specified language.
    ///
    /// # Arguments
    /// * `language` - Language identifier: "rust", "python"
    pub fn new(language: String) -> Self {
        Self { language }
    }

    /// Get the tree-sitter Language for parsing.
    fn get_language(&self) -> Result<Language> {
        match self.language.as_str() {
            "rust" => Ok(tree_sitter_rust::LANGUAGE.into()),
            "python" => Ok(tree_sitter_python::LANGUAGE.into()),
            _ => Err(anyhow::anyhow!(
                "Unsupported language for AST parsing: {}. Supported: rust, python",
                self.language
            )),
        }
    }

    /// Check if a node kind represents a semantic unit we want to extract.
    fn is_semantic_unit(&self, kind: &str) -> bool {
        match self.language.as_str() {
            "rust" => matches!(
                kind,
                "function_item"
                    | "struct_item"
                    | "enum_item"
                    | "impl_item"
                    | "mod_item"
                    | "trait_item"
                    | "use_declaration"
                    | "macro_definition"
            ),
            "python" => matches!(
                kind,
                "function_definition"
                    | "class_definition"
                    | "import_statement"
                    | "import_from_statement"
                    | "decorated_definition"
            ),
            _ => false,
        }
    }
}

impl ModalityPreTokenizer for CodePreTokenizer {
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>> {
        let mut parser = Parser::new();

        let lang = self.get_language()?;
        parser
            .set_language(&lang)
            .map_err(|e| anyhow::anyhow!("Failed to set language: {e}"))?;

        // Parse the source code
        let tree = parser
            .parse(data, None)
            .ok_or_else(|| anyhow::anyhow!("Failed to parse code"))?;

        let root = tree.root_node();
        let mut segments = Vec::new();
        let mut cursor = root.walk();

        // Traverse the tree looking for semantic units
        let mut reached_root = false;
        while !reached_root {
            let node = cursor.node();
            let kind = node.kind();

            if self.is_semantic_unit(kind) {
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
                                "language": self.language,
                                "start_row": node.start_position().row,
                                "start_col": node.start_position().column,
                                "end_row": node.end_position().row,
                                "end_col": node.end_position().column,
                                "is_named": node.is_named()
                            })),
                        }),
                    });
                }

                // Skip children of this node (we captured the whole unit)
                if !cursor.goto_next_sibling() {
                    // Go up and try next sibling
                    while !cursor.goto_next_sibling() {
                        if !cursor.goto_parent() {
                            reached_root = true;
                            break;
                        }
                    }
                }
            } else {
                // Not a semantic unit, descend into children
                if !cursor.goto_first_child() {
                    // No children, try next sibling
                    if !cursor.goto_next_sibling() {
                        // No sibling, go up
                        while !cursor.goto_next_sibling() {
                            if !cursor.goto_parent() {
                                reached_root = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        // If no semantic units found, return the entire code as a single segment
        if segments.is_empty() {
            segments.push(ByteSegment {
                bytes: data.to_vec(),
                label: Some("code_snippet".to_string()),
                metadata: Some(SegmentMetadata {
                    start_offset: 0,
                    end_offset: data.len(),
                    confidence: 0.5, // Lower confidence for fallback
                    extra: Some(serde_json::json!({
                        "language": self.language,
                        "fallback": true,
                        "reason": "no_semantic_units_found"
                    })),
                }),
            });
        }

        Ok(segments)
    }

    fn modality(&self) -> &'static str {
        "code_ast"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rust_function_extraction() {
        let pretok = CodePreTokenizer::new("rust".to_string());
        let code = b"fn hello() { println!(\"Hello\"); }";
        let segments = pretok.pre_tokenize(code).unwrap();

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].label, Some("function_item".to_string()));
    }

    #[test]
    fn test_python_function_extraction() {
        let pretok = CodePreTokenizer::new("python".to_string());
        let code = b"def hello():\n    print('Hello')";
        let segments = pretok.pre_tokenize(code).unwrap();

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].label, Some("function_definition".to_string()));
    }

    #[test]
    fn test_unsupported_language() {
        let pretok = CodePreTokenizer::new("javascript".to_string());
        let code = b"function hello() {}";
        let result = pretok.pre_tokenize(code);

        assert!(result.is_err());
    }
}
