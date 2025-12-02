//! Binary file pre-tokenization using Goblin.
//!
//! Parses ELF, PE, and Mach-O binaries to extract sections as segments.
//! Each section becomes a separate ByteSegment with metadata about its
//! type and purpose.
//!
//! # Supported Formats
//!
//! - ELF (Linux, BSD, etc.) - Full section extraction
//! - PE (Windows) - Full section extraction via Goblin
//! - Mach-O (macOS, iOS) - Full section extraction via Goblin

use anyhow::Result;
use goblin::elf::Elf;
use goblin::mach::Mach;
use goblin::pe::PE;
use goblin::Object;

use super::{ByteSegment, ModalityPreTokenizer, SegmentMetadata};

/// Binary pre-tokenizer using Goblin for executable analysis.
///
/// Extracts sections from ELF, PE, and Mach-O binaries.
pub struct BinaryPreTokenizer;

impl BinaryPreTokenizer {
    /// Extract sections from an ELF binary.
    fn extract_elf_sections(data: &[u8], elf: &Elf) -> Result<Vec<ByteSegment>> {
        let mut segments = Vec::new();

        for section in &elf.section_headers {
            let start = section.sh_offset as usize;
            let size = section.sh_size as usize;
            let end = start + size;

            // Skip empty sections or sections beyond data bounds
            if size == 0 || end > data.len() {
                continue;
            }

            // Get section name from string table
            let name = elf.shdr_strtab.get_at(section.sh_name).unwrap_or("unknown");

            // Classify section type
            let section_type = match section.sh_type {
                goblin::elf::section_header::SHT_PROGBITS => "progbits",
                goblin::elf::section_header::SHT_SYMTAB => "symtab",
                goblin::elf::section_header::SHT_STRTAB => "strtab",
                goblin::elf::section_header::SHT_RELA => "rela",
                goblin::elf::section_header::SHT_HASH => "hash",
                goblin::elf::section_header::SHT_DYNAMIC => "dynamic",
                goblin::elf::section_header::SHT_NOTE => "note",
                goblin::elf::section_header::SHT_NOBITS => "nobits",
                goblin::elf::section_header::SHT_REL => "rel",
                goblin::elf::section_header::SHT_DYNSYM => "dynsym",
                _ => "other",
            };

            // Determine if section is executable, writable, etc.
            let flags = section.sh_flags;
            let is_executable = flags & goblin::elf::section_header::SHF_EXECINSTR as u64 != 0;
            let is_writable = flags & goblin::elf::section_header::SHF_WRITE as u64 != 0;
            let is_alloc = flags & goblin::elf::section_header::SHF_ALLOC as u64 != 0;

            segments.push(ByteSegment {
                bytes: data[start..end].to_vec(),
                label: Some(format!("elf_section_{name}")),
                metadata: Some(SegmentMetadata {
                    start_offset: start,
                    end_offset: end,
                    confidence: 1.0,
                    extra: Some(serde_json::json!({
                        "section_name": name,
                        "section_type": section_type,
                        "section_type_raw": section.sh_type,
                        "virtual_address": section.sh_addr,
                        "alignment": section.sh_addralign,
                        "flags": {
                            "executable": is_executable,
                            "writable": is_writable,
                            "allocatable": is_alloc
                        }
                    })),
                }),
            });
        }

        Ok(segments)
    }

    /// Extract sections from a PE (Windows) binary.
    fn extract_pe_sections(data: &[u8], pe: &PE) -> Result<Vec<ByteSegment>> {
        let mut segments = Vec::new();

        for section in &pe.sections {
            // Get section name (fixed 8-byte null-padded)
            let name_bytes = &section.name;
            let name = std::str::from_utf8(name_bytes)
                .unwrap_or("unknown")
                .trim_matches('\0')
                .to_string();

            // Skip empty or nameless sections
            if name.is_empty() || section.size_of_raw_data == 0 {
                continue;
            }

            let start = section.pointer_to_raw_data as usize;
            let size = section.size_of_raw_data as usize;
            let end = start + size;

            // Bounds check
            if end > data.len() {
                continue;
            }

            // Parse characteristics flags
            let characteristics = section.characteristics;
            let is_executable = characteristics & 0x2000_0000 != 0; // IMAGE_SCN_MEM_EXECUTE
            let is_readable = characteristics & 0x4000_0000 != 0; // IMAGE_SCN_MEM_READ
            let is_writable = characteristics & 0x8000_0000 != 0; // IMAGE_SCN_MEM_WRITE
            let is_code = characteristics & 0x0000_0020 != 0; // IMAGE_SCN_CNT_CODE
            let is_initialized_data = characteristics & 0x0000_0040 != 0; // IMAGE_SCN_CNT_INITIALIZED_DATA
            let is_uninitialized_data = characteristics & 0x0000_0080 != 0; // IMAGE_SCN_CNT_UNINITIALIZED_DATA

            segments.push(ByteSegment {
                bytes: data[start..end].to_vec(),
                label: Some(format!("pe_section_{name}")),
                metadata: Some(SegmentMetadata {
                    start_offset: start,
                    end_offset: end,
                    confidence: 1.0,
                    extra: Some(serde_json::json!({
                        "section_name": name,
                        "virtual_address": section.virtual_address,
                        "virtual_size": section.virtual_size,
                        "size": size,
                        "characteristics": characteristics,
                        "flags": {
                            "executable": is_executable,
                            "readable": is_readable,
                            "writable": is_writable,
                            "code": is_code,
                            "initialized_data": is_initialized_data,
                            "uninitialized_data": is_uninitialized_data
                        }
                    })),
                }),
            });
        }

        // If no sections found, return the whole binary as a single segment
        if segments.is_empty() {
            segments.push(ByteSegment {
                bytes: data.to_vec(),
                label: Some("pe_binary_raw".to_string()),
                metadata: Some(SegmentMetadata {
                    start_offset: 0,
                    end_offset: data.len(),
                    confidence: 0.7,
                    extra: Some(serde_json::json!({
                        "format": "PE",
                        "reason": "no_valid_sections_found"
                    })),
                }),
            });
        }

        Ok(segments)
    }

    /// Extract sections from a Mach-O (macOS/iOS) binary.
    fn extract_macho_sections(data: &[u8], mach: &Mach) -> Result<Vec<ByteSegment>> {
        match mach {
            Mach::Binary(macho) => Self::extract_single_macho(data, macho),
            Mach::Fat(fat) => {
                // For fat (universal) binaries, extract from the first architecture
                // In production, you might want to select based on target arch
                if let Some(arch) = fat.arches().ok().and_then(|arches| arches.first().copied()) {
                    let start = arch.offset as usize;
                    let end = start + arch.size as usize;
                    if end <= data.len() {
                        if let Ok(Object::Mach(Mach::Binary(inner))) =
                            Object::parse(&data[start..end])
                        {
                            return Self::extract_single_macho(&data[start..end], &inner);
                        }
                    }
                }
                // Fallback: return whole binary as single segment
                Ok(vec![ByteSegment {
                    bytes: data.to_vec(),
                    label: Some("macho_fat_binary".to_string()),
                    metadata: Some(SegmentMetadata {
                        start_offset: 0,
                        end_offset: data.len(),
                        confidence: 0.6,
                        extra: Some(serde_json::json!({
                            "format": "Mach-O Fat",
                            "reason": "fat_binary_extraction_fallback"
                        })),
                    }),
                }])
            }
        }
    }

    /// Extract sections from a single (non-fat) Mach-O binary.
    fn extract_single_macho(data: &[u8], macho: &goblin::mach::MachO) -> Result<Vec<ByteSegment>> {
        let mut segments_out = Vec::new();

        for segment in &macho.segments {
            // Get segment name
            let seg_name = segment.name().unwrap_or("__UNKNOWN");

            for (section, section_data) in segment.sections().unwrap_or_default() {
                // Get section name
                let sect_name = section.name().unwrap_or("unknown");

                // Skip empty sections
                if section_data.is_empty() {
                    continue;
                }

                // Calculate offset in file
                let start = section.offset as usize;
                let end = start + section.size as usize;

                // Parse section flags
                let flags = section.flags;
                let section_type = flags & 0xFF; // Lower 8 bits = type

                segments_out.push(ByteSegment {
                    bytes: section_data.to_vec(),
                    label: Some(format!("macho_section_{seg_name}_{sect_name}")),
                    metadata: Some(SegmentMetadata {
                        start_offset: start,
                        end_offset: end,
                        confidence: 1.0,
                        extra: Some(serde_json::json!({
                            "segment_name": seg_name,
                            "section_name": sect_name,
                            "address": section.addr,
                            "size": section.size,
                            "offset": section.offset,
                            "section_type": section_type,
                            "flags": flags
                        })),
                    }),
                });
            }
        }

        // If no sections found, return the whole binary as a single segment
        if segments_out.is_empty() {
            segments_out.push(ByteSegment {
                bytes: data.to_vec(),
                label: Some("macho_binary_raw".to_string()),
                metadata: Some(SegmentMetadata {
                    start_offset: 0,
                    end_offset: data.len(),
                    confidence: 0.7,
                    extra: Some(serde_json::json!({
                        "format": "Mach-O",
                        "reason": "no_valid_sections_found"
                    })),
                }),
            });
        }

        Ok(segments_out)
    }
}

impl ModalityPreTokenizer for BinaryPreTokenizer {
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>> {
        // Parse the binary to determine its type
        match Object::parse(data) {
            Ok(Object::Elf(elf)) => Self::extract_elf_sections(data, &elf),
            Ok(Object::PE(pe)) => Self::extract_pe_sections(data, &pe),
            Ok(Object::Mach(mach)) => Self::extract_macho_sections(data, &mach),
            Ok(Object::Archive(_)) => Ok(vec![ByteSegment {
                bytes: data.to_vec(),
                label: Some("archive".to_string()),
                metadata: Some(SegmentMetadata {
                    start_offset: 0,
                    end_offset: data.len(),
                    confidence: 0.5,
                    extra: Some(serde_json::json!({
                        "format": "Archive",
                        "status": "archive_extraction_not_implemented"
                    })),
                }),
            }]),
            Ok(Object::Unknown(_)) | Ok(_) | Err(_) => {
                // Unknown binary format - return as raw bytes
                Ok(vec![ByteSegment {
                    bytes: data.to_vec(),
                    label: Some("unknown_binary".to_string()),
                    metadata: Some(SegmentMetadata {
                        start_offset: 0,
                        end_offset: data.len(),
                        confidence: 0.3,
                        extra: Some(serde_json::json!({
                            "format": "unknown",
                            "reason": "failed_to_parse_or_unsupported"
                        })),
                    }),
                }])
            }
        }
    }

    fn modality(&self) -> &'static str {
        "binary"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_pretokenizer() {
        let pretok = BinaryPreTokenizer;
        assert_eq!(pretok.modality(), "binary");
    }

    #[test]
    fn test_unknown_binary() {
        let pretok = BinaryPreTokenizer;
        let data = b"not a valid binary";
        let segments = pretok.pre_tokenize(data).unwrap();

        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].label, Some("unknown_binary".to_string()));
    }

    #[test]
    fn test_pe_magic_detection() {
        // PE DOS header starts with "MZ"
        let pe_header: &[u8] = b"MZ";
        let pretok = BinaryPreTokenizer;
        // This should parse as PE (or fail gracefully) but not panic
        let _ = pretok.pre_tokenize(pe_header);
    }

    #[test]
    fn test_macho_magic_detection() {
        // Mach-O 64-bit magic (little-endian): 0xFEEDFACF
        let macho_header: [u8; 4] = [0xCF, 0xFA, 0xED, 0xFE];
        let pretok = BinaryPreTokenizer;
        // This should parse as Mach-O (or fail gracefully) but not panic
        let _ = pretok.pre_tokenize(&macho_header);
    }
}
