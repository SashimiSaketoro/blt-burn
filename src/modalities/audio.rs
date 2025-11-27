//! Audio pre-tokenization using Symphonia (pure Rust).
//!
//! Decodes audio files (WAV, MP3, OGG, FLAC, etc.) and segments
//! them into fixed-size frames suitable for the entropy model.
//!
//! # Supported Formats
//!
//! Via Symphonia's pure-Rust codecs:
//! - WAV (PCM)
//! - MP3
//! - OGG Vorbis
//! - FLAC
//! - AAC (MP4/M4A)

use anyhow::Result;
use std::io::Cursor;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::CODEC_TYPE_NULL;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use super::{ByteSegment, ModalityPreTokenizer, SegmentMetadata};

/// Audio pre-tokenizer using Symphonia for pure-Rust decoding.
///
/// Segments audio into fixed-size frames of PCM samples.
pub struct AudioPreTokenizer {
    /// Number of samples per frame.
    frame_size: usize,
    /// Expected sample rate (for metadata, not resampling).
    sample_rate: u32,
}

impl AudioPreTokenizer {
    /// Create a new audio pre-tokenizer.
    ///
    /// # Arguments
    /// * `frame_size` - Number of samples per frame (e.g., 160 for 10ms at 16kHz)
    /// * `sample_rate` - Expected sample rate in Hz (used for metadata)
    pub fn new(frame_size: usize, sample_rate: u32) -> Self {
        Self {
            frame_size,
            sample_rate,
        }
    }
}

impl ModalityPreTokenizer for AudioPreTokenizer {
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>> {
        // Create media source from bytes
        let mss = MediaSourceStream::new(Box::new(Cursor::new(data.to_vec())), Default::default());

        // Probe the media source to identify format
        let probed = symphonia::default::get_probe()
            .format(
                &Hint::new(),
                mss,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .map_err(|e| anyhow::anyhow!("Failed to probe audio format: {}", e))?;

        let mut format = probed.format;

        // Find the first audio track
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| anyhow::anyhow!("No supported audio track found"))?;

        // Create decoder for the track
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &Default::default())
            .map_err(|e| anyhow::anyhow!("Failed to create decoder: {}", e))?;

        let track_id = track.id;
        let actual_sample_rate = track
            .codec_params
            .sample_rate
            .unwrap_or(self.sample_rate);
        let channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

        let mut segments = Vec::new();
        let mut byte_offset = 0;
        let mut frame_index = 0;

        // Decode loop
        loop {
            let packet = match format.next_packet() {
                Ok(packet) => packet,
                Err(symphonia::core::errors::Error::IoError(ref e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break; // End of stream
                }
                Err(e) => {
                    // Log but continue - some packets may be recoverable
                    eprintln!("Warning: Error reading audio packet: {}", e);
                    break;
                }
            };

            // Skip packets from other tracks
            if packet.track_id() != track_id {
                continue;
            }

            match decoder.decode(&packet) {
                Ok(decoded) => {
                    // Convert to 16-bit interleaved PCM
                    let spec = *decoded.spec();
                    let duration = decoded.capacity() as u64;
                    let mut sample_buf = SampleBuffer::<i16>::new(duration, spec);
                    sample_buf.copy_interleaved_ref(decoded);

                    let samples = sample_buf.samples();
                    // Convert i16 samples to bytes (little-endian)
                    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

                    // Chunk into frames
                    let bytes_per_frame = self.frame_size * 2; // i16 = 2 bytes per sample
                    for chunk in bytes.chunks(bytes_per_frame) {
                        // Only keep full frames
                        if chunk.len() == bytes_per_frame {
                            segments.push(ByteSegment {
                                bytes: chunk.to_vec(),
                                label: Some("audio_frame".to_string()),
                                metadata: Some(SegmentMetadata {
                                    start_offset: byte_offset,
                                    end_offset: byte_offset + chunk.len(),
                                    confidence: 1.0,
                                    extra: Some(serde_json::json!({
                                        "frame_index": frame_index,
                                        "sample_rate": actual_sample_rate,
                                        "channels": channels,
                                        "frame_size_samples": self.frame_size,
                                        "duration_ms": (self.frame_size as f32 / actual_sample_rate as f32) * 1000.0
                                    })),
                                }),
                            });
                            byte_offset += chunk.len();
                            frame_index += 1;
                        }
                    }
                }
                Err(symphonia::core::errors::Error::DecodeError(msg)) => {
                    // Decode errors are sometimes recoverable
                    eprintln!("Warning: Audio decode error (skipping): {}", msg);
                    continue;
                }
                Err(e) => {
                    eprintln!("Warning: Audio decode error: {}", e);
                    continue;
                }
            }
        }

        Ok(segments)
    }

    fn modality(&self) -> &str {
        "audio"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_pretokenizer_creation() {
        let pretok = AudioPreTokenizer::new(160, 16000);
        assert_eq!(pretok.frame_size, 160);
        assert_eq!(pretok.sample_rate, 16000);
        assert_eq!(pretok.modality(), "audio");
    }
}

