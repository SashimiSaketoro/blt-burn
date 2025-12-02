//! Video pre-tokenization using FFmpeg.
//!
//! Extracts video frames as RGB segments for the entropy model.
//!
//! # Requirements
//!
//! - FFmpeg must be installed on the system
//! - Build with `--features video` to enable
//!
//! # Status
//!
//! **Beta**: Full frame extraction with RGB pixel data. Supports H.264/AV1/etc. via FFmpeg.
//! Configurable sampling; scales frames to 224x224 for consistency.

use anyhow::Result;

use super::{ByteSegment, ModalityPreTokenizer};

#[cfg(feature = "video")]
use super::SegmentMetadata;

/// Target output dimensions for frames (ViT standard)
#[cfg(feature = "video")]
const TARGET_WIDTH: u32 = 224;
#[cfg(feature = "video")]
const TARGET_HEIGHT: u32 = 224;
/// Bytes per pixel for RGB24 format
#[cfg(feature = "video")]
const RGB_BYTES_PER_PIXEL: usize = 3;

/// Video pre-tokenizer using FFmpeg via ffmpeg-next.
///
/// Extracts keyframes at configurable intervals for downstream processing.
/// Frames are scaled to 224x224 RGB24 format for consistent downstream embedding.
pub struct VideoPreTokenizer {
    /// Target frame rate for extraction.
    #[allow(dead_code)]
    frame_rate: u32,
    /// Sample every Nth frame (default: 2 = every other frame)
    #[allow(dead_code)]
    sample_interval: usize,
}

impl VideoPreTokenizer {
    /// Create a new video pre-tokenizer with default settings.
    ///
    /// # Arguments
    /// * `frame_rate` - Target frame rate for extraction (e.g., 30 = 1 frame per 1/30s)
    pub fn new(frame_rate: u32) -> Self {
        Self {
            frame_rate,
            sample_interval: 2, // Sample every other frame by default
        }
    }

    /// Create a new video pre-tokenizer with custom sampling interval.
    ///
    /// # Arguments
    /// * `frame_rate` - Target frame rate for extraction
    /// * `sample_interval` - Sample every Nth frame (1 = all frames, 2 = every other, etc.)
    pub fn with_interval(frame_rate: u32, sample_interval: usize) -> Self {
        Self {
            frame_rate,
            sample_interval: sample_interval.max(1), // Ensure at least 1
        }
    }

    /// Extract frames using FFmpeg with RGB24 conversion.
    ///
    /// Frames are scaled to 224x224 RGB24 format for consistent downstream processing.
    /// Only available when compiled with the `video` feature.
    #[cfg(feature = "video")]
    fn extract_with_ffmpeg(&self, data: &[u8]) -> Result<Vec<ByteSegment>> {
        use ffmpeg_next as ffmpeg;
        use ffmpeg_next::format::Pixel;
        use ffmpeg_next::software::scaling::{context::Context as ScalerContext, flag::Flags};
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Write to temp file for FFmpeg processing
        let mut temp = NamedTempFile::new()?;
        temp.write_all(data)?;
        let path = temp.into_temp_path();

        ffmpeg::init().map_err(|e| anyhow::anyhow!("FFmpeg init failed: {:?}", e))?;

        let mut format_context = ffmpeg::format::input(&path)?;
        let (video_stream_index, time_base) = {
            let video_stream = format_context
                .streams()
                .best(ffmpeg::media::Type::Video)
                .ok_or_else(|| anyhow::anyhow!("No video stream found"))?;
            (video_stream.index(), video_stream.time_base())
        };

        let video_stream = format_context
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| anyhow::anyhow!("No video stream found"))?;

        let decoder_context =
            ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())?;
        let mut decoder = decoder_context.decoder().video()?;

        // Create scaler for RGB24 output at target dimensions (lazily initialized)
        let mut scaler: Option<ScalerContext> = None;

        let mut segments = Vec::new();
        let mut frame_count = 0usize;
        let sample_interval = self.sample_interval;

        let mut packet_iter = format_context.packets();
        while let Some((stream, packet)) = packet_iter.next() {
            if stream.index() == video_stream_index {
                decoder.send_packet(&packet)?;

                let mut frame = ffmpeg::frame::Video::empty();
                while decoder.receive_frame(&mut frame).is_ok() {
                    let orig_width = frame.width();
                    let orig_height = frame.height();
                    let timestamp = frame.timestamp().unwrap_or(0) as f64
                        / time_base.denominator() as f64
                        * time_base.numerator() as f64;

                    // Only sample at intervals to reduce output size
                    if frame_count % sample_interval == 0 {
                        // Initialize scaler on first frame (we need the source format)
                        if scaler.is_none() {
                            scaler = Some(
                                ScalerContext::get(
                                    frame.format(),
                                    orig_width,
                                    orig_height,
                                    Pixel::RGB24,
                                    TARGET_WIDTH,
                                    TARGET_HEIGHT,
                                    Flags::BILINEAR,
                                )
                                .map_err(|e| anyhow::anyhow!("Scaler init failed: {}", e))?,
                            );
                        }

                        // Scale and convert to RGB24
                        let mut rgb_frame = ffmpeg::frame::Video::empty();
                        if let Some(ref mut scaler_ctx) = scaler {
                            scaler_ctx
                                .run(&frame, &mut rgb_frame)
                                .map_err(|e| anyhow::anyhow!("Scaling failed: {}", e))?;
                        }

                        // Extract RGB bytes from plane 0 (packed RGB24)
                        // Expected size: 224 * 224 * 3 = 150528 bytes
                        let raw_bytes = rgb_frame.data(0).to_vec();

                        // Calculate timestamp in milliseconds for offset
                        let timestamp_ms = (timestamp * 1000.0) as usize;

                        segments.push(ByteSegment {
                            bytes: raw_bytes,
                            label: Some(format!("video_frame_{}", frame_count)),
                            metadata: Some(SegmentMetadata {
                                start_offset: timestamp_ms,
                                end_offset: timestamp_ms + 33, // ~1/30s in ms
                                confidence: 1.0,
                                extra: Some(serde_json::json!({
                                    "timestamp": timestamp,
                                    "original_width": orig_width,
                                    "original_height": orig_height,
                                    "width": TARGET_WIDTH,
                                    "height": TARGET_HEIGHT,
                                    "format": "RGB24",
                                    "frame_idx": frame_count,
                                    "source": "ffmpeg-next"
                                })),
                            }),
                        });
                    }
                    frame_count += 1;
                }
            }
        }

        // Flush decoder and process remaining frames
        decoder.send_eof()?;
        let mut frame = ffmpeg::frame::Video::empty();
        while decoder.receive_frame(&mut frame).is_ok() {
            // Skip flushed frames for now (could process them too if needed)
        }

        Ok(segments)
    }
}

#[cfg(feature = "video")]
impl ModalityPreTokenizer for VideoPreTokenizer {
    fn pre_tokenize(&self, data: &[u8]) -> Result<Vec<ByteSegment>> {
        self.extract_with_ffmpeg(data)
    }

    fn modality(&self) -> &'static str {
        "video"
    }
}

#[cfg(not(feature = "video"))]
impl ModalityPreTokenizer for VideoPreTokenizer {
    fn pre_tokenize(&self, _data: &[u8]) -> Result<Vec<ByteSegment>> {
        anyhow::bail!(
            "Video support not compiled. Build with --features video and ensure FFmpeg is installed."
        )
    }

    fn modality(&self) -> &'static str {
        "video"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_pretokenizer_creation() {
        let pretok = VideoPreTokenizer::new(30);
        assert_eq!(pretok.frame_rate, 30);
        assert_eq!(pretok.sample_interval, 2);
        assert_eq!(pretok.modality(), "video");
    }

    #[test]
    fn test_video_pretokenizer_with_interval() {
        let pretok = VideoPreTokenizer::with_interval(30, 5);
        assert_eq!(pretok.frame_rate, 30);
        assert_eq!(pretok.sample_interval, 5);
    }

    #[test]
    fn test_video_pretokenizer_interval_min() {
        // Ensure sample_interval is at least 1
        let pretok = VideoPreTokenizer::with_interval(30, 0);
        assert_eq!(pretok.sample_interval, 1);
    }

    #[cfg(not(feature = "video"))]
    #[test]
    fn test_video_disabled_error() {
        let pretok = VideoPreTokenizer::new(30);
        let result = pretok.pre_tokenize(&[]);
        assert!(result.is_err());
    }

    #[cfg(feature = "video")]
    #[test]
    fn test_rgb_frame_size() {
        // Verify expected RGB24 frame size: 224 * 224 * 3 = 150528 bytes
        let expected_size =
            (TARGET_WIDTH as usize) * (TARGET_HEIGHT as usize) * RGB_BYTES_PER_PIXEL;
        assert_eq!(expected_size, 150528);
    }
}
