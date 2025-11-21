use anyhow::Result;
use blt_burn::pretokenize::PreTokenizerType;

fn main() -> Result<()> {
    // Example 1: Text with simple whitespace tokenization
    println!("=== Text Pre-Tokenization (Whitespace) ===");
    let text_pretokenizer = PreTokenizerType::TextSimple.create()?;
    let text = b"The quick brown fox jumps over the lazy dog.";
    let segments = text_pretokenizer.pre_tokenize(text)?;
    println!("Input: {}", String::from_utf8_lossy(text));
    println!("Segments: {}", segments.len());
    for (i, seg) in segments.iter().take(5).enumerate() {
        println!(
            "  [{}] {:?} ({})",
            i,
            String::from_utf8_lossy(&seg.bytes),
            seg.bytes.len()
        );
    }

    // Example 2: Image patches
    println!("\n=== Image Pre-Tokenization (Patch-based) ===");
    let image_pretokenizer = PreTokenizerType::Image {
        patch_size: 196, // 14x14 pixels at 1 byte per pixel (grayscale)
        stride: 196,
    }
    .create()?;
    let fake_image = vec![0u8; 784]; // 28x28 grayscale image
    let segments = image_pretokenizer.pre_tokenize(&fake_image)?;
    println!("Input: {} bytes (28x28 image)", fake_image.len());
    println!("Patches: {}", segments.len());
    for (i, seg) in segments.iter().enumerate() {
        println!("  Patch {}: {} bytes", i, seg.bytes.len());
    }

    // Example 3: Audio frames
    println!("\n=== Audio Pre-Tokenization (Frame-based) ===");
    let audio_pretokenizer = PreTokenizerType::Audio {
        frame_size: 160, // 10ms at 16kHz (160 samples)
        sample_rate: 16000,
    }
    .create()?;
    let fake_audio = vec![0u8; 1600]; // 100ms of audio
    let segments = audio_pretokenizer.pre_tokenize(&fake_audio)?;
    println!("Input: {} bytes (100ms @ 16kHz)", fake_audio.len());
    println!("Frames: {}", segments.len());

    // Example 4: Code
    println!("\n=== Code Pre-Tokenization (Line-based) ===");
    let code_pretokenizer = PreTokenizerType::Code {
        language: "rust".to_string(),
    }
    .create()?;
    let code = b"fn main() {\n    println!(\"Hello, world!\");\n}\n";
    let segments = code_pretokenizer.pre_tokenize(code)?;
    println!("Input:\n{}", String::from_utf8_lossy(code));
    println!("Lines: {}", segments.len());
    for (i, seg) in segments.iter().enumerate() {
        println!(
            "  Line {}: {:?}",
            i + 1,
            String::from_utf8_lossy(&seg.bytes)
        );
    }

    println!("\n✅ All pre-tokenizers working!");
    println!("\nNext steps:");
    println!("- Feed these segments to the entropy model");
    println!("- Extract embeddings for each segment");
    println!("- Use water-filling to organize on the sphere");

    Ok(())
}
