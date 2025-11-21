#!/bin/bash
# Test the FFmpeg installation flow

echo "=== Testing FFmpeg Detection Flow ==="

# Save original PATH
ORIGINAL_PATH=$PATH

# Test 1: FFmpeg found on PATH
echo -e "\nTest 1: FFmpeg already installed"
PATH=$ORIGINAL_PATH
cargo run --release --bin ingest -- --text "Test" --output test_out --no-audio-video 2>&1 | grep -E "(ffmpeg|FFmpeg|video)"

# Test 2: FFmpeg NOT on PATH (simulating not installed)
echo -e "\nTest 2: FFmpeg not found (simulated)"
PATH=/usr/bin:/bin  # Remove homebrew paths
echo "Current PATH: $PATH"
echo "FFmpeg available: $(which ffmpeg 2>/dev/null || echo 'NOT FOUND')"

# This should trigger the interactive prompt
echo -e "\nRunning ingest without FFmpeg on PATH..."
# We'll pipe '1' to select "Continue without audio/video support"
echo "1" | cargo run --release --bin ingest -- --text "Test without FFmpeg" --output test_out_no_ffmpeg 2>&1 | grep -A10 -B5 -E "(ffmpeg|FFmpeg|video|audio|What would you like to do)"

# Test 3: Auto-install flag
echo -e "\nTest 3: Testing --auto-install-ffmpeg flag"
PATH=/usr/bin:/bin  # Keep FFmpeg unavailable
cargo run --release --bin ingest -- --text "Test auto" --output test_out_auto --auto-install-ffmpeg --no-audio-video 2>&1 | grep -E "(ffmpeg|FFmpeg|auto)"

# Restore PATH
PATH=$ORIGINAL_PATH

echo -e "\n=== FFmpeg Flow Tests Complete ===\n"
