#!/bin/bash
# Setup FFmpeg environment variables for building blt-burn with video feature
# Usage: source scripts/setup_ffmpeg_env.sh
#        cargo build --features video

set -e

echo "🔧 Setting up FFmpeg environment for blt-burn..."

# Detect OS and set up FFmpeg paths
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - find Homebrew FFmpeg
    if command -v brew &> /dev/null; then
        FFMPEG_PREFIX=$(brew --prefix ffmpeg 2>/dev/null || echo "")
        
        if [ -n "$FFMPEG_PREFIX" ] && [ -d "$FFMPEG_PREFIX" ]; then
            export FFMPEG_DIR="$FFMPEG_PREFIX"
            export PKG_CONFIG_PATH="$FFMPEG_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
            
            # For bindgen to find headers
            export BINDGEN_EXTRA_CLANG_ARGS="-I$FFMPEG_PREFIX/include"
            
            echo "✅ FFMPEG_DIR=$FFMPEG_DIR"
            echo "✅ PKG_CONFIG_PATH includes: $FFMPEG_PREFIX/lib/pkgconfig"
            echo "✅ BINDGEN_EXTRA_CLANG_ARGS=$BINDGEN_EXTRA_CLANG_ARGS"
            echo ""
            echo "🚀 Ready to build! Run:"
            echo "   cargo build --features video"
        else
            echo "❌ FFmpeg not found via Homebrew"
            echo "   Install with: brew install ffmpeg pkg-config"
            return 1 2>/dev/null || exit 1
        fi
    else
        echo "❌ Homebrew not found"
        echo "   Install Homebrew first: https://brew.sh"
        return 1 2>/dev/null || exit 1
    fi

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux - check if FFmpeg dev packages are installed
    if pkg-config --exists libavcodec 2>/dev/null; then
        echo "✅ FFmpeg development libraries found via pkg-config"
        
        # Get the include path from pkg-config
        FFMPEG_CFLAGS=$(pkg-config --cflags libavcodec libavformat libswscale 2>/dev/null || echo "")
        if [ -n "$FFMPEG_CFLAGS" ]; then
            export BINDGEN_EXTRA_CLANG_ARGS="$FFMPEG_CFLAGS"
            echo "✅ BINDGEN_EXTRA_CLANG_ARGS=$BINDGEN_EXTRA_CLANG_ARGS"
        fi
        
        echo ""
        echo "🚀 Ready to build! Run:"
        echo "   cargo build --features video"
    else
        echo "❌ FFmpeg development libraries not found"
        echo "   Install with:"
        echo "     Ubuntu/Debian: sudo apt install libavcodec-dev libavformat-dev libswscale-dev libavutil-dev pkg-config"
        echo "     Fedora: sudo dnf install ffmpeg-devel"
        echo "     Arch: sudo pacman -S ffmpeg"
        return 1 2>/dev/null || exit 1
    fi

elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows
    if [ -n "$FFMPEG_DIR" ] && [ -d "$FFMPEG_DIR" ]; then
        echo "✅ FFMPEG_DIR=$FFMPEG_DIR"
        export PATH="$FFMPEG_DIR/bin:$PATH"
        echo ""
        echo "🚀 Ready to build! Run:"
        echo "   cargo build --features video"
    else
        echo "❌ FFMPEG_DIR not set"
        echo "   1. Download FFmpeg from: https://www.gyan.dev/ffmpeg/builds/"
        echo "   2. Extract to a folder (e.g., C:\\ffmpeg)"
        echo "   3. Set: export FFMPEG_DIR=\"/c/ffmpeg\""
        return 1 2>/dev/null || exit 1
    fi
else
    echo "❌ Unsupported OS: $OSTYPE"
    return 1 2>/dev/null || exit 1
fi

