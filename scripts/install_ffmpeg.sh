#!/bin/bash
# Install FFmpeg on various platforms (automatic installation for BLT-Burn)
# Installs both the binary AND development headers needed for Rust bindings

set -e

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew is not installed. Please install it first:" >&2
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"" >&2
        exit 1
    fi
    
    # Determine Homebrew prefix (different for Intel vs Apple Silicon)
    BREW_PREFIX=$(brew --prefix)
    echo "ℹ️  Homebrew prefix: $BREW_PREFIX" >&2
    
    # Install pkg-config if missing (required by ffmpeg-sys-next)
    if ! command -v pkg-config &> /dev/null; then
        echo "📦 Installing pkg-config via Homebrew..." >&2
        if ! brew install pkg-config; then
            echo "❌ pkg-config installation failed" >&2
            exit 1
        fi
    fi
    
    # Install FFmpeg if missing
    if ! command -v ffmpeg &> /dev/null; then
        echo "📦 Installing FFmpeg via Homebrew..." >&2
        if ! brew install ffmpeg; then
            echo "❌ FFmpeg installation failed" >&2
            exit 1
        fi
    fi
    
    # Set up PKG_CONFIG_PATH for Homebrew FFmpeg (needed for ffmpeg-sys-next)
    FFMPEG_PKG_PATH="$BREW_PREFIX/opt/ffmpeg/lib/pkgconfig"
    if [ -d "$FFMPEG_PKG_PATH" ]; then
        echo "✅ FFmpeg pkg-config path: $FFMPEG_PKG_PATH" >&2
        
        # Check if pkg-config can find libavcodec
        export PKG_CONFIG_PATH="$FFMPEG_PKG_PATH:$PKG_CONFIG_PATH"
        if pkg-config --exists libavcodec 2>/dev/null; then
            echo "✅ pkg-config can find FFmpeg libraries" >&2
        else
            echo "⚠️  pkg-config cannot find FFmpeg. Add to your shell profile:" >&2
            echo "   export PKG_CONFIG_PATH=\"$FFMPEG_PKG_PATH:\$PKG_CONFIG_PATH\"" >&2
        fi
        
        # Print instructions for shell profile
        echo "" >&2
        echo "📋 Add these lines to your ~/.zshrc or ~/.bashrc:" >&2
        echo "   export PKG_CONFIG_PATH=\"$FFMPEG_PKG_PATH:\$PKG_CONFIG_PATH\"" >&2
        echo "   export FFMPEG_DIR=\"$BREW_PREFIX/opt/ffmpeg\"" >&2
    else
        echo "⚠️  FFmpeg pkgconfig not found at expected location" >&2
        echo "   Try: brew reinstall ffmpeg" >&2
    fi
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    # Check if FFmpeg is already installed
    if command -v ffmpeg &> /dev/null; then
        exit 0
    fi
    
    # Detect distro
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
    else
        echo "❌ Cannot detect Linux distribution" >&2
        exit 1
    fi
    
    if [[ $OS == *"Ubuntu"* ]] || [[ $OS == *"Debian"* ]]; then
        echo "📦 Installing FFmpeg with development headers via apt..." >&2
        if ! sudo apt update && sudo apt install -y ffmpeg libavcodec-dev libavformat-dev libswscale-dev libavutil-dev libavfilter-dev libavdevice-dev pkg-config; then
            echo "❌ FFmpeg installation failed" >&2
            exit 1
        fi
    elif [[ $OS == *"Fedora"* ]] || [[ $OS == *"Red Hat"* ]] || [[ $OS == *"CentOS"* ]]; then
        echo "📦 Installing FFmpeg with development headers via dnf..." >&2
        if ! sudo dnf install -y ffmpeg ffmpeg-devel; then
            echo "❌ FFmpeg installation failed" >&2
            exit 1
        fi
    elif [[ $OS == *"Arch"* ]] || [[ $OS == *"Manjaro"* ]]; then
        echo "📦 Installing FFmpeg via pacman..." >&2
        # Arch packages include headers by default
        if ! sudo pacman -S --noconfirm ffmpeg; then
            echo "❌ FFmpeg installation failed" >&2
            exit 1
        fi
    else
        echo "❌ Unsupported Linux distribution: $OS" >&2
        echo "Please install FFmpeg manually using your package manager" >&2
        exit 1
    fi
    
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows
    # Check if FFmpeg is already installed
    if command -v ffmpeg &> /dev/null; then
        exit 0
    fi
    
    # Try winget first
    if command -v winget &> /dev/null; then
        echo "📦 Installing FFmpeg via winget..." >&2
        if winget install ffmpeg; then
            exit 0
        fi
    fi
    
    echo "❌ FFmpeg not found. Please install manually:" >&2
    echo "1. Download from: https://www.gyan.dev/ffmpeg/builds/" >&2
    echo "2. Extract to a folder (e.g., C:\\ffmpeg)" >&2
    echo "3. Add the bin folder to your PATH environment variable" >&2
    exit 1
else
    echo "❌ Unsupported operating system: $OSTYPE" >&2
    exit 1
fi

# Verify installation
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ FFmpeg installation completed but not found in PATH" >&2
    exit 1
fi
