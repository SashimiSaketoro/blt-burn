#!/bin/bash
# Install FFmpeg on various platforms (automatic installation for BLT-Burn)

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
    
    # Check if FFmpeg is already installed
    if command -v ffmpeg &> /dev/null && command -v pkg-config &> /dev/null; then
        exit 0
    fi
    
    # Install pkg-config if missing (required by ffmpeg-sys-next)
    if ! command -v pkg-config &> /dev/null; then
        echo "📦 Installing pkg-config via Homebrew..." >&2
        if ! brew install pkg-config; then
            echo "❌ pkg-config installation failed" >&2
            exit 1
        fi
    fi
    
    # Install FFmpeg
    if ! command -v ffmpeg &> /dev/null; then
        echo "📦 Installing FFmpeg via Homebrew..." >&2
        if ! brew install ffmpeg; then
            echo "❌ FFmpeg installation failed" >&2
            exit 1
        fi
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
        echo "📦 Installing FFmpeg via apt..." >&2
        if ! sudo apt update && sudo apt install -y ffmpeg; then
            echo "❌ FFmpeg installation failed" >&2
            exit 1
        fi
    elif [[ $OS == *"Fedora"* ]] || [[ $OS == *"Red Hat"* ]] || [[ $OS == *"CentOS"* ]]; then
        echo "📦 Installing FFmpeg via dnf..." >&2
        if ! sudo dnf install -y ffmpeg; then
            echo "❌ FFmpeg installation failed" >&2
            exit 1
        fi
    elif [[ $OS == *"Arch"* ]] || [[ $OS == *"Manjaro"* ]]; then
        echo "📦 Installing FFmpeg via pacman..." >&2
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
