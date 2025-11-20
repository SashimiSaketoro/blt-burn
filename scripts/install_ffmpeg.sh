#!/bin/bash
# Install FFmpeg on various platforms

set -e

echo "🎥 BLT-Burn FFmpeg Installer"
echo "=========================="
echo

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Detected macOS"
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew is not installed. Please install it first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    echo "📦 Installing FFmpeg via Homebrew..."
    brew install ffmpeg
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    echo "Detected Linux"
    
    # Detect distro
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$NAME
    fi
    
    if [[ $OS == *"Ubuntu"* ]] || [[ $OS == *"Debian"* ]]; then
        echo "📦 Installing FFmpeg via apt..."
        sudo apt update
        sudo apt install -y ffmpeg
    elif [[ $OS == *"Fedora"* ]] || [[ $OS == *"Red Hat"* ]]; then
        echo "📦 Installing FFmpeg via dnf..."
        sudo dnf install -y ffmpeg
    elif [[ $OS == *"Arch"* ]] || [[ $OS == *"Manjaro"* ]]; then
        echo "📦 Installing FFmpeg via pacman..."
        sudo pacman -S --noconfirm ffmpeg
    else
        echo "❌ Unsupported Linux distribution: $OS"
        echo "Please install FFmpeg manually using your package manager"
        exit 1
    fi
    
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    # Windows
    echo "Detected Windows"
    echo
    echo "Please install FFmpeg manually:"
    echo "1. Download from: https://www.gyan.dev/ffmpeg/builds/"
    echo "2. Extract to a folder (e.g., C:\\ffmpeg)"
    echo "3. Add the bin folder to your PATH environment variable"
    echo
    echo "Or use Windows Package Manager (winget):"
    echo "   winget install ffmpeg"
    exit 1
else
    echo "❌ Unsupported operating system: $OSTYPE"
    exit 1
fi

# Verify installation
if command -v ffmpeg &> /dev/null; then
    echo
    echo "✅ FFmpeg installed successfully!"
    echo "Version: $(ffmpeg -version | head -n1)"
else
    echo "❌ FFmpeg installation failed"
    exit 1
fi
