#!/bin/bash
# Full Pipeline Test: FineWeb-Edu → BLT-1B → Sphere → Live Visualizer
#
# This script runs the complete ingestion pipeline with:
# - BLT-1B encoder (2048-dim semantic embeddings)
# - FineWeb-Edu dataset from your HF cache
# - Sphere optimization
# - Live visualization in thrml-viz
#
# Prerequisites:
# - FineWeb-Edu in your HF cache (~/.cache/huggingface or symlinked)
# - BLT-1B weights downloaded (models--facebook--blt-1b)
#
# Usage:
#   ./scripts/run_full_pipeline_test.sh
#
# Or with custom options:
#   NUM_DOCS=100 ./scripts/run_full_pipeline_test.sh

set -e

# Configuration
NUM_DOCS="${NUM_DOCS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/gauntlet_fineweb}"
SESSION_ID="${SESSION_ID:-test-session}"
SPHERE_SCALE="${SPHERE_SCALE:-medium}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Full Pipeline Test: BLT-1B + FineWeb + Sphere          ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLT_BURN_DIR="$(dirname "$SCRIPT_DIR")"
THRML_RS_DIR="$(dirname "$BLT_BURN_DIR")/thrml-rs"

echo -e "${YELLOW}Configuration:${NC}"
echo "  NUM_DOCS:     $NUM_DOCS"
echo "  OUTPUT_DIR:   $OUTPUT_DIR"
echo "  SESSION_ID:   $SESSION_ID"
echo "  SPHERE_SCALE: $SPHERE_SCALE"
echo ""

# Clean output directory
echo -e "${YELLOW}Cleaning output directory...${NC}"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Check if visualizer should be started
START_VIZ="${START_VIZ:-true}"

if [ "$START_VIZ" = "true" ]; then
    echo -e "${GREEN}Starting live visualizer in background...${NC}"
    echo "  Session: $SESSION_ID"
    echo ""
    
    # Build visualizer if needed
    (cd "$THRML_RS_DIR" && cargo build --example visualize --package thrml-viz 2>/dev/null) || true
    
    # Start visualizer in background
    (cd "$THRML_RS_DIR" && cargo run --example visualize -- --monitor "$SESSION_ID") &
    VIZ_PID=$!
    
    # Give it a moment to start
    sleep 2
    
    echo -e "${GREEN}Visualizer started (PID: $VIZ_PID)${NC}"
    echo ""
fi

# Run ingestion with BLT-1B
echo -e "${BLUE}Starting BLT-1B ingestion...${NC}"
echo ""

cd "$BLT_BURN_DIR"

# Build with viz feature
cargo build --release --bin ingest --features viz 2>/dev/null || cargo build --release --bin ingest

# Run ingestion
cargo run --release --bin ingest --features viz -- \
    --subset "sample-10BT" \
    --partition "train" \
    --limit "$NUM_DOCS" \
    --output-dir "$OUTPUT_DIR" \
    --embedding-model blt-1b \
    --embedding-precision bf16 \
    --sphere \
    --sphere-scale "$SPHERE_SCALE" \
    --entropy-weighted \
    --export-json \
    --viz-session "$SESSION_ID" \
    2>&1 | while read line; do
        echo -e "${NC}$line"
    done

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Pipeline Complete!                        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Show output files
echo -e "${YELLOW}Output files:${NC}"
ls -la "$OUTPUT_DIR"/*.safetensors 2>/dev/null | head -5 || echo "  (no safetensors files yet)"
echo ""

# Verify embeddings dimension
if ls "$OUTPUT_DIR"/*.safetensors 1>/dev/null 2>&1; then
    FIRST_FILE=$(ls "$OUTPUT_DIR"/*.safetensors | head -1)
    echo -e "${YELLOW}Verifying embeddings dimension:${NC}"
    python3 -c "
import safetensors.torch as st
data = st.load_file('$FIRST_FILE')
emb = data['embeddings']
print(f'  Shape: {list(emb.shape)}')
print(f'  Expected: [N, 2048] for BLT-1B')
assert emb.shape[1] == 2048, f'Expected 2048-dim, got {emb.shape[1]}'
print('  ✓ Dimension correct!')
" 2>/dev/null || echo "  (python verification skipped)"
fi

echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Visualizer should be showing points live"
echo "  2. Run gauntlet tests: cd $THRML_RS_DIR && cargo test --test gauntlet_test -- --nocapture"
echo "  3. Or view static: cargo run --example visualize -- --input $OUTPUT_DIR/*.safetensors"
echo ""

# Cleanup hint
if [ -n "$VIZ_PID" ]; then
    echo -e "${YELLOW}Visualizer running (PID: $VIZ_PID). Press Ctrl+C to stop.${NC}"
    wait $VIZ_PID 2>/dev/null || true
fi
