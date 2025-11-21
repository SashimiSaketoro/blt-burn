#!/usr/bin/env python3
"""Test script to verify dataset structures and download functionality"""

import json
import sqlite3
import sys

def check_gdpval_structure():
    """Check the structure of GDPval dataset"""
    print("=== Checking GDPval dataset structure ===")
    
    db_path = "/Users/joemiles/.cache/burn-dataset/openaigdpval.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get first row to analyze structure
    cursor.execute("SELECT * FROM train LIMIT 1")
    columns = [desc[0] for desc in cursor.description]
    row = cursor.fetchone()
    
    print(f"Columns: {columns}")
    
    # Parse each field
    for i, (col, val) in enumerate(zip(columns, row)):
        print(f"\n{col} (type: {type(val).__name__}):")
        
        if col in ['reference_files', 'reference_file_urls', 'reference_file_hf_uris']:
            # These should be JSON arrays
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    print(f"  JSON array with {len(parsed)} items")
                    if parsed:
                        print(f"  First item: {parsed[0]}")
                except:
                    print(f"  Not JSON, raw value: {repr(val[:100])}")
            else:
                print(f"  Binary data: {len(val) if val else 0} bytes")
        else:
            if isinstance(val, str):
                print(f"  Value: {val[:200]}...")
            else:
                print(f"  Value: {val}")
    
    conn.close()

def check_treevgr_structure():
    """Check TreeVGR dataset structure from a sample"""
    print("\n=== TreeVGR dataset structure ===")
    
    sample = {
        "images": ["images/ai2d/abc_images/426.png"],
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat is a carnivore?\nA. plant\nB. sun\nC. praying mantis\nD. snake"
            },
            {
                "from": "gpt", 
                "value": "<think>To determine a carnivore among the options, we analyze based on the food-web diagram. A carnivore is an organism that eats other animals. A plant <box>[0,74,60,136]</box> is a producer and makes its own food, so it's not a carnivore. The sun is not a living organism and thus not a carnivore. The praying mantis <box>[51,72,177,141]</box> eats the plant in the diagram, so it's a herbivore here. The snake <box>[195,115,273,155]</box> preys on the frog, an animal, which makes it a carnivore.</think> <answer>D</answer>"
            }
        ],
        "system": "A conversation between user and assistant..."
    }
    
    print("Sample structure:")
    print(f"- images: {sample['images']}")
    print(f"- conversations: {len(sample['conversations'])} turns")
    print(f"- system prompt: {sample['system'][:50]}...")
    
    # Extract bounding boxes
    gpt_response = sample['conversations'][1]['value']
    import re
    boxes = re.findall(r'<box>\[(\d+),(\d+),(\d+),(\d+)\]</box>', gpt_response)
    print(f"\nFound {len(boxes)} bounding boxes:")
    for box in boxes:
        print(f"  Box: x1={box[0]}, y1={box[1]}, x2={box[2]}, y2={box[3]}")

if __name__ == "__main__":
    check_gdpval_structure()
    check_treevgr_structure()
