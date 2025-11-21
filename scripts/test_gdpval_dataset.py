#!/usr/bin/env python3
"""Test loading the OpenAI GDPval dataset from Hugging Face."""

from datasets import load_dataset
import json

def main():
    print("Loading OpenAI GDPval dataset...")
    
    # Load the dataset
    dataset = load_dataset("openai/gdpval", split="train", streaming=True)
    
    # Look at first few examples
    print("\nFirst 3 examples from GDPval:\n")
    
    for i, example in enumerate(dataset.take(3)):
        print(f"--- Example {i+1} ---")
        print(f"Task ID: {example['task_id']}")
        print(f"Sector: {example['sector']}")
        print(f"Occupation: {example['occupation']}")
        print(f"Prompt length: {len(example['prompt'])} chars")
        print(f"Reference files: {len(example['reference_files'])} files")
        
        if example['reference_file_urls']:
            print(f"First reference URL: {example['reference_file_urls'][0]}")
        
        print(f"\nPrompt preview (first 200 chars):")
        print(example['prompt'][:200] + "...")
        print()

    # Show the structure
    print("\nDataset structure:")
    example = next(iter(dataset.take(1)))
    for key, value in example.items():
        if isinstance(value, list):
            print(f"  {key}: list of length {len(value)}")
        elif isinstance(value, str):
            print(f"  {key}: string of length {len(value)}")
        else:
            print(f"  {key}: {type(value)}")

if __name__ == "__main__":
    main()
