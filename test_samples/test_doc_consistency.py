#!/usr/bin/env python3
"""
Test documentation consistency and accuracy.
"""

import re
from pathlib import Path

def check_version_consistency():
    """Check that version numbers are consistent across docs."""
    print("Checking version consistency...")
    
    version_pattern = r"(\d+\.\d+\.\d+)"
    versions_found = {}
    
    doc_files = [
        "README.md",
        "docs/README.md", 
        "docs/API_REFERENCE.md"
    ]
    
    for doc_file in doc_files:
        path = Path(doc_file)
        if path.exists():
            content = path.read_text()
            # Look for version patterns
            matches = re.findall(r"[Vv]ersion[:\s]+(\d+\.\d+\.\d+)", content)
            if matches:
                versions_found[doc_file] = matches
    
    print(f"  Versions found: {versions_found}")
    
    # Check they all match
    all_versions = [v for versions in versions_found.values() for v in versions]
    if len(set(all_versions)) > 1:
        print(f"  ⚠️  Inconsistent versions: {set(all_versions)}")
    else:
        print("  ✓ All versions consistent")


def check_threshold_values():
    """Check that threshold values are documented correctly."""
    print("\nChecking threshold documentation...")
    
    # Check that 1.35 is documented as default
    readme = Path("README.md").read_text()
    api_ref = Path("docs/API_REFERENCE.md").read_text()
    
    if "1.35" in api_ref and "default" in api_ref:
        print("  ✓ Default threshold 1.35 documented")
    else:
        print("  ⚠️  Default threshold not properly documented")
    
    if "1.55" in api_ref:
        print("  ✓ Alternative threshold 1.55 mentioned")
    else:
        print("  ⚠️  Alternative threshold 1.55 not mentioned")


def check_removed_claims():
    """Check that misleading claims have been removed."""
    print("\nChecking for removed misleading claims...")
    
    bad_patterns = [
        r"37.*billion.*times",  # 37 billion times claim
        r"190MB.*380MB",        # Incorrect size comparison
        r"100-200 tokens/sec",  # Made up performance metrics
        r"converges.*12.*iterations",  # Unverified convergence
        r"Hollow Core",         # Hypothetical topology
        r"Infinite Crust"       # Hypothetical topology
    ]
    
    all_docs = []
    for doc_path in Path("docs").rglob("*.md"):
        all_docs.append(doc_path)
    all_docs.append(Path("README.md"))
    
    issues = []
    for doc_path in all_docs:
        if doc_path.exists():
            content = doc_path.read_text().lower()
            for pattern in bad_patterns:
                if re.search(pattern.lower(), content):
                    issues.append(f"{doc_path}: found pattern '{pattern}'")
    
    if issues:
        print("  ⚠️  Found potentially misleading content:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ✓ No misleading claims found")


def check_new_features():
    """Check that new v0.2 features are documented."""
    print("\nChecking v0.2 feature documentation...")
    
    features = {
        "JAX sharding": ["sharding", "JAX", "distributed"],
        "SQLite storage": ["SQLite", "rusqlite", ".hypergraph.db"],
        "Threshold tuning": ["tune_entropy_threshold.py", "patch size distribution"],
        "Interactive FFmpeg": ["dialoguer", "interactive", "user-controlled"]
    }
    
    readme = Path("README.md").read_text()
    api_ref = Path("docs/API_REFERENCE.md").read_text() if Path("docs/API_REFERENCE.md").exists() else ""
    all_content = readme + api_ref
    
    for feature, keywords in features.items():
        found = any(keyword in all_content for keyword in keywords)
        if found:
            print(f"  ✓ {feature} documented")
        else:
            print(f"  ⚠️  {feature} not properly documented")


if __name__ == "__main__":
    print("=== Documentation Consistency Test ===\n")
    
    check_version_consistency()
    check_threshold_values()
    check_removed_claims()
    check_new_features()
    
    print("\n✅ Documentation check complete!")
