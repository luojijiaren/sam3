#!/usr/bin/env python3
"""
Script to update all .cuda() calls to device-agnostic .to(device) calls in SAM3 codebase.
"""

import re
import os
from pathlib import Path

# Files to update
FILES_TO_UPDATE = [
    "sam3/model/io_utils.py",
    "sam3/model/sam3_tracker_base.py",
    "sam3/model/sam3_tracking_predictor.py",
    "sam3/agent/helpers/memory.py",
    "sam3/train/trainer.py",
    "sam3/train/utils/train_utils.py",
    "sam3/train/utils/distributed.py",
    "sam3/sam/transformer.py",
]

def add_import_if_needed(content: str) -> str:
    """Add device_utils import if not already present."""
    if "from sam3.utils.device_utils import" in content:
        return content
    
    # Find where to insert the import
    import_lines = []
    other_lines = []
    in_imports = True
    
    for line in content.split('\n'):
        if in_imports and (line.startswith('import ') or line.startswith('from ')):
            import_lines.append(line)
        elif in_imports and line.strip() and not line.startswith('#'):
            in_imports = False
            other_lines.append(line)
        else:
            other_lines.append(line)
    
    # Add our import after torch imports
    torch_import_idx = -1
    for i, line in enumerate(import_lines):
        if 'import torch' in line:
            torch_import_idx = i
    
    if torch_import_idx >= 0:
        import_lines.insert(torch_import_idx + 1, "from sam3.utils.device_utils import get_device")
    else:
        import_lines.append("from sam3.utils.device_utils import get_device")
    
    return '\n'.join(import_lines) + '\n' + '\n'.join(other_lines)

def replace_cuda_calls(content: str) -> str:
    """Replace .cuda() calls with .to(get_device()) or device-aware alternatives."""
    
    # Pattern 1: .cuda() -> .to(get_device())
    content = re.sub(r'\.cuda\(\)', '.to(get_device())', content)
    
    # Pattern 2: .cuda(non_blocking=True) -> .to(get_device(), non_blocking=True)
    content = re.sub(r'\.cuda\(non_blocking=True\)', '.to(get_device(), non_blocking=True)', content)
    
    # Pattern 3: torch.cuda.empty_cache() -> from utils
    content = re.sub(
        r'torch\.cuda\.empty_cache\(\)',
        'get_device(); empty_cache()',  # Need to import empty_cache too
        content
    )
    
    return content

def update_file(filepath: str):
    """Update a single file."""
    print(f"Updating {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"  Skipping (not found): {filepath}")
        return
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Add import
    content = add_import_if_needed(content)
    
    # Replace CUDA calls
    content = replace_cuda_calls(content)
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✓ Updated")
    else:
        print(f"  - No changes needed")

def main():
    os.chdir()
    
    for filepath in FILES_TO_UPDATE:
        update_file(filepath)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
