#!/usr/bin/env python3
"""
Patch BookNLP to fix position_ids compatibility with transformers 4.x+
"""
import re
import sys

# Find the correct path
try:
    import booknlp
    module_path = booknlp.__path__[0]
    file_path = f"{module_path}/english/entity_tagger.py"
except ImportError:
    print("BookNLP not found, skipping patch")
    sys.exit(0)

print(f"Patching: {file_path}")

with open(file_path, 'r') as f:
    content = f.read()

# Find and replace the load_state_dict line
old_code = r'self\.model\.load_state_dict\(torch\.load\(model_file, map_location=device\)\)'
new_code = '''state_dict = torch.load(model_file, map_location=device)
        if "bert.embeddings.position_ids" in state_dict:
            del state_dict["bert.embeddings.position_ids"]
        self.model.load_state_dict(state_dict)'''

new_content = re.sub(old_code, new_code, content)

if new_content == content:
    print("Warning: Pattern not found, patch may already be applied")
else:
    with open(file_path, 'w') as f:
        f.write(new_content)
    print("✓ BookNLP patched for transformers 4.x+ compatibility")
