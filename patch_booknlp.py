#!/usr/bin/env python3
"""
Patch BookNLP to fix position_ids compatibility with transformers 4.x+
"""
import re

file_path = "/opt/conda/lib/python3.11/site-packages/booknlp/english/entity_tagger.py"

with open(file_path, 'r') as f:
    content = f.read()

# Find and replace the load_state_dict line
old_code = r'self\.model\.load_state_dict\(torch\.load\(model_file, map_location=device\)\)'
new_code = '''state_dict = torch.load(model_file, map_location=device)
        if "bert.embeddings.position_ids" in state_dict:
            del state_dict["bert.embeddings.position_ids"]
        self.model.load_state_dict(state_dict)'''

content = re.sub(old_code, new_code, content)

with open(file_path, 'w') as f:
    f.write(content)

print("✓ BookNLP patched for transformers 4.x+ compatibility")
