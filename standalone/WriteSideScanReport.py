"""Atomic strict-JSON diagnostics boundary, including NumPy booleans/arrays."""
from pathlib import Path
import json


def dump(path, value):
    from oceansim.slam import sanitize_for_strict_json
    safe, nonfinite = sanitize_for_strict_json(value)
    if isinstance(safe, dict) and nonfinite:
        safe['JSON_nonfinite_values_replaced_with_null'] = nonfinite
    payload = json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    target = Path(path); temporary = target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(payload)
    temporary.replace(target)
