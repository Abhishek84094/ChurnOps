import pathlib, sys

script_path = pathlib.Path("scripts/build_phase9_notebook.py")
script = script_path.read_text(encoding="utf-8")

idx = script.find('"PROJECT_ROOT = Path.cwd()')
if idx == -1:
    print("ERROR: target not found")
    sys.exit(1)

# Find the end of the block (after the sys.path.insert line)
end_marker = r'sys.path.insert(0, str(PROJECT_ROOT))\\n\\n"'
end_idx = script.find(end_marker, idx)
if end_idx == -1:
    print("ERROR: end marker not found")
    print("Dumping around target:")
    print(repr(script[idx:idx+300]))
    sys.exit(1)

end_idx += len(end_marker)

old_block = script[idx:end_idx]
print("OLD BLOCK:")
print(repr(old_block))
print()

new_block = (
    r'"# Find project root robustly (handles notebooks/ kernel cwd)\n"'
    "\n"
    r'    "def _find_root(start):\n"'
    "\n"
    r'    "    for p in [start] + list(start.parents):\n"'
    "\n"
    r'    "        if (p / \"params.yaml\").exists():\n"'
    "\n"
    r'    "            return p\n"'
    "\n"
    r'    "    return start\n\n"'
    "\n"
    r'    "import os; PROJECT_ROOT = _find_root(pathlib.Path.cwd()); os.chdir(PROJECT_ROOT)\n"'
    "\n"
    r'    "if str(PROJECT_ROOT) not in sys.path:\n"'
    "\n"
    r'    "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"'
)

patched = script[:idx] + new_block + script[end_idx:]
# Also ensure pathlib is imported in the notebook cell
patched = patched.replace(
    r'"import sys, json, time, warnings\n"',
    r'"import sys, json, time, warnings, pathlib\n"'
)

script_path.write_text(patched, encoding="utf-8")
print("Patch applied successfully")
print(f"New size: {len(patched)} bytes")
