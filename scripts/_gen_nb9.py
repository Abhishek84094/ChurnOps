import json, pathlib

def md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src, 'id': 'md' + str(abs(hash(src)))[:6]}

def code(src):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': src, 'id': 'co' + str(abs(hash(src)))[:6]}

# Read cell sources from a data file
data = json.loads(pathlib.Path('scripts/_nb9_cells.json').read_text(encoding='utf-8'))

cells = []
for item in data:
    if item['type'] == 'md':
        cells.append(md(item['source']))
    else:
        cells.append(code(item['source']))

nb = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.8'}
    },
    'cells': cells
}

out = pathlib.Path('notebooks/08_baseline_models.ipynb')
out.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print(f'Written: {out}  ({out.stat().st_size:,} bytes)')
