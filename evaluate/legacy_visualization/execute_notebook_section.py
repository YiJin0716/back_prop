"""Execute only the appended V3 cells and embed their outputs in visual.ipynb.

Run with the existing visual Python after GPU inference has produced its cache.
Earlier cells and their outputs are preserved. No other notebook section runs.
"""
import json
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    path = Path(__file__).resolve().parents[3] / 'visual.ipynb'
    original = path.read_text()
    notebook = nbformat.reads(original, as_version=4)
    shell = InteractiveShell.instance()
    for cell in notebook.cells:
        if not cell.get('id', '').startswith('v3-seg-') or cell.cell_type != 'code':
            continue
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        if result.error_before_exec or result.error_in_exec:
            print(captured.stdout)
            print(captured.stderr)
            raise RuntimeError(f"Cell {cell.id} failed") from (result.error_before_exec or result.error_in_exec)
        outputs = []
        for name, text in [('stdout', captured.stdout), ('stderr', captured.stderr)]:
            if text:
                outputs.append(nbformat.v4.new_output('stream', name=name, text=text))
        for rich in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data', data=rich.data, metadata=rich.metadata))
        cell.outputs = outputs
        cell.execution_count = shell.execution_count - 1
        print(f'Executed {cell.id}: {len(outputs)} outputs')
    nbformat.validate(notebook)
    if path.read_text() != original:
        raise RuntimeError('Notebook changed during execution; refusing to overwrite concurrent edits')
    # Keep existing cell JSON exactly as supplied, including source list formatting.
    updated = json.loads(original)
    completed = {c.id: c for c in notebook.cells if c.get('id', '').startswith('v3-seg-')}
    for cell in updated['cells']:
        if cell.get('id') in completed and cell['cell_type'] == 'code':
            cell['outputs'] = completed[cell['id']].outputs
            cell['execution_count'] = completed[cell['id']].execution_count
    temporary = path.with_suffix('.executing.tmp')
    temporary.write_text(json.dumps(updated, ensure_ascii=False, indent=1)+'\n')
    temporary.replace(path)
    print('Saved executed V3 cells:', path)


if __name__ == '__main__':
    main()
