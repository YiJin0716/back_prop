"""Reuse deterministic per-CT arrays; keep separate, audited fold metadata."""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from .build import HERE_CV, RUN
from ..cohort import HERE, load, sha256
from ..prepare import CACHE, prepare_case

NAMES = ('hu_1mm.npy', 'sybil.npy', 'sybil_attention.npy')


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_arrays(directory, metadata):
    for name in NAMES:
        value = np.load(directory / name, mmap_mode='r')
        assert list(value.shape) == metadata['array_shapes'][name]
        assert value.dtype == {'hu_1mm.npy': np.float32, 'sybil.npy': np.float16,
                               'sybil_attention.npy': np.uint8}[name]


def main():
    plan = json.loads((HERE_CV / 'plan.json').read_text())
    rows = json.loads((RUN / 'case_union.json').read_text())
    digest = sha256(RUN / 'case_union.json')
    shared = RUN / 'shared_cache'; shared.mkdir(exist_ok=True)
    old = load(); old_digest = sha256(HERE / 'cohort.json')
    old_rows = {r['key']: r for split in old['splits'].values() for r in split}
    pending = []
    for row in rows:
        dest = shared / row['key']; dest.mkdir(exist_ok=True)
        if (dest / 'metadata.json').exists():
            meta = json.loads((dest / 'metadata.json').read_text())
            assert meta['cohort_sha256'] == digest
            verify_arrays(dest, meta)
        elif (CACHE / row['key'] / 'metadata.json').exists():
            assert row == old_rows[row['key']], 'Source case changed'
            meta = json.loads((CACHE / row['key'] / 'metadata.json').read_text())
            assert meta['cohort_sha256'] == old_digest
            verify_arrays(CACHE / row['key'], meta)
            for name in NAMES:
                link = dest / name
                if not link.exists():
                    link.symlink_to(CACHE / row['key'] / name)
                assert link.resolve() == (CACHE / row['key'] / name).resolve()
            meta.update(cohort_sha256=digest, reused_from=str(CACHE / row['key']),
                        original_cohort_sha256=old_digest)
            atomic_json(dest / 'metadata.json', meta)
        else:
            pending.append(row)
    print(json.dumps({'reused': len(rows) - len(pending), 'new': len(pending)}), flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        tasks = [pool.submit(prepare_case, row, shared, digest) for row in pending]
        for i, task in enumerate(as_completed(tasks), 1):
            print(json.dumps({'prepared': i, 'total_new': len(tasks), **task.result()}), flush=True)
    summaries = []
    for fold in plan['folds']:
        cohort = load(fold['cohort']); target = Path(fold['directory']) / 'cache'
        target.mkdir(exist_ok=True)
        nodule_count, skipped = 0, []
        # Training consumers cannot see held-out cache entries through this directory.
        for row in cohort['splits']['training']:
            source = shared / row['key']; dest = target / row['key']; dest.mkdir(exist_ok=True)
            meta = json.loads((source / 'metadata.json').read_text())
            assert meta['cohort_sha256'] == digest
            verify_arrays(source, meta)
            for name in NAMES:
                link = dest / name
                if not link.exists():
                    link.symlink_to(source / name)
                assert link.resolve() == (source / name).resolve()
            expected_ids = {n['nodule_id'] for n in row['nodules']}
            actual_ids = {n['nodule_id'] for n in meta['nodules']}
            assert actual_ids <= expected_ids
            skipped_ids = sorted(expected_ids - actual_ids)
            assert sorted(meta.get('skipped_empty_consensus_nodules', skipped_ids)) == skipped_ids
            meta['skipped_empty_consensus_nodules'] = skipped_ids
            meta.update(cohort_sha256=fold['cohort_sha256'], shared_case_union_sha256=digest)
            atomic_json(dest / 'metadata.json', meta)
            nodule_count += sum(not n['ignored'] for n in meta['nodules'])
            if meta['skipped_empty_consensus_nodules']:
                skipped.append({'key': row['key'], 'nodules': meta['skipped_empty_consensus_nodules']})
        assert nodule_count == cohort['policy']['retained_physical_nodules'], 'Missing retained nodule masks'
        assert {p.name for p in target.iterdir()} == {r['key'] for r in cohort['splits']['training']}
        summary = {'fold': fold['fold'], 'cohort_sha256': fold['cohort_sha256'],
                   'scans': len(cohort['splits']['training']), 'retained_nodules': nodule_count,
                   'skipped_empty_consensus': skipped}
        atomic_json(Path(fold['directory']) / 'cache_audit.json', summary)
        summaries.append(summary)
    atomic_json(RUN / 'cache_complete.json', {'folds': summaries, 'union_sha256': digest})
    print(json.dumps({'cache_complete': True, 'folds': summaries}), flush=True)


if __name__ == '__main__':
    main()
