import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from back_prop.common import source_compat


class SourceCompatibilityTests(unittest.TestCase):
    def test_relocated_source_without_old_directory_and_later_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / 'common/data.py'
            current.parent.mkdir()
            current.write_text('relocated implementation')
            before = hashlib.sha256(b'original implementation').hexdigest()
            after = source_compat._sha256(current)
            registry = root / 'sources.json'
            registry.write_text(json.dumps({'old/data.py': {
                'path': 'common/data.py', 'before': before, 'after': after}}))
            with patch.object(source_compat, 'ROOT', root), \
                 patch.object(source_compat, 'REGISTRY', registry):
                self.assertFalse((root / 'old').exists())
                self.assertTrue(source_compat.source_matches('old/data.py', before))
                self.assertTrue(source_compat.source_matches(current, before))
                self.assertTrue(source_compat.source_matches(current, after))
                self.assertFalse(source_compat.source_matches('old/data.py', 'unknown'))
                self.assertFalse(source_compat.source_matches('unknown/data.py', before))
                self.assertEqual(source_compat.cache_source_identity(current),
                                 (str(root / 'old/data.py'), before))
                current.write_text('changed implementation')
                self.assertFalse(source_compat.source_matches('old/data.py', before))
                self.assertFalse(source_compat.source_matches(current, after))
                self.assertEqual(source_compat.cache_source_identity(current),
                                 (str(current), source_compat._sha256(current)))

    def test_retired_entries_do_not_match_external_or_unknown_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / 'sources.json'
            registry.write_text(json.dumps({'old/__init__.py': {
                'path': None, 'before': 'original', 'after': None}}))
            with patch.object(source_compat, 'ROOT', root), \
                 patch.object(source_compat, 'REGISTRY', registry):
                self.assertTrue(source_compat.source_matches('old/__init__.py', 'original'))
                self.assertFalse(source_compat.source_matches('old/__init__.py', 'changed'))
                self.assertFalse(source_compat.source_matches(root.parent / 'missing', 'original'))
