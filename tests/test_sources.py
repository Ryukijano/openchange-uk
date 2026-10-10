import tempfile
import unittest
from datetime import date
from pathlib import Path

from openchange.sources import (
    DEFAULT_PATH,
    REQUIRED,
    SourceNotApproved,
    load_document,
    load_sources,
    require_approved,
    validate_document,
    validate_entry,
)

TODAY = date(2026, 10, 6)


def entry(**overrides):
    base = {
        "id": "example-imagery",
        "url": "https://example.org/collection",
        "license": "Example Open Licence 1.0",
        "license_url": "https://example.org/licence",
        "attribution": "Contains example data",
        "expected_size_bytes": 1_000_000,
        "use": "imagery",
        "splits": ["pilot"],
        "redistribution": "manifest-only",
        "approved_by": "reviewer",
        "approved_on": "2026-10-01",
    }
    base.update(overrides)
    return base


class EntryTest(unittest.TestCase):
    def test_complete_entry_is_valid(self) -> None:
        self.assertEqual(validate_entry(entry(), TODAY), [])

    def test_every_required_field_is_enforced(self) -> None:
        for key in REQUIRED:
            bad = entry()
            del bad[key]
            self.assertIn(f"missing {key}", validate_entry(bad, TODAY), key)

    def test_rejections(self) -> None:
        cases = {
            "url": entry(url="http://example.org/x"),
            "license": entry(license="TBD"),
            "attribution": entry(attribution=""),
            "expected_size_bytes": entry(expected_size_bytes="12 GB"),
            "zero size": entry(expected_size_bytes=0),
            "use": entry(use="everything"),
            "splits": entry(splits=["random"]),
            "empty splits": entry(splits=[]),
            "redistribution": entry(redistribution="maybe"),
            "future approval": entry(approved_on="2027-01-01"),
            "bad date": entry(approved_on="yesterday"),
            "unknown field": entry(licence="typo"),
            "bad id": entry(id="Has Spaces"),
            "checksum": entry(checksum_sha256="abc"),
        }
        for name, bad in cases.items():
            self.assertTrue(validate_entry(bad, TODAY), name)


class DocumentTest(unittest.TestCase):
    def test_repo_sources_file_is_valid_and_empty(self) -> None:
        self.assertEqual(validate_document(load_document(DEFAULT_PATH)), [])
        self.assertEqual(load_sources(DEFAULT_PATH), [])

    def test_duplicate_ids_and_unknown_keys(self) -> None:
        doc = {"approved_public_sources": [entry(), entry()], "extra": 1}
        errors = validate_document(doc, TODAY)
        self.assertTrue(any("duplicate id" in e for e in errors))
        self.assertTrue(any("unknown top-level key" in e for e in errors))

    def test_require_approved(self) -> None:
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sources.yaml"
            path.write_text(yaml.safe_dump({"approved_public_sources": [entry()]}))
            source = require_approved("example-imagery", "pilot", path, TODAY)
            self.assertEqual(source.splits, ("pilot",))
            with self.assertRaises(SourceNotApproved):
                require_approved("example-imagery", "test", path, TODAY)
            with self.assertRaises(SourceNotApproved):
                require_approved("not-listed", None, path, TODAY)

    def test_invalid_file_blocks_every_source(self) -> None:
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sources.yaml"
            path.write_text(yaml.safe_dump({"approved_public_sources": [entry(license="")]}))
            with self.assertRaises(SourceNotApproved):
                require_approved("example-imagery", None, path, TODAY)

    def test_empty_repo_config_blocks_fetch(self) -> None:
        with self.assertRaises(SourceNotApproved):
            require_approved("anything")


if __name__ == "__main__":
    unittest.main()
