from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.data.multicorpus import (
    CorpusSource,
    build_multicorpus_catalog,
    build_pdmx_catalog,
    parse_symbtr_filename,
    prepare_jazzmus_musicxml,
)


SIMPLE_MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1"><part-name>Music</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>1</duration><type>quarter</type>
      </note>
    </measure>
  </part>
</score-partwise>
"""


class MultiCorpusTests(unittest.TestCase):
    def test_parse_symbtr_filename(self) -> None:
        parsed = parse_symbtr_filename("hicaz--sarki--duyek--bahar_olsa--fahri_kopuz.xml")
        self.assertEqual(parsed["makam"], "hicaz")
        self.assertEqual(parsed["form"], "sarki")
        self.assertEqual(parsed["usul"], "duyek")
        self.assertEqual(parsed["composer_hint"], "Fahri Kopuz")

    def test_prepare_jazzmus_musicxml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            json_path = root / "blue-monk.json"
            json_path.write_text(
                json.dumps({"encodings": {"musicxml": SIMPLE_MUSICXML}}, ensure_ascii=False),
                encoding="utf-8",
            )
            exported = prepare_jazzmus_musicxml(root, root / "musicxml")
            self.assertEqual(exported.iloc[0]["status"], "written")
            self.assertTrue((root / "musicxml" / "blue-monk.musicxml").exists())

    def test_build_pdmx_catalog_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            score_dir = root / "scores"
            score_dir.mkdir()
            score_path = score_dir / "sample.musicxml"
            score_path.write_text(SIMPLE_MUSICXML, encoding="utf-8")
            manifest = pd.DataFrame(
                [
                    {
                        "path": "scores/sample.musicxml",
                        "title": "Sample Piece",
                        "composer": "Anon",
                        "license_conflict": False,
                        "subset:no_license_conflict": True,
                        "genre": "mixed",
                    }
                ]
            )
            manifest.to_csv(root / "PDMX.csv", index=False)
            catalog = build_pdmx_catalog(root)
            self.assertEqual(len(catalog), 1)
            self.assertEqual(catalog.iloc[0]["source_type"], "pdmx")
            self.assertEqual(catalog.iloc[0]["title"], "Sample Piece")

    def test_build_multicorpus_catalog_mixes_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            generic = root / "generic"
            symbtr = root / "symbtr" / "MusicXML"
            generic.mkdir(parents=True)
            symbtr.mkdir(parents=True)
            (generic / "simple.musicxml").write_text(SIMPLE_MUSICXML, encoding="utf-8")
            (symbtr / "rast--sarki--duyek--ornek--dede_efendi.xml").write_text(SIMPLE_MUSICXML, encoding="utf-8")

            catalog = build_multicorpus_catalog(
                [
                    CorpusSource(name="Generic", source_type="generic", root_dir=generic),
                    CorpusSource(name="SymbTr", source_type="symbtr", root_dir=symbtr.parent),
                ]
            )
            self.assertEqual(set(catalog["source_type"]), {"generic", "symbtr"})


if __name__ == "__main__":
    unittest.main()
