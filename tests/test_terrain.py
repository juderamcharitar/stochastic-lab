import unittest,sys,tempfile
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'src'))
from export_terrain import export
class TestTerrain(unittest.TestCase):
    def test_offline_build(self):
        with tempfile.TemporaryDirectory() as d:
            p=export(ROOT/'outputs'/'rolling_surface.csv',Path(d)/'terrain.html')
            s=p.read_text()
            self.assertIn('2026-09-30',s)
            self.assertIn('VISUALLY EXAGGERATED',s)
            self.assertNotIn('__DATA__',s)
            self.assertIn('null',s)
    def test_terrain_records_match(self):
        d=pd.read_csv(ROOT/'outputs'/'rolling_surface.csv')
        self.assertEqual(d.date.nunique(),12)
        self.assertEqual(len(d),6900)
if __name__=='__main__':unittest.main()
