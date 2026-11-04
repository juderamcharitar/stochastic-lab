import unittest,sys,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from export_volatility_terrain import make_frame
class TestVolatilityTerrain(unittest.TestCase):
    def test_real_data_frames_are_finite_and_vary(self):
        d=json.loads((ROOT/'outputs'/'volatility_terrain_data.json').read_text())
        self.assertEqual(len(d['frames']),16)
        arr=np.array([[[np.nan if x is None else x for x in row] for row in f] for f in d['frames']])
        self.assertGreater(float(np.nanstd(arr[0])),.1)
        self.assertGreater(float(np.nanmax(arr)),2)
        self.assertTrue(np.isfinite(np.nanmedian(arr)))
    def test_flat_price_no_fake_mountains(self):
        idx=pd.date_range('2026-10-01',periods=1440,freq='min',tz='UTC')
        s=pd.Series(75000.,index=idx)
        f=make_frame(s)
        self.assertTrue(all(v is None for row in f for v in row))
if __name__=='__main__':unittest.main()
