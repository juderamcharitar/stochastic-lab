"""Rebuild the standalone terrain HTML from any compatible rolling_surface.csv."""
from pathlib import Path
import pandas as pd, json, sys
HERE=Path(__file__).resolve().parent.parent
TEMPLATE=HERE/"src"/"terrain_template.html"

def export(csv_path=None,html_path=None):
    csv_path=Path(csv_path or HERE/"outputs"/"rolling_surface.csv")
    html_path=Path(html_path or HERE/"outputs"/"neon_terrain.html")
    df=pd.read_csv(csv_path)
    req={"date","alpha","beta","loss"}
    if not req.issubset(df.columns):raise ValueError(f"Missing columns: {req-set(df.columns)}")
    if df.duplicated(["date","alpha","beta"]).any():raise ValueError("Duplicate parameter cells")
    dates=sorted(df.date.unique());alpha=sorted(df.alpha.unique());beta=sorted(df.beta.unique())
    frames=[]
    for d in dates:
        a=df[df.date==d].pivot(index="alpha",columns="beta",values="loss").reindex(index=alpha,columns=beta).to_numpy()
        frames.append([[round(float(x),7) if pd.notna(x) else None for x in row] for row in a])
    data={"dates":dates,"alpha":alpha,"beta":beta,"frames":frames,"provenance":"USER-SUPPLIED rolling_surface.csv; provenance must be verified separately"}
    text=TEMPLATE.read_text(encoding="utf-8").replace("__DATA__",json.dumps(data,separators=(",",":")))
    html_path.parent.mkdir(parents=True,exist_ok=True);html_path.write_text(text,encoding="utf-8")
    return html_path

if __name__=="__main__":print(export(sys.argv[1] if len(sys.argv)>1 else None,sys.argv[2] if len(sys.argv)>2 else None))
