"""Bundle data and Plotly into a self-contained offline dashboard."""
import json
from pathlib import Path
from plotly.offline import get_plotlyjs
from .exploration import CHARTS, plotly_figures


def records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def build_dashboard(root, clean, audit, manifest, tables, prepared=None):
    payload = {"audit": audit, "manifest": manifest, "observations": records(clean),
               "charts": {key: {"title": title, "note": note} for key,(title,note) in CHARTS.items()},
               "exploration": plotly_figures(clean, prepared),
               "predictions": records(tables["predictions"]), "forecast": records(tables["forecast"]),
               "candidate_forecasts": records(tables["candidate_forecasts"])}
    template = Path(__file__).with_name("dashboard_template.html").read_text(encoding="utf-8")
    data = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    page = template.replace("__PAYLOAD__", data).replace("__PLOTLY__", get_plotlyjs())
    (Path(root)/"reports/dashboard.html").write_text(page, encoding="utf-8")
