from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web import db


ROOT_DIR = Path(__file__).resolve().parents[1]
FIGURE_EXTENSIONS = {".svg", ".png", ".jpg", ".jpeg", ".webp", ".gif"}
FIGURE_PRIORITY = [".svg", ".png", ".jpg", ".jpeg", ".webp", ".gif"]

app = FastAPI(title="IDR-LLPS Database")

static_dir = ROOT_DIR / "web" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

reports_dir = ROOT_DIR / "reports"
if reports_dir.exists():
    app.mount("/reports", StaticFiles(directory=reports_dir), name="reports")

ml_results_dir = ROOT_DIR / "ml_dl" / "results"
if ml_results_dir.exists():
    app.mount("/ml_results", StaticFiles(directory=ml_results_dir), name="ml_results")

data_dir = ROOT_DIR / "data"
if data_dir.exists():
    app.mount("/data", StaticFiles(directory=data_dir), name="data")

templates = Jinja2Templates(directory=str(ROOT_DIR / "web" / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
        },
    )


@app.get("/samples", response_class=HTMLResponse)
async def samples_page(
    request: Request,
    label: str | None = Query(default=None),
    label_source: str | None = Query(default=None),
    taxon_group: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
) -> HTMLResponse:
    page_size = 50
    offset = (page - 1) * page_size
    rows, total = db.list_samples(
        label=label,
        label_source=label_source,
        taxon_group=taxon_group,
        limit=page_size,
        offset=offset,
    )
    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    return templates.TemplateResponse(
        "samples.html",
        {
            "request": request,
            "rows": rows,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "label": label or "",
            "label_source": label_source or "",
            "taxon_group": taxon_group or "",
        },
    )


@app.get("/samples/{sample_id}", response_class=HTMLResponse)
async def sample_detail(request: Request, sample_id: str) -> HTMLResponse:
    sample = db.get_sample(sample_id)
    if not sample:
        return templates.TemplateResponse(
            "sample_detail.html",
            {"request": request, "sample": None, "features": None},
            status_code=404,
        )
    features = db.get_sample_features(sample_id)
    return templates.TemplateResponse(
        "sample_detail.html",
        {
            "request": request,
            "sample": sample,
            "features": features,
        },
    )


@app.get("/plots", response_class=HTMLResponse)
async def plots_page(request: Request) -> HTMLResponse:
    figures_dir = ROOT_DIR / "reports" / "figures"
    # dedupe by stem, prefer vector formats (svg) over png/jpg, etc.
    by_stem: dict[str, Path] = {}
    if figures_dir.exists():
        for path in figures_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in FIGURE_EXTENSIONS:
                continue
            # normalize key to avoid duplicates caused by case/spacing variants.
            stem = path.stem
            key = stem.lower().replace(" ", "_").replace("-", "_")
            current = by_stem.get(key)
            if current is None:
                by_stem[key] = path
                continue
            # choose better extension according to FIGURE_PRIORITY
            def _score(p: Path) -> int:
                suffix = p.suffix.lower()
                return FIGURE_PRIORITY.index(suffix) if suffix in FIGURE_PRIORITY else len(FIGURE_PRIORITY)

            if _score(path) < _score(current):
                by_stem[key] = path

    report_figures: list[dict[str, str]] = []
    for _, path in sorted(by_stem.items(), key=lambda kv: kv[0]):
        report_figures.append(
            {
                "src": f"/reports/figures/{path.name}",
                "name": path.name,
                "title": path.stem.replace("_", " "),
            }
        )

    return templates.TemplateResponse(
        "plots.html",
        {
            "request": request,
            "report_figures": report_figures,
        },
    )


@app.get("/downloads", response_class=HTMLResponse)
async def downloads_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "downloads.html",
        {
            "request": request,
        },
    )


@app.get("/api/samples")
async def api_samples(
    label: str | None = Query(default=None),
    label_source: str | None = Query(default=None),
    taxon_group: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> JSONResponse:
    rows, total = db.list_samples(
        label=label,
        label_source=label_source,
        taxon_group=taxon_group,
        limit=limit,
        offset=offset,
    )
    return JSONResponse(
        {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": rows,
        }
    )


@app.get("/api/samples/{sample_id}")
async def api_sample_detail(sample_id: str) -> JSONResponse:
    sample = db.get_sample(sample_id)
    if not sample:
        return JSONResponse({"detail": "sample not found"}, status_code=404)
    features = db.get_sample_features(sample_id)
    return JSONResponse({"sample": sample, "features": features})


@app.get("/api/stats")
async def api_stats() -> JSONResponse:
    stats = db.get_stats()
    return JSONResponse(stats)


@app.get("/health", response_class=HTMLResponse)
async def health() -> dict[str, str]:
    return {"status": "ok"}


