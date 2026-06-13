from __future__ import annotations

import cgi
import json
import logging
import mimetypes
import re
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.pipeline.context import PipelineContext
from src.pipeline.pipeline import CADQuantityPipeline
from src.utils.config import load_pipeline_config
from src.utils.logger import configure_logger
from src.utils.runtime_paths import configure_workspace_runtime_dirs


ROOT = Path(__file__).resolve().parent
UI_RUNS_DIR = ROOT / "outputs" / "ui_runs"
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf", ".dxf"}
DEFAULT_DETECTORS = "yolo,rtdetr"
DEFAULT_SEGMENTER = "segformer"
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def main() -> int:
    configure_workspace_runtime_dirs()
    configure_logger(debug=False)
    UI_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    load_persisted_jobs()

    host = "127.0.0.1"
    port = _find_port(8765)
    server = ThreadingHTTPServer((host, port), UIRequestHandler)
    print(f"Basic CAD pipeline UI running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping UI server.")
    finally:
        server.server_close()
    return 0


class UIRequestHandler(BaseHTTPRequestHandler):
    server_version = "CADPipelineUI/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/api/jobs":
            self._send_json({"jobs": list_jobs()})
            return
        match = re.fullmatch(r"/api/jobs/([^/]+)", parsed.path)
        if match:
            job = get_job(match.group(1))
            if not job:
                self._send_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(expand_job(job))
            return
        match = re.fullmatch(r"/artifacts/([^/]+)/(.+)", parsed.path)
        if match:
            self._send_artifact(match.group(1), match.group(2))
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/jobs":
            self._handle_create_job()
            return
        match = re.fullmatch(r"/api/jobs/([^/]+)/validation", parsed.path)
        if match:
            self._handle_validation(match.group(1))
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args) -> None:
        logging.getLogger("cad_quantity_pipeline.ui.http").debug(format, *args)

    def _handle_create_job(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send_json({"error": "expected multipart/form-data"}, HTTPStatus.BAD_REQUEST)
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )
        if "file" not in form:
            self._send_json({"error": "missing file"}, HTTPStatus.BAD_REQUEST)
            return
        file_item = form["file"]
        if not getattr(file_item, "filename", None):
            self._send_json({"error": "empty file upload"}, HTTPStatus.BAD_REQUEST)
            return

        original_name = Path(file_item.filename).name
        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            self._send_json({"error": f"unsupported file type: {suffix}"}, HTTPStatus.BAD_REQUEST)
            return

        job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run_dir = UI_RUNS_DIR / job_id
        input_dir = run_dir / "input"
        output_dir = run_dir / "pipeline"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / safe_filename(original_name)
        with input_path.open("wb") as handle:
            handle.write(file_item.file.read())

        detectors = form_value(form, "detectors", DEFAULT_DETECTORS)
        segmenter = form_value(form, "segmenter", DEFAULT_SEGMENTER)
        scale = parse_float(form_value(form, "scale", "0.01"))
        debug = form_value(form, "debug", "false").lower() in {"1", "true", "yes", "on"}

        job = {
            "id": job_id,
            "status": "queued",
            "created_at": timestamp(),
            "started_at": None,
            "finished_at": None,
            "input_name": original_name,
            "input_path": str(input_path.relative_to(ROOT)),
            "output_dir": str(output_dir.relative_to(ROOT)),
            "detectors": detectors,
            "segmenter": segmenter,
            "scale": scale,
            "debug": debug,
            "errors": [],
            "warnings": [],
        }
        save_job(job)
        thread = threading.Thread(target=run_job, args=(job_id,), daemon=True)
        thread.start()
        self._send_json(expand_job(job), HTTPStatus.CREATED)

    def _handle_validation(self, job_id: str) -> None:
        job = get_job(job_id)
        if not job:
            self._send_json({"error": "job not found"}, HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        decision = str(payload.get("decision") or "").strip().lower()
        if decision not in {"approved", "rejected", "needs_review"}:
            self._send_json({"error": "decision must be approved, rejected, or needs_review"}, HTTPStatus.BAD_REQUEST)
            return
        validation = {
            "job_id": job_id,
            "decision": decision,
            "notes": str(payload.get("notes") or "").strip(),
            "saved_at": timestamp(),
        }
        run_dir = UI_RUNS_DIR / job_id
        (run_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
        job["validation"] = validation
        save_job(job)
        self._send_json(expand_job(job))

    def _send_artifact(self, job_id: str, relative_name: str) -> None:
        job = get_job(job_id)
        if not job:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        output_dir = (ROOT / job["output_dir"]).resolve()
        candidate = (output_dir / relative_name).resolve()
        if output_dir not in candidate.parents and candidate != output_dir:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not candidate.exists() or not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        data = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, value: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(value, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run_job(job_id: str) -> None:
    job = get_job(job_id)
    if not job:
        return
    update_job(job_id, status="running", started_at=timestamp())
    log_path = UI_RUNS_DIR / job_id / "pipeline.log"
    logger = logging.getLogger(f"cad_quantity_pipeline.ui.{job_id}")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if job.get("debug") else logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)

    try:
        input_path = ROOT / job["input_path"]
        output_dir = ROOT / job["output_dir"]
        context = PipelineContext(
            input_path=input_path,
            output_dir=output_dir,
            scale_ratio=job.get("scale"),
            config=load_pipeline_config(),
            detector_names=[item.strip() for item in job["detectors"].split(",") if item.strip()],
            segmenter_name=job["segmenter"],
            debug=bool(job.get("debug")),
        )
        result = CADQuantityPipeline(logger).run(context)
        status = "failed" if result.errors else "completed"
        update_job(
            job_id,
            status=status,
            finished_at=timestamp(),
            errors=result.errors,
            warnings=result.warnings,
        )
    except Exception as exc:
        logger.error("UI job failed: %s", exc)
        logger.debug(traceback.format_exc())
        update_job(job_id, status="failed", finished_at=timestamp(), errors=[str(exc)])
    finally:
        logger.removeHandler(handler)
        handler.close()


def expand_job(job: dict) -> dict:
    expanded = dict(job)
    output_dir = ROOT / job["output_dir"]
    report_path = output_dir / "processing_report.md"
    quantities_path = output_dir / "quantities.json"
    merged_path = output_dir / "merged_objects.json"
    graph_path = output_dir / "drawing_graph.json"
    validation_path = UI_RUNS_DIR / job["id"] / "validation.json"
    expanded["artifacts"] = {
        "rendered": artifact_url(job, "rendered.png"),
        "preprocessed": artifact_url(job, "preprocessed.png"),
        "overlay": artifact_url(job, "debug_overlay.png"),
        "tiles_overlay": artifact_url(job, "debug_overlay_tiles.png"),
        "excel": artifact_url(job, "quantities.xlsx"),
        "report": artifact_url(job, "processing_report.md"),
    }
    expanded["report_text"] = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    expanded["quantities"] = read_json(quantities_path, [])
    expanded["object_counts"] = object_counts(read_json(merged_path, []))
    graph = read_json(graph_path, {})
    expanded["graph_edges"] = len(graph.get("edges", [])) if isinstance(graph, dict) else 0
    if validation_path.exists():
        expanded["validation"] = read_json(validation_path, None)
    return expanded


def artifact_url(job: dict, filename: str) -> str | None:
    output_dir = ROOT / job["output_dir"]
    return f"/artifacts/{job['id']}/{filename}" if (output_dir / filename).exists() else None


def object_counts(objects: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in objects:
        object_type = str(item.get("type") or "unknown")
        counts[object_type] = counts.get(object_type, 0) + 1
    return counts


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def list_jobs() -> list[dict]:
    load_persisted_jobs()
    with JOBS_LOCK:
        return [expand_job(job) for job in sorted(JOBS.values(), key=lambda item: item["created_at"], reverse=True)]


def get_job(job_id: str) -> dict | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            return dict(job)
    path = UI_RUNS_DIR / job_id / "job.json"
    if path.exists():
        job = read_json(path, None)
        if job:
            with JOBS_LOCK:
                JOBS[job_id] = job
            return dict(job)
    return None


def update_job(job_id: str, **updates) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job.update(updates)
        current = dict(job)
    save_job(current)


def save_job(job: dict) -> None:
    with JOBS_LOCK:
        JOBS[job["id"]] = dict(job)
    run_dir = UI_RUNS_DIR / job["id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "job.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")


def load_persisted_jobs() -> None:
    if not UI_RUNS_DIR.exists():
        return
    with JOBS_LOCK:
        for path in UI_RUNS_DIR.glob("*/job.json"):
            job = read_json(path, None)
            if job:
                JOBS[job["id"]] = job


def form_value(form: cgi.FieldStorage, name: str, default: str) -> str:
    if name not in form:
        return default
    value = form[name].value
    return str(value).strip() or default


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._")
    return cleaned or "upload"


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _find_port(start: int) -> int:
    import socket

    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No available UI port found.")


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CAD Pipeline Validation</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --border: #d8dde5;
      --text: #151922;
      --muted: #667085;
      --accent: #0f766e;
      --danger: #b42318;
      --warn: #b54708;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--text);
      font-size: 14px;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      border-bottom: 1px solid var(--border);
      background: var(--panel);
    }
    h1 { margin: 0; font-size: 18px; font-weight: 700; }
    main {
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
      gap: 14px;
      padding: 14px;
      min-height: calc(100vh - 56px);
    }
    section, aside {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
    }
    aside { padding: 14px; overflow: auto; }
    .workspace { display: grid; grid-template-rows: auto minmax(0, 1fr); overflow: hidden; }
    .toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
    }
    h2 { margin: 0 0 10px; font-size: 15px; }
    label { display: block; margin: 10px 0 5px; color: var(--muted); font-size: 12px; font-weight: 700; }
    input, select, textarea, button {
      width: 100%;
      font: inherit;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 9px 10px;
      background: white;
    }
    textarea { min-height: 82px; resize: vertical; }
    button {
      cursor: pointer;
      background: var(--accent);
      border-color: var(--accent);
      color: white;
      font-weight: 700;
    }
    button.secondary { background: white; color: var(--text); border-color: var(--border); }
    button.danger { background: var(--danger); border-color: var(--danger); }
    button.warn { background: var(--warn); border-color: var(--warn); }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .hint, .meta { color: var(--muted); font-size: 12px; line-height: 1.4; }
    .job-list { display: grid; gap: 8px; margin-top: 14px; }
    .job {
      display: grid;
      gap: 4px;
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: white;
      cursor: pointer;
    }
    .job.active { border-color: var(--accent); box-shadow: inset 3px 0 0 var(--accent); }
    .job-title { font-weight: 700; overflow-wrap: anywhere; }
    .status {
      display: inline-flex;
      align-items: center;
      width: fit-content;
      border-radius: 999px;
      padding: 3px 8px;
      background: #e7f7f4;
      color: #096055;
      font-size: 12px;
      font-weight: 700;
    }
    .status.failed { background: #fee4e2; color: #912018; }
    .status.running, .status.queued { background: #fff4e5; color: #93370d; }
    .content {
      min-height: 0;
      overflow: auto;
      padding: 14px;
      display: grid;
      gap: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      background: #fbfcfd;
    }
    .metric strong { display: block; font-size: 18px; margin-top: 3px; }
    .viewer {
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(320px, .9fr);
      gap: 14px;
      align-items: start;
    }
    .image-wrap {
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: auto;
      background: #eef1f5;
      min-height: 360px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .image-wrap img { max-width: 100%; height: auto; display: block; }
    pre {
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfd;
      max-height: 520px;
      overflow: auto;
    }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid var(--border); padding: 7px 6px; text-align: left; }
    th { color: var(--muted); font-size: 12px; }
    .actions { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
    @media (max-width: 980px) {
      main, .viewer { grid-template-columns: 1fr; }
      .grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <h1>CAD Pipeline Validation</h1>
    <button class="secondary" id="refreshBtn" style="width:auto">Refresh</button>
  </header>
  <main>
    <aside>
      <h2>Run Pipeline</h2>
      <form id="uploadForm">
        <label for="file">CAD file</label>
        <input id="file" name="file" type="file" accept=".png,.jpg,.jpeg,.pdf,.dxf" required>
        <label for="detectors">Detectors</label>
        <select id="detectors" name="detectors">
          <option value="yolo,rtdetr">YOLO + RT-DETR</option>
          <option value="yolo">YOLO</option>
          <option value="rtdetr">RT-DETR</option>
          <option value="mock">Mock</option>
        </select>
        <label for="segmenter">Segmenter</label>
        <select id="segmenter" name="segmenter">
          <option value="segformer">SegFormer</option>
          <option value="mock">Mock</option>
        </select>
        <label for="scale">Scale ratio</label>
        <input id="scale" name="scale" value="0.01">
        <label><input id="debug" name="debug" type="checkbox" style="width:auto"> Debug overlays</label>
        <button id="runBtn" type="submit">Run</button>
      </form>
      <p class="hint">This is a temporary validation UI. It writes runs to <code>outputs/ui_runs</code>.</p>
      <h2 style="margin-top:18px">Runs</h2>
      <div id="jobList" class="job-list"></div>
    </aside>
    <section class="workspace">
      <div class="toolbar">
        <div>
          <div id="selectedTitle" style="font-weight:700">No run selected</div>
          <div id="selectedMeta" class="meta"></div>
        </div>
        <span id="selectedStatus" class="status">idle</span>
      </div>
      <div id="content" class="content">
        <p class="hint">Upload a drawing or select a previous run.</p>
      </div>
    </section>
  </main>
  <script>
    let jobs = [];
    let selectedId = null;
    let poll = null;

    const uploadForm = document.getElementById('uploadForm');
    const jobList = document.getElementById('jobList');
    const content = document.getElementById('content');
    const selectedTitle = document.getElementById('selectedTitle');
    const selectedMeta = document.getElementById('selectedMeta');
    const selectedStatus = document.getElementById('selectedStatus');

    uploadForm.addEventListener('submit', async (event) => {
      event.preventDefault();
      const button = document.getElementById('runBtn');
      button.disabled = true;
      button.textContent = 'Running...';
      try {
        const response = await fetch('/api/jobs', { method: 'POST', body: new FormData(uploadForm) });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Upload failed');
        selectedId = payload.id;
        await loadJobs();
        await loadJob(selectedId);
      } catch (error) {
        alert(error.message);
      } finally {
        button.disabled = false;
        button.textContent = 'Run';
      }
    });
    document.getElementById('refreshBtn').addEventListener('click', async () => {
      await loadJobs();
      if (selectedId) await loadJob(selectedId);
    });

    async function loadJobs() {
      const response = await fetch('/api/jobs');
      const payload = await response.json();
      jobs = payload.jobs || [];
      renderJobList();
      if (!selectedId && jobs[0]) {
        selectedId = jobs[0].id;
        await loadJob(selectedId);
      }
    }

    async function loadJob(id) {
      const response = await fetch('/api/jobs/' + encodeURIComponent(id));
      const job = await response.json();
      selectedId = job.id;
      renderJobList();
      renderJob(job);
      if (['queued', 'running'].includes(job.status)) startPolling();
      else stopPolling();
    }

    function renderJobList() {
      jobList.innerHTML = '';
      for (const job of jobs) {
        const node = document.createElement('div');
        node.className = 'job' + (job.id === selectedId ? ' active' : '');
        node.innerHTML = `<div class="job-title">${escapeHtml(job.input_name)}</div>
          <span class="status ${job.status}">${escapeHtml(job.status)}</span>
          <div class="meta">${escapeHtml(job.created_at)} · ${escapeHtml(job.detectors)} · ${escapeHtml(job.segmenter)}</div>`;
        node.onclick = () => loadJob(job.id);
        jobList.appendChild(node);
      }
    }

    function renderJob(job) {
      selectedTitle.textContent = job.input_name || job.id;
      selectedMeta.textContent = `${job.detectors} · ${job.segmenter} · scale ${job.scale ?? 'none'}`;
      selectedStatus.className = 'status ' + job.status;
      selectedStatus.textContent = job.status;
      const counts = job.object_counts || {};
      const quantities = job.quantities || [];
      const overlay = job.artifacts && job.artifacts.overlay;
      const report = job.report_text || '';
      content.innerHTML = `
        <div class="grid">
          ${metric('Doors', counts.door || 0)}
          ${metric('Windows', counts.window || 0)}
          ${metric('Rooms', counts.room || 0)}
          ${metric('Walls', counts.wall || 0)}
          ${metric('Graph Edges', job.graph_edges || 0)}
          ${metric('Warnings', (job.warnings || []).length)}
          ${metric('Errors', (job.errors || []).length)}
          ${metric('Validation', job.validation ? job.validation.decision : 'open')}
        </div>
        <div class="viewer">
          <div class="image-wrap">${overlay ? `<img src="${overlay}" alt="Debug overlay">` : `<span class="hint">Overlay will appear after export.</span>`}</div>
          <div>
            <h2>Validation</h2>
            ${validationForm(job)}
            <h2 style="margin-top:14px">Quantities</h2>
            ${quantityTable(quantities)}
          </div>
        </div>
        <h2>Processing Report</h2>
        <pre>${escapeHtml(report || 'Report will appear after the run completes.')}</pre>
      `;
      const form = document.getElementById('validationForm');
      if (form) {
        form.addEventListener('submit', async (event) => {
          event.preventDefault();
          const clicked = event.submitter;
          await saveValidation(job.id, clicked.value, document.getElementById('notes').value);
        });
      }
    }

    function validationForm(job) {
      const notes = job.validation ? job.validation.notes || '' : '';
      return `<form id="validationForm">
        <textarea id="notes" placeholder="Validation notes">${escapeHtml(notes)}</textarea>
        <div class="actions" style="margin-top:8px">
          <button name="decision" value="approved">Approve</button>
          <button class="warn" name="decision" value="needs_review">Needs Review</button>
          <button class="danger" name="decision" value="rejected">Reject</button>
        </div>
      </form>`;
    }

    async function saveValidation(id, decision, notes) {
      const response = await fetch(`/api/jobs/${encodeURIComponent(id)}/validation`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision, notes })
      });
      const payload = await response.json();
      if (!response.ok) alert(payload.error || 'Validation save failed');
      else renderJob(payload);
    }

    function quantityTable(items) {
      if (!items.length) return '<p class="hint">No quantities yet.</p>';
      return `<table><thead><tr><th>Name</th><th>Quantity</th><th>Unit</th></tr></thead><tbody>` +
        items.map(item => `<tr><td>${escapeHtml(item.name || item.category)}</td><td>${Number(item.quantity).toFixed(4)}</td><td>${escapeHtml(item.unit || '')}</td></tr>`).join('') +
        `</tbody></table>`;
    }

    function metric(label, value) {
      return `<div class="metric"><span class="meta">${escapeHtml(label)}</span><strong>${escapeHtml(String(value))}</strong></div>`;
    }

    function startPolling() {
      if (poll) return;
      poll = setInterval(async () => {
        await loadJobs();
        if (selectedId) await loadJob(selectedId);
      }, 2500);
    }

    function stopPolling() {
      if (!poll) return;
      clearInterval(poll);
      poll = null;
    }

    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[char]));
    }

    loadJobs();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
