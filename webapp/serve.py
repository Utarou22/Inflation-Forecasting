from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from io import StringIO
from pathlib import Path
import json
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DATA_DIR = ROOT / "data"
DEFAULT_DASHBOARD_PATH = DATA_DIR / "dashboard.json"
DEFAULT_RESULTS_PATH = DATA_DIR / "dashboard_results.csv"
UPLOAD_RESULTS_PATH = DATA_DIR / "uploaded_results.csv"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import webapp_export as exporter


class WebAppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.path == "/api/dashboard":
            self.serve_dashboard()
            return

        super().do_GET()

    def do_POST(self):
        if self.path == "/api/upload-csv":
            self.handle_csv_upload()
            return

        self.send_error(404, "Not found")

    def serve_dashboard(self):
        if not DEFAULT_DASHBOARD_PATH.exists():
            payload, export_frame = exporter.build_dashboard_payload()
            exporter.write_dashboard_payload(payload, DEFAULT_DASHBOARD_PATH)
            exporter.export_results_csv(export_frame, DEFAULT_RESULTS_PATH)

        payload = json.loads(DEFAULT_DASHBOARD_PATH.read_text(encoding="utf-8"))
        export_url = "/data/dashboard_results.csv" if DEFAULT_RESULTS_PATH.exists() else None
        self.send_json({"payload": payload, "export_url": export_url})

    def handle_csv_upload(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise ValueError("Upload body is empty.")

            csv_text = self.rfile.read(content_length).decode("utf-8-sig")
            uploaded_df = pd.read_csv(StringIO(csv_text))

            payload, export_frame = exporter.build_dashboard_payload(main_df=uploaded_df)
            exporter.export_results_csv(export_frame, UPLOAD_RESULTS_PATH)

            self.send_json({
                "payload": payload,
                "export_url": "/data/uploaded_results.csv",
                "message": "Upload processed successfully.",
            })
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8000), WebAppHandler)
    print("Serving webapp at http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
