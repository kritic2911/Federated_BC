"""
Lightweight Prometheus Telemetry Exporter for Phase P4 Systems Experiments.
Serves Prometheus-compatible metrics on HTTP port 9090 using standard library http.server.
"""

import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger(__name__)


class MetricsState:
    """Thread-safe state container for FL systems telemetry metrics."""

    def __init__(self):
        self._lock = threading.Lock()
        self.round_current: int = 0
        self.cohort_size: int = 0
        self.aborted_rounds_total: int = 0
        self.dropouts_total: int = 0
        self.bytes_transferred_total: int = 0
        self.round_duration_seconds: float = 0.0
        self.phase1_duration_seconds: float = 0.0
        self.phase2_duration_seconds: float = 0.0
        self.last_abort_reason: str = "none"

    def update_round_start(self, round_num: int):
        with self._lock:
            self.round_current = round_num

    def update_phase1(self, duration_s: float, cohort_size: int):
        with self._lock:
            self.phase1_duration_seconds = duration_s
            self.cohort_size = cohort_size

    def update_phase2(self, duration_s: float, surviving_size: int, bytes_added: int):
        with self._lock:
            self.phase2_duration_seconds = duration_s
            self.cohort_size = surviving_size
            self.bytes_transferred_total += bytes_added

    def record_abort(self, reason: str):
        with self._lock:
            self.aborted_rounds_total += 1
            self.last_abort_reason = reason

    def record_dropouts(self, count: int):
        with self._lock:
            self.dropouts_total += count

    def update_round_end(self, total_duration_s: float):
        with self._lock:
            self.round_duration_seconds = total_duration_s

    def render_prometheus_text(self) -> str:
        """Formats the current telemetry metrics into standard Prometheus exposition format."""
        with self._lock:
            lines = [
                "# HELP fl_round_current Current communication round index",
                "# TYPE fl_round_current gauge",
                f"fl_round_current {self.round_current}",
                "",
                "# HELP fl_cohort_size Number of actively participating hospital clients",
                "# TYPE fl_cohort_size gauge",
                f"fl_cohort_size {self.cohort_size}",
                "",
                "# HELP fl_aborted_rounds_total Total count of rounds aborted due to cohort guard or timeout",
                "# TYPE fl_aborted_rounds_total counter",
                f"fl_aborted_rounds_total {self.aborted_rounds_total}",
                "",
                "# HELP fl_dropouts_total Total count of client dropouts",
                "# TYPE fl_dropouts_total counter",
                f"fl_dropouts_total {self.dropouts_total}",
                "",
                "# HELP fl_round_duration_seconds Duration in seconds of the last communication round",
                "# TYPE fl_round_duration_seconds gauge",
                f"fl_round_duration_seconds {self.round_duration_seconds:.4f}",
                "",
                "# HELP fl_phase1_duration_seconds Duration in seconds of Phase 1 (Announcement to Ready)",
                "# TYPE fl_phase1_duration_seconds gauge",
                f"fl_phase1_duration_seconds {self.phase1_duration_seconds:.4f}",
                "",
                "# HELP fl_phase2_duration_seconds Duration in seconds of Phase 2 (Upload and Aggregation)",
                "# TYPE fl_phase2_duration_seconds gauge",
                f"fl_phase2_duration_seconds {self.phase2_duration_seconds:.4f}",
                "",
                "# HELP fl_bytes_transferred_total Total cumulative wire payload bytes exchanged",
                "# TYPE fl_bytes_transferred_total counter",
                f"fl_bytes_transferred_total {self.bytes_transferred_total}",
                "",
            ]
            return "\n".join(lines)


class _PrometheusHTTPHandler(BaseHTTPRequestHandler):
    state: MetricsState

    def do_GET(self):
        if self.path == "/metrics":
            output = self.state.render_prometheus_text().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(output)))
            self.end_headers()
            self.wfile.write(output)
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK\n")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence standard HTTP request logging to keep console clean
        pass


class PrometheusMetricsExporter:
    """Background HTTP server exposing Prometheus metrics on port 9090."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9090):
        self.host = host
        self.port = port
        self.state = MetricsState()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self):
        handler = _PrometheusHTTPHandler
        handler.state = self.state
        self._server = HTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("Prometheus metrics exporter started at http://%s:%d/metrics", self.host, self.port)

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            logger.info("Prometheus metrics exporter stopped.")
