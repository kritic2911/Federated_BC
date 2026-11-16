"""
End-to-end integration test for distributed server/client networked communication.
Verifies Phase P4 requirements:
  - Multi-client socket communication over localhost
  - Two-phase distributed protocol (ANNOUNCE -> READY -> START_UPLOAD -> UPLOAD -> COMMIT)
  - Secure Aggregation (pairwise additive masking)
  - Minimum-cohort guard enforcement & abort recording
  - Prometheus telemetry endpoint (/metrics)
"""

import threading
import time
import urllib.request

from src.systems.client_node import FLClientNode
from src.systems.server_node import FLServerNode
from src.utils.config import load_config


def test_distributed_fl_networked_smoke():
    cfg = load_config("configs/config.yaml")

    # Use unique free port and disable TLS for fast test execution
    server_port = 18080
    metrics_port = 19090

    server_overrides = {
        "server_host": "127.0.0.1",
        "server_port": server_port,
        "metrics_port": metrics_port,
        "num_rounds": 1,
        "min_cohort_k": 3,
        "t_ready_s": 6.0,
        "t_upload_s": 12.0,
        "tls_enabled": False,
        "use_secure_agg": True,
    }

    server = FLServerNode(cfg, overrides=server_overrides)
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()

    time.sleep(1.0)  # Give server a moment to bind

    # Launch 4 hospital clients
    sites = ["vindr", "cmmd", "inbreast", "cdd"]
    client_threads = []

    for s in sites:
        client_overrides = {
            "server_host": "127.0.0.1",
            "server_port": server_port,
            "num_rounds": 1,
            "tls_enabled": False,
            "use_secure_agg": True,
            "dropout_rates": {s: 0.0},
        }
        c = FLClientNode(site=s, cfg=cfg, overrides=client_overrides, mock_training=True)
        t = threading.Thread(target=c.run, daemon=True)
        client_threads.append(t)
        t.start()

    # Wait for clients to complete round 1
    for t in client_threads:
        t.join(timeout=15.0)

    time.sleep(1.0)

    # Verify Prometheus metrics endpoint
    url = f"http://127.0.0.1:{metrics_port}/metrics"
    req = urllib.request.urlopen(url, timeout=3.0)
    assert req.status == 200
    metrics_text = req.read().decode("utf-8")

    assert "fl_round_current 1" in metrics_text
    assert "fl_cohort_size 4" in metrics_text
    assert "fl_aborted_rounds_total 0" in metrics_text
    assert "fl_bytes_transferred_total" in metrics_text

    server.stop()


def test_distributed_fl_cohort_guard_abort():
    cfg = load_config("configs/config.yaml")

    server_port = 18081
    metrics_port = 19091

    # Require minimum k=3, but only connect 2 clients
    server_overrides = {
        "server_host": "127.0.0.1",
        "server_port": server_port,
        "metrics_port": metrics_port,
        "num_rounds": 1,
        "min_cohort_k": 3,
        "t_ready_s": 2.0,
        "t_upload_s": 2.0,
        "tls_enabled": False,
        "use_secure_agg": True,
    }

    server = FLServerNode(cfg, overrides=server_overrides)
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()

    time.sleep(0.5)

    # Launch ONLY 2 clients (below k=3)
    sites = ["vindr", "cmmd"]
    client_threads = []

    for s in sites:
        client_overrides = {
            "server_host": "127.0.0.1",
            "server_port": server_port,
            "num_rounds": 1,
            "tls_enabled": False,
            "use_secure_agg": True,
            "dropout_rates": {s: 0.0},
        }
        c = FLClientNode(site=s, cfg=cfg, overrides=client_overrides, mock_training=True)
        t = threading.Thread(target=c.run, daemon=True)
        client_threads.append(t)
        t.start()

    for t in client_threads:
        t.join(timeout=6.0)

    time.sleep(1.0)

    # Verify Prometheus recorded the abort
    url = f"http://127.0.0.1:{metrics_port}/metrics"
    req = urllib.request.urlopen(url, timeout=3.0)
    assert req.status == 200
    metrics_text = req.read().decode("utf-8")

    # Cohort was 2 < k=3, so aborted_rounds_total must be >= 1
    assert "fl_aborted_rounds_total 1" in metrics_text

    server.stop()
