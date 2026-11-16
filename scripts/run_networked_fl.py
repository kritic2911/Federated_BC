"""
Standalone runner script for Phase P4 Distributed Federated Learning.
Launches the FL Server and 4 Hospital Clients (vindr, cmmd, inbreast, cdd)
over networked sockets with Secure Aggregation, mTLS, and Prometheus telemetry.
"""

import argparse
import logging
import multiprocessing
import sys
import time
import urllib.request
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.systems.client_node import FLClientNode
from src.systems.server_node import FLServerNode
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Runner] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Runner")


def _run_server_process(cfg, overrides):
    server = FLServerNode(cfg, overrides=overrides)
    server.start()
    time.sleep(4.0)
    server.stop()


def _run_client_process(site, cfg, overrides, mock_training):
    client = FLClientNode(site=site, cfg=cfg, overrides=overrides, mock_training=mock_training)
    client.run()


def main():
    parser = argparse.ArgumentParser(description="Run Distributed Networked FL Simulation (Phase P4)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to master config YAML")
    parser.add_argument("--rounds", type=int, default=None, help="Communication rounds override")
    parser.add_argument("--min-k", type=int, default=None, help="Minimum cohort threshold k override")
    parser.add_argument("--port", type=int, default=None, help="Server port override")
    parser.add_argument("--metrics-port", type=int, default=None, help="Prometheus metrics port override")
    parser.add_argument("--enable-tls", action="store_true", help="Enable mutual TLS (mTLS)")
    parser.add_argument("--no-secagg", action="store_true", help="Disable secure aggregation masking")
    parser.add_argument("--real-training", action="store_true", help="Execute real local dataset training instead of mock updates")
    args = parser.parse_args()

    cfg = load_config(args.config)
    systems_cfg = cfg.get("systems", {})

    server_port = args.port or int(systems_cfg.get("server_port", 8080))
    metrics_port = args.metrics_port or int(systems_cfg.get("metrics_port", 9090))
    rounds = args.rounds or int(systems_cfg.get("num_rounds", 3))
    min_k = args.min_k or int(systems_cfg.get("min_cohort_k", 3))
    tls_enabled = args.enable_tls or bool(systems_cfg.get("tls_enabled", False))
    use_secagg = not args.no_secagg and bool(systems_cfg.get("use_secure_agg", True))

    logger.info("==================================================================")
    logger.info(">>> Launching Distributed Networked FL (Phase P4)")
    logger.info(">>> Server Port: %d | Metrics Port: %d | Rounds: %d", server_port, metrics_port, rounds)
    logger.info(">>> Mutual TLS: %s | Secure Aggregation: %s | Min-Cohort k: %d", tls_enabled, use_secagg, min_k)
    logger.info("==================================================================")

    server_overrides = {
        "server_host": "127.0.0.1",
        "server_port": server_port,
        "metrics_port": metrics_port,
        "num_rounds": rounds,
        "min_cohort_k": min_k,
        "tls_enabled": tls_enabled,
        "use_secure_agg": use_secagg,
    }

    # Start Server Process
    server_proc = multiprocessing.Process(
        target=_run_server_process,
        args=(cfg, server_overrides),
        daemon=True,
    )
    server_proc.start()
    time.sleep(1.5)  # Wait for server to bind and start listening

    # Start 4 Client Processes
    sites = ["vindr", "cmmd", "inbreast", "cdd"]
    client_procs = []

    for s in sites:
        client_overrides = {
            "server_host": "127.0.0.1",
            "server_port": server_port,
            "num_rounds": rounds,
            "tls_enabled": tls_enabled,
            "use_secure_agg": use_secagg,
        }
        p = multiprocessing.Process(
            target=_run_client_process,
            args=(s, cfg, client_overrides, not args.real_training),
            daemon=True,
        )
        client_procs.append(p)
        p.start()

    # Wait for all client processes to complete
    for p in client_procs:
        p.join()

    time.sleep(1.0)

    # Scrape Prometheus Telemetry Metrics
    logger.info("==================================================================")
    logger.info(">>> Distributed Communication Completed. Scraping Telemetry...")
    logger.info("==================================================================")
    try:
        url = f"http://127.0.0.1:{metrics_port}/metrics"
        with urllib.request.urlopen(url, timeout=5.0) as resp:
            metrics_content = resp.read().decode("utf-8")
            print("\n--- PROMETHEUS TELEMETRY EXPORT (/metrics) ---")
            print(metrics_content.strip())
            print("----------------------------------------------\n")
    except Exception as e:
        logger.warning("Could not scrape Prometheus metrics: %s", e)

    # Terminate server process cleanly
    server_proc.terminate()
    server_proc.join()
    logger.info("Distributed FL run finished.")


if __name__ == "__main__":
    main()
