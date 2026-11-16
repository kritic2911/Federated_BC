"""
Distributed Federated Learning Server Node (Phase P4).
Coordinates the two-phase distributed round protocol with minimum-cohort guard (k),
mutual TLS authentication, secure aggregation, and Prometheus telemetry.
"""

import argparse
import json
import logging
import os
import select
import socket
import ssl
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torchvision.models import resnet18

from src.systems.metrics import PrometheusMetricsExporter
from src.systems.protocol import (
    MessageType,
    deserialize_tensor_dict,
    recv_msg,
    send_msg,
    serialize_tensor_dict,
)
from src.systems.secagg import (
    flatten_state_dict,
    unmask_and_aggregate,
)
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [FL-Server] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FL-Server")


class FLServerNode:
    """Central Aggregation Server managing networked hospital silos."""

    def __init__(self, cfg: Dict[str, Any], overrides: Dict[str, Any] | None = None):
        self.cfg = cfg
        self.systems_cfg = cfg.get("systems", {})

        # Apply CLI overrides if present
        if overrides:
            self.systems_cfg.update(overrides)

        self.host = self.systems_cfg.get("server_host", "0.0.0.0")
        self.port = int(self.systems_cfg.get("server_port", 8080))
        self.metrics_port = int(self.systems_cfg.get("metrics_port", 9090))
        self.num_rounds = int(self.systems_cfg.get("num_rounds", 3))
        self.min_cohort_k = int(self.systems_cfg.get("min_cohort_k", 3))
        self.t_ready_s = float(self.systems_cfg.get("t_ready_s", 15.0))
        self.t_upload_s = float(self.systems_cfg.get("t_upload_s", 90.0))
        self.use_secure_agg = bool(self.systems_cfg.get("use_secure_agg", True))
        self.tls_enabled = bool(self.systems_cfg.get("tls_enabled", False))
        self.cert_dir = Path(self.systems_cfg.get("cert_dir", "certs"))
        self.telemetry_path = Path("results/systems_telemetry.jsonl")

        # Initialize Prometheus Exporter
        self.exporter = PrometheusMetricsExporter(host="0.0.0.0", port=self.metrics_port)

        # Initialize Global Model
        self.model = resnet18(weights=None)
        self.model.fc = torch.nn.Linear(512, 3)
        self.global_weights = {k: v.cpu() for k, v in self.model.state_dict().items()}
        _, self.param_shapes, self.param_dtypes = flatten_state_dict(self.global_weights)

        self._server_sock: socket.socket | None = None
        self._ssl_context: ssl.SSLContext | None = None

        if self.tls_enabled:
            self._setup_tls()

    def _setup_tls(self):
        """Configures mutual TLS (mTLS) context requiring valid client certificates."""
        server_cert = self.cert_dir / "server.crt"
        server_key = self.cert_dir / "server.key"
        ca_cert = self.cert_dir / "ca.crt"

        if not (server_cert.exists() and server_key.exists() and ca_cert.exists()):
            raise FileNotFoundError(
                f"Missing TLS certificates in {self.cert_dir}. Run scripts/generate_tls_certs.py first."
            )

        self._ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        self._ssl_context.load_cert_chain(certfile=str(server_cert), keyfile=str(server_key))
        self._ssl_context.load_verify_locations(cafile=str(ca_cert))
        self._ssl_context.verify_mode = ssl.CERT_REQUIRED
        logger.info("Mutual TLS configured successfully with CA: %s", ca_cert)

    def start(self):
        """Starts the server listener and enters the distributed FL round execution loop."""
        self.exporter.start()

        raw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        raw_sock.bind((self.host, self.port))
        raw_sock.listen(10)
        self._server_sock = raw_sock

        logger.info("Server listening on %s:%d (TLS: %s, SecAgg: %s, k: %d)",
                    self.host, self.port, self.tls_enabled, self.use_secure_agg, self.min_cohort_k)

        try:
            self._run_round_loop()
        except Exception as e:
            logger.error("Error in server round loop: %s", e)

    def stop(self):
        """Stops server and closes all sockets."""
        if self._server_sock:
            self._server_sock.close()
            self._server_sock = None
        self.exporter.stop()
        logger.info("Server shutdown complete.")

    def _run_round_loop(self):
        """Coordinates communication rounds over connected clients."""
        for r in range(1, self.num_rounds + 1):
            round_start_t = time.time()
            self.exporter.state.update_round_start(r)
            logger.info("==================================================")
            logger.info(">>> Commencing Communication Round %d / %d", r, self.num_rounds)
            logger.info("==================================================")

            # Phase 1: Wait for client connections and READY signals
            phase1_start = time.time()
            active_clients = self._phase1_collect_ready(round_idx=r, timeout_s=self.t_ready_s)
            phase1_duration = time.time() - phase1_start
            self.exporter.state.update_phase1(phase1_duration, len(active_clients))

            # Minimum-Cohort Guard Check at Phase 1
            if len(active_clients) < self.min_cohort_k:
                abort_msg = (
                    f"Phase 1 Guard Violated: Only {len(active_clients)} clients ready; "
                    f"minimum cohort k={self.min_cohort_k} required."
                )
                logger.warning(abort_msg)
                self.exporter.state.record_abort(abort_msg)
                self._broadcast_abort(active_clients, abort_msg)
                self._log_round_telemetry(
                    round_idx=r,
                    phase1_s=phase1_duration,
                    phase2_s=0.0,
                    total_s=time.time() - round_start_t,
                    ready_cohort=len(active_clients),
                    upload_cohort=0,
                    dropouts=0,
                    bytes_round=0,
                    aborted=True,
                    abort_reason=abort_msg,
                )
                time.sleep(2.0)
                continue

            logger.info("Phase 1 Complete: Cohort of %d clients verified >= k=%d in %.2fs",
                        len(active_clients), self.min_cohort_k, phase1_duration)

            # Phase 2: Send global weights, receive local/masked updates
            phase2_start = time.time()
            surviving_updates, total_bytes = self._phase2_collect_updates(
                round_idx=r,
                cohort_clients=active_clients,
                timeout_s=self.t_upload_s,
            )
            phase2_duration = time.time() - phase2_start
            self.exporter.state.update_phase2(phase2_duration, len(surviving_updates), total_bytes)

            # Check dropouts and Minimum-Cohort Guard at Phase 2
            dropouts = len(active_clients) - len(surviving_updates)
            if dropouts > 0:
                logger.warning("Phase 2 Dropouts Detected: %d client(s) failed to upload updates.", dropouts)
                self.exporter.state.record_dropouts(dropouts)

            if len(surviving_updates) < self.min_cohort_k:
                abort_msg = (
                    f"Phase 2 Guard Violated: Only {len(surviving_updates)} updates received "
                    f"after {dropouts} dropout(s); min k={self.min_cohort_k} violated."
                )
                logger.warning(abort_msg)
                self.exporter.state.record_abort(abort_msg)
                self._broadcast_abort(active_clients, abort_msg)
                self._log_round_telemetry(
                    round_idx=r,
                    phase1_s=phase1_duration,
                    phase2_s=phase2_duration,
                    total_s=time.time() - round_start_t,
                    ready_cohort=len(active_clients),
                    upload_cohort=len(surviving_updates),
                    dropouts=dropouts,
                    bytes_round=total_bytes,
                    aborted=True,
                    abort_reason=abort_msg,
                )
                time.sleep(2.0)
                continue

            # Aggregation Phase
            logger.info("Aggregating %d updates (SecAgg: %s)...", len(surviving_updates), self.use_secure_agg)
            self._aggregate_updates(surviving_updates)

            # Commit Phase
            self._broadcast_commit(active_clients, round_idx=r)

            round_duration = time.time() - round_start_t
            self.exporter.state.update_round_end(round_duration)
            self._log_round_telemetry(
                round_idx=r,
                phase1_s=phase1_duration,
                phase2_s=phase2_duration,
                total_s=round_duration,
                ready_cohort=len(active_clients),
                upload_cohort=len(surviving_updates),
                dropouts=dropouts,
                bytes_round=total_bytes,
                aborted=False,
                abort_reason="none",
            )
            logger.info("Round %d Committed Successfully in %.2fs total", r, round_duration)
            time.sleep(1.0)

    def _log_round_telemetry(
        self,
        round_idx: int,
        phase1_s: float,
        phase2_s: float,
        total_s: float,
        ready_cohort: int,
        upload_cohort: int,
        dropouts: int,
        bytes_round: int,
        aborted: bool,
        abort_reason: str,
    ):
        """Appends structured telemetry for the round to results/systems_telemetry.jsonl."""
        record = {
            "round": round_idx,
            "timestamp": time.time(),
            "phase1_duration_s": round(phase1_s, 4),
            "phase2_duration_s": round(phase2_s, 4),
            "total_round_duration_s": round(total_s, 4),
            "ready_cohort_size": ready_cohort,
            "upload_cohort_size": upload_cohort,
            "dropouts": dropouts,
            "min_cohort_k": self.min_cohort_k,
            "bytes_transferred": bytes_round,
            "cumulative_bytes": self.exporter.state.bytes_transferred_total,
            "use_secure_agg": self.use_secure_agg,
            "tls_enabled": self.tls_enabled,
            "aborted": aborted,
            "abort_reason": abort_reason,
        }
        try:
            self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.telemetry_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except (PermissionError, OSError) as e:
            logger.warning("Could not append to telemetry file %s: %s", self.telemetry_path, e)

    def _phase1_collect_ready(self, round_idx: int, timeout_s: float) -> Dict[str, socket.socket]:
        """Waits for clients to connect and send READY within timeout_s."""
        deadline = time.time() + timeout_s
        ready_clients: Dict[str, socket.socket] = {}

        logger.info("Waiting up to %.1fs for clients to announce READY...", timeout_s)

        while time.time() < deadline:
            remaining = max(0.1, deadline - time.time())
            readable, _, _ = select.select([self._server_sock], [], [], remaining)
            if not readable:
                continue

            client_sock, addr = self._server_sock.accept()
            conn = client_sock
            client_id = f"unknown_{addr[0]}_{addr[1]}"

            if self.tls_enabled and self._ssl_context:
                try:
                    conn = self._ssl_context.wrap_socket(client_sock, server_side=True)
                    peer_cert = conn.getpeercert()
                    # Extract Common Name (CN) from subject
                    for field in peer_cert.get("subject", ()):
                        for k, v in field:
                            if k == "commonName":
                                client_id = v
                except Exception as e:
                    logger.error("TLS Handshake failure from %s: %s", addr, e)
                    client_sock.close()
                    continue

            # Wait for READY message from newly connected client
            try:
                conn.settimeout(3.0)
                msg_type, meta, _ = recv_msg(conn)
                if msg_type == MessageType.READY:
                    cid = meta.get("client_id", client_id)
                    ready_clients[cid] = conn
                    logger.info("Received READY from client: '%s' (%s)", cid, addr)
                else:
                    logger.warning("Expected READY, received %s from %s", msg_type, addr)
                    conn.close()
            except Exception as e:
                logger.error("Failed receiving initial READY from %s: %s", addr, e)
                conn.close()

            # Early exit if we already have full expected cohort (all 4 sites)
            if len(ready_clients) >= 4:
                break

        return ready_clients

    def _phase2_collect_updates(
        self,
        round_idx: int,
        cohort_clients: Dict[str, socket.socket],
        timeout_s: float,
    ) -> Tuple[Dict[str, Dict[str, Any]], int]:
        """Sends START_UPLOAD and global weights, then collects client UPLOADs."""
        cohort_list = sorted(list(cohort_clients.keys()))
        weights_bytes = serialize_tensor_dict(self.global_weights)
        total_wire_bytes = 0

        # Broadcast START_UPLOAD with cohort list and global model
        for cid, conn in list(cohort_clients.items()):
            try:
                sent_b = send_msg(
                    conn,
                    msg_type=MessageType.START_UPLOAD,
                    metadata={"cohort": cohort_list, "round": round_idx, "round_seed": round_idx * 100},
                    payload_bytes=weights_bytes,
                )
                total_wire_bytes += sent_b
            except Exception as e:
                logger.error("Failed sending START_UPLOAD to %s: %s", cid, e)
                cohort_clients.pop(cid, None)

        # Await UPLOAD from each client
        surviving_updates: Dict[str, Dict[str, Any]] = {}
        deadline = time.time() + timeout_s

        for cid, conn in list(cohort_clients.items()):
            remaining = max(0.5, deadline - time.time())
            conn.settimeout(remaining)
            try:
                msg_type, meta, payload = recv_msg(conn)
                total_wire_bytes += len(payload)
                if msg_type == MessageType.UPLOAD:
                    if self.use_secure_agg:
                        # Payload is serialized 1D flat masked tensor (float32 bytes)
                        flat_tensor = torch.frombuffer(bytearray(payload), dtype=torch.float32).clone()
                        surviving_updates[cid] = {
                            "masked_flat": flat_tensor,
                            "num_samples": meta.get("num_samples", 1),
                        }
                    else:
                        tensors = deserialize_tensor_dict(payload)
                        surviving_updates[cid] = {
                            "weights": tensors,
                            "num_samples": meta.get("num_samples", 1),
                        }
                    logger.info("Received UPLOAD from '%s' (%d bytes payload)", cid, len(payload))
                else:
                    logger.warning("Client '%s' sent unexpected %s during upload phase", cid, msg_type)
            except Exception as e:
                logger.warning("Dropout: Client '%s' failed to upload update within timeout: %s", cid, e)

        return surviving_updates, total_wire_bytes

    def _aggregate_updates(self, updates: Dict[str, Dict[str, Any]]):
        """Aggregates received updates into new global model weights."""
        cohort = sorted(list(updates.keys()))
        if self.use_secure_agg:
            masked_list = [updates[c]["masked_flat"] for c in cohort]
            total_weight = float(len(cohort))
            new_weights = unmask_and_aggregate(
                masked_updates=masked_list,
                cohort=cohort,
                min_cohort_k=self.min_cohort_k,
                shapes=self.param_shapes,
                dtypes=self.param_dtypes,
                total_weight=total_weight,
            )
            self.global_weights = new_weights
        else:
            # Plain FedAvg average
            new_weights = {}
            for k in self.global_weights:
                new_weights[k] = sum(updates[c]["weights"][k] for c in cohort) / len(cohort)
            self.global_weights = new_weights

    def _broadcast_abort(self, clients: Dict[str, socket.socket], reason: str):
        """Broadcasts ABORT to all connected clients."""
        for cid, conn in list(clients.items()):
            try:
                send_msg(conn, MessageType.ABORT, metadata={"reason": reason})
                conn.close()
            except Exception:
                pass
        clients.clear()

    def _broadcast_commit(self, clients: Dict[str, socket.socket], round_idx: int):
        """Broadcasts COMMIT to all surviving clients."""
        for cid, conn in list(clients.items()):
            try:
                send_msg(conn, MessageType.COMMIT, metadata={"round": round_idx})
                conn.close()
            except Exception:
                pass
        clients.clear()


def parse_args():
    default_config = os.environ.get("CONFIG_FILE", "configs/config.yaml")
    parser = argparse.ArgumentParser(description="Distributed FL Server Node (Phase P4)")
    parser.add_argument("--config", type=str, default=default_config, help="Path to config.yaml")
    parser.add_argument("--port", type=int, default=None, help="Listening port override")
    parser.add_argument("--metrics-port", type=int, default=None, help="Metrics exporter port override")
    parser.add_argument("--rounds", type=int, default=None, help="Number of FL rounds")
    parser.add_argument("--min-k", type=int, default=None, help="Minimum cohort threshold k")
    parser.add_argument("--no-tls", action="store_true", help="Disable mutual TLS")
    parser.add_argument("--no-secagg", action="store_true", help="Disable secure aggregation masking")
    parser.add_argument("--exit-on-finish", action="store_true", help="Exit immediately when rounds finish (for tests)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    overrides = {}
    if args.port:
        overrides["server_port"] = args.port
    if args.metrics_port:
        overrides["metrics_port"] = args.metrics_port
    if args.rounds:
        overrides["num_rounds"] = args.rounds
    if args.min_k:
        overrides["min_cohort_k"] = args.min_k
    if args.no_tls:
        overrides["tls_enabled"] = False
    if args.no_secagg:
        overrides["use_secure_agg"] = False

    server = FLServerNode(cfg, overrides=overrides)
    server.start()

    if not args.exit_on_finish:
        logger.info(
            "FL communication rounds completed. Keeping Prometheus exporter live on port %d (Ctrl+C to stop)...",
            server.metrics_port,
        )
        try:
            while True:
                time.sleep(30)
        except KeyboardInterrupt:
            server.stop()


if __name__ == "__main__":
    main()
