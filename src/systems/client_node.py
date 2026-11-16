"""
Distributed Federated Learning Client Node (Phase P4).
Runs as a hospital silo client process (e.g. client_vindr, client_cmmd, client_inbreast, client_cdd).
Handles mutual TLS authentication, local training/simulation, pairwise additive masking,
and failure/dropout injection based on hospital profile.
"""

import argparse
import logging
import os
import random
import socket
import ssl
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from torchvision.models import resnet18

from src.fl.client import FLClient
from src.systems.protocol import (
    MessageType,
    deserialize_tensor_dict,
    recv_msg,
    send_msg,
    serialize_tensor_dict,
)
from src.systems.secagg import (
    mask_model_weights,
)
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Client-%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)


class FLClientNode:
    """Hospital Silo Client Node communicating with central FL server."""

    def __init__(
        self,
        site: str,
        cfg: Dict[str, Any],
        overrides: Dict[str, Any] | None = None,
        mock_training: bool = True,
    ):
        self.site = site
        self.client_id = f"client_{site}"
        self.logger = logging.getLogger(site.upper())
        self.cfg = cfg
        self.systems_cfg = cfg.get("systems", {})
        self.mock_training = mock_training

        if overrides:
            self.systems_cfg.update(overrides)

        self.server_host = self.systems_cfg.get("server_host", "127.0.0.1")
        if self.server_host == "0.0.0.0":
            self.server_host = "127.0.0.1"
        self.server_port = int(self.systems_cfg.get("server_port", 8080))
        self.num_rounds = int(self.systems_cfg.get("num_rounds", 3))
        self.use_secure_agg = bool(self.systems_cfg.get("use_secure_agg", True))
        self.tls_enabled = bool(self.systems_cfg.get("tls_enabled", False))
        self.cert_dir = Path(self.systems_cfg.get("cert_dir", "certs"))

        # Profile and dropout rates
        site_profiles = self.systems_cfg.get("site_profiles", {})
        self.profile = site_profiles.get(site, "metro_hospital")
        dropout_rates = self.systems_cfg.get("dropout_rates", {})
        if site in dropout_rates:
            self.dropout_rate = float(dropout_rates[site])
        else:
            self.dropout_rate = float(dropout_rates.get(self.profile, 0.0))

        # Real PyTorch training initialization (if requested and dataset is mounted)
        self.fl_client = None
        if not self.mock_training:
            manifest_path = Path("data/manifest.csv")
            if manifest_path.exists():
                try:
                    df = pd.read_csv(manifest_path)
                    train_df = df[(df["site"] == self.site) & (df["split"] == "train")]
                    
                    max_samples = int(os.environ.get("MAX_SAMPLES", "0"))
                    if max_samples > 0 and len(train_df) > max_samples:
                        train_df = train_df.iloc[:max_samples]
                        self.logger.info("Restricted %s training partition to tiny sample of %d images", self.site, max_samples)

                    # Multi-container default: use CPU or GPU with small batch to prevent OOM
                    device_str = os.environ.get("TORCH_DEVICE", "cpu")
                    device = torch.device(device_str)
                    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
                    
                    self.fl_client = FLClient(
                        site=self.site,
                        train_df=train_df,
                        data_dir="data",
                        device=device,
                        batch_size=batch_size,
                    )
                    self.logger.info("Initialized real FLClient with %d local train images on %s (batch_size=%d)", len(train_df), device, batch_size)
                except Exception as e:
                    self.logger.warning("Could not initialize real FLClient (%s); falling back to mock", e)
                    self.mock_training = True
            else:
                self.logger.warning("Manifest data/manifest.csv not found; falling back to mock training")
                self.mock_training = True

        # Local Model
        self.model = resnet18(weights=None)
        self.model.fc = torch.nn.Linear(512, 3)
        self.local_weights = {k: v.cpu() for k, v in self.model.state_dict().items()}

        self._ssl_context: ssl.SSLContext | None = None
        if self.tls_enabled:
            self._setup_tls()

    def _setup_tls(self):
        """Configures mutual TLS client context."""
        client_cert = self.cert_dir / f"{self.client_id}.crt"
        client_key = self.cert_dir / f"{self.client_id}.key"
        ca_cert = self.cert_dir / "ca.crt"

        if not (client_cert.exists() and client_key.exists() and ca_cert.exists()):
            raise FileNotFoundError(
                f"Missing TLS certificates for {self.client_id} in {self.cert_dir}."
            )

        self._ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self._ssl_context.load_cert_chain(certfile=str(client_cert), keyfile=str(client_key))
        self._ssl_context.load_verify_locations(cafile=str(ca_cert))
        self._ssl_context.check_hostname = False
        self._ssl_context.verify_mode = ssl.CERT_REQUIRED
        self.logger.info("Mutual TLS initialized with client cert: %s", client_cert)

    def run(self):
        """Main loop connecting to server across all communication rounds."""
        self.logger.info(
            "Starting FL Client for site '%s' (Profile: %s, Dropout: %.1f%%, Target: %s:%d)",
            self.site,
            self.profile,
            self.dropout_rate * 100,
            self.server_host,
            self.server_port,
        )

        for r in range(1, self.num_rounds + 1):
            success = self._run_single_round(r)
            if not success:
                self.logger.warning("Round %d was aborted or skipped.", r)
            time.sleep(1.0)

    def _run_single_round(self, round_idx: int) -> bool:
        """Executes a single communication round connection with server."""
        sock = None
        try:
            # 1. Connect to Server
            raw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            raw_sock.settimeout(20.0)

            # Resilient retry while waiting for server startup (up to 60s)
            connected = False
            for attempt in range(60):
                try:
                    raw_sock.connect((self.server_host, self.server_port))
                    connected = True
                    break
                except (ConnectionRefusedError, OSError):
                    time.sleep(1.0)

            if not connected:
                self.logger.error("Could not reach server at %s:%d after 60s retry.", self.server_host, self.server_port)
                return False

            if self.tls_enabled and self._ssl_context:
                sock = self._ssl_context.wrap_socket(raw_sock, server_hostname=self.server_host)
            else:
                sock = raw_sock

            # 2. Phase 1: Send READY message
            self.logger.info("Connected to server. Sending READY for round %d...", round_idx)
            send_msg(
                sock,
                msg_type=MessageType.READY,
                metadata={"client_id": self.client_id, "site": self.site, "num_samples": 500},
            )

            # 3. Phase 2: Await START_UPLOAD or ABORT
            msg_type, meta, payload = recv_msg(sock)
            if msg_type == MessageType.ABORT:
                self.logger.warning("Round %d Aborted by Server: %s", round_idx, meta.get("reason"))
                return False

            if msg_type != MessageType.START_UPLOAD:
                self.logger.error("Unexpected message: %s", msg_type)
                return False

            cohort = meta.get("cohort", [])
            round_seed = meta.get("round_seed", round_idx * 100)
            global_weights = deserialize_tensor_dict(payload)
            self.logger.info("Phase 1 Succeeded. Confirmed cohort: %s. Training...", cohort)

            # 4. Check Dropout Failure Injection
            if self.dropout_rate > 0.0 and random.random() < self.dropout_rate:
                self.logger.warning("INJECTED FAILURE: Simulating network dropout for %s during upload!", self.client_id)
                # Intentionally drop connection before upload
                sock.close()
                return False

            # 5. Local Training Step (Simulated/Mock or PyTorch)
            updated_weights = self._perform_local_training(global_weights, round_idx=round_idx)

            # 6. Prepare and Send UPLOAD Payload
            if self.use_secure_agg:
                masked_vec, shapes, dtypes = mask_model_weights(
                    weights=updated_weights,
                    client_id=self.client_id,
                    cohort=cohort,
                    weight_factor=1.0,
                    round_seed=round_seed,
                )
                payload_bytes = masked_vec.numpy().tobytes()
            else:
                payload_bytes = serialize_tensor_dict(updated_weights)

            self.logger.info("Uploading local update (%d bytes) to server...", len(payload_bytes))
            send_msg(
                sock,
                msg_type=MessageType.UPLOAD,
                metadata={"client_id": self.client_id, "num_samples": 500},
                payload_bytes=payload_bytes,
            )

            # 7. Await COMMIT or ABORT
            msg_type, meta, _ = recv_msg(sock)
            if msg_type == MessageType.COMMIT:
                self.logger.info("Round %d committed successfully by server!", round_idx)
                return True
            elif msg_type == MessageType.ABORT:
                self.logger.warning("Round %d was aborted during Phase 2: %s", round_idx, meta.get("reason"))
                return False
            else:
                self.logger.warning("Received unexpected message %s after upload", msg_type)
                return False

        except Exception as e:
            self.logger.error("Communication error during round %d: %s", round_idx, e)
            return False
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    def _perform_local_training(
        self, global_weights: Dict[str, torch.Tensor], round_idx: int = 1
    ) -> Dict[str, torch.Tensor]:
        """Performs local training on site data or computes mock weight update."""
        if self.mock_training or self.fl_client is None:
            # Deterministic light perturbation to simulate gradient update
            updated = {}
            for k, v in global_weights.items():
                if v.is_floating_point():
                    updated[k] = v + (torch.randn_like(v) * 0.001)
                else:
                    updated[k] = v.clone()
            return updated
        else:
            self.logger.info("Executing real local training epoch for site %s (Round %d)...", self.site, round_idx)
            self.fl_client.set_global_weights(global_weights)
            updated_weights, n_samples, loss, _ = self.fl_client.train_round(round_idx=round_idx)
            self.logger.info("Local training epoch finished: loss=%.4f (n=%d)", loss, n_samples)
            return updated_weights


def parse_args():
    default_site = os.environ.get("SITE_NAME")
    default_config = os.environ.get("CONFIG_FILE", "configs/config.yaml")

    server_addr = os.environ.get("SERVER_ADDRESS", "")
    default_host = None
    default_port = None
    if server_addr and ":" in server_addr:
        parts = server_addr.split(":")
        default_host = parts[0]
        try:
            default_port = int(parts[1])
        except ValueError:
            pass
    elif server_addr:
        default_host = server_addr

    parser = argparse.ArgumentParser(description="Distributed FL Client Node (Phase P4)")
    parser.add_argument(
        "--site",
        type=str,
        default=default_site,
        required=(default_site is None),
        choices=["vindr", "cmmd", "inbreast", "cdd"],
        help="Hospital site identifier",
    )
    parser.add_argument("--config", type=str, default=default_config, help="Path to config.yaml")
    parser.add_argument("--server-host", type=str, default=default_host, help="Server IP/hostname override")
    parser.add_argument("--server-port", type=int, default=default_port, help="Server port override")
    parser.add_argument("--rounds", type=int, default=None, help="Number of rounds")
    parser.add_argument("--dropout", type=float, default=None, help="Force specific dropout probability")
    parser.add_argument("--no-tls", action="store_true", help="Disable TLS")
    parser.add_argument("--no-secagg", action="store_true", help="Disable secure aggregation masking")
    parser.add_argument("--real-training", action="store_true", help="Run real training instead of mock updates")
    parser.add_argument("--max-samples", type=int, default=None, help="Restrict partition to tiny sample size")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_samples is not None:
        os.environ["MAX_SAMPLES"] = str(args.max_samples)
    cfg = load_config(args.config)

    overrides = {}
    if args.server_host:
        overrides["server_host"] = args.server_host
    if args.server_port:
        overrides["server_port"] = args.server_port
    if args.rounds:
        overrides["num_rounds"] = args.rounds
    if args.no_tls:
        overrides["tls_enabled"] = False
    if args.no_secagg:
        overrides["use_secure_agg"] = False
    if args.dropout is not None:
        overrides.setdefault("dropout_rates", {})[args.site] = args.dropout

    real_training = args.real_training or (os.environ.get("REAL_TRAINING", "0").lower() in ("1", "true", "yes"))

    client = FLClientNode(
        site=args.site,
        cfg=cfg,
        overrides=overrides,
        mock_training=not real_training,
    )
    client.run()


if __name__ == "__main__":
    main()
