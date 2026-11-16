#!/usr/bin/env python3
"""Generate Mutual TLS (mTLS) Certificates for Cross-Silo Federated Learning.

Adheres strictly to AGENTS.md Phase P4, Step 2:
  - Generates Root Certificate Authority (CA)
  - Generates Server Certificate with Subject Alternative Names (SANs)
  - Generates unique Client Certificates for each hospital silo
  - All paths, server hostnames, and site lists are configured via configs/config.yaml
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

_repo_dir = Path(__file__).resolve().parent.parent
if str(_repo_dir) not in sys.path:
    sys.path.insert(0, str(_repo_dir))

from src.utils.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_cmd(cmd: list[str]) -> str:
    """Run an OpenSSL command securely and log output."""
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error("Command failed: %s\nError: %s", " ".join(cmd), res.stderr)
        raise RuntimeError(f"OpenSSL command failed: {res.stderr}")
    return res.stdout


def generate_certificates(config_path: str = "configs/config.yaml", force: bool = False) -> Path:
    """Generate CA, Server, and Hospital Client certificates based on config.yaml."""
    cfg = load_config(config_path)
    sys_cfg = cfg.get("systems", {})
    data_cfg = cfg.get("data", {})

    cert_dir = Path(sys_cfg.get("cert_dir", "certs"))
    server_host = sys_cfg.get("server_host", "0.0.0.0")
    sites = sys_cfg.get("sites", data_cfg.get("all_sites", ["vindr", "cmmd", "inbreast", "cdd"]))

    cert_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating mTLS certificates in directory: %s", cert_dir.resolve())

    ca_key = cert_dir / "ca.key"
    ca_crt = cert_dir / "ca.crt"
    server_key = cert_dir / "server.key"
    server_csr = cert_dir / "server.csr"
    server_crt = cert_dir / "server.crt"
    server_ext = cert_dir / "server.ext"

    # --------------------------------------------------------------------------
    # 1. Root Certificate Authority (CA)
    # --------------------------------------------------------------------------
    if not ca_key.exists() or not ca_crt.exists() or force:
        logger.info("[1/3] Generating Root CA (ca.key, ca.crt)...")
        # 4096-bit RSA key for Root CA
        run_cmd(["openssl", "genrsa", "-out", str(ca_key), "4096"])
        run_cmd([
            "openssl", "req", "-x509", "-new", "-nodes",
            "-key", str(ca_key),
            "-sha256", "-days", "365",
            "-out", str(ca_crt),
            "-subj", "/C=US/ST=Healthcare/L=Medical/O=FederatedMammoCA/CN=FL-Root-CA"
        ])
        os.chmod(ca_key, 0o600)
    else:
        logger.info("[1/3] Existing Root CA found at %s (skipping, use --force to overwrite)", ca_crt)

    # --------------------------------------------------------------------------
    # 2. Server Certificate with Subject Alternative Names (SAN)
    # --------------------------------------------------------------------------
    if not server_key.exists() or not server_crt.exists() or force:
        logger.info("[2/3] Generating Server Certificate (server.key, server.crt)...")
        run_cmd(["openssl", "genrsa", "-out", str(server_key), "2048"])
        run_cmd([
            "openssl", "req", "-new",
            "-key", str(server_key),
            "-out", str(server_csr),
            "-subj", "/C=US/ST=Healthcare/L=Medical/O=FederatedMammo/CN=fl-server"
        ])

        # Create OpenSSL extensions file for SANs (allows Docker service name 'fl-server', 'localhost', and IPs)
        san_hosts = ["fl-server", "localhost"]
        if server_host not in ["0.0.0.0", ""]:
            san_hosts.append(server_host)

        ext_content = (
            "authorityKeyIdentifier=keyid,issuer\n"
            "basicConstraints=CA:FALSE\n"
            "keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment\n"
            "subjectAltName = @alt_names\n\n"
            "[alt_names]\n"
        )
        for i, h in enumerate(san_hosts, 1):
            ext_content += f"DNS.{i} = {h}\n"
        ext_content += "IP.1 = 127.0.0.1\n"

        with open(server_ext, "w", encoding="utf-8") as f:
            f.write(ext_content)

        run_cmd([
            "openssl", "x509", "-req",
            "-in", str(server_csr),
            "-CA", str(ca_crt),
            "-CAkey", str(ca_key),
            "-CAcreateserial",
            "-out", str(server_crt),
            "-days", "365",
            "-sha256",
            "-extfile", str(server_ext)
        ])
        os.chmod(server_key, 0o600)
    else:
        logger.info("[2/3] Existing Server certificate found at %s (skipping)", server_crt)

    # --------------------------------------------------------------------------
    # 3. Hospital Client Certificates (One per hospital silo)
    # --------------------------------------------------------------------------
    for site in sites:
        client_key = cert_dir / f"client_{site}.key"
        client_csr = cert_dir / f"client_{site}.csr"
        client_crt = cert_dir / f"client_{site}.crt"

        if not client_key.exists() or not client_crt.exists() or force:
            logger.info("[3/3] Generating Client Certificate for Site [%s]...", site)
            run_cmd(["openssl", "genrsa", "-out", str(client_key), "2048"])
            run_cmd([
                "openssl", "req", "-new",
                "-key", str(client_key),
                "-out", str(client_csr),
                "-subj", f"/C=US/ST=Healthcare/L=Hospital/O=HospitalSilo/CN=client-{site}"
            ])
            run_cmd([
                "openssl", "x509", "-req",
                "-in", str(client_csr),
                "-CA", str(ca_crt),
                "-CAkey", str(ca_key),
                "-CAcreateserial",
                "-out", str(client_crt),
                "-days", "365",
                "-sha256"
            ])
            os.chmod(client_key, 0o600)
        else:
            logger.info("[3/3] Existing Client certificate found for Site [%s] (skipping)", site)

    # --------------------------------------------------------------------------
    # 4. Verification Check
    # --------------------------------------------------------------------------
    logger.info("Verifying generated certificates against Root CA:")
    verify_server = run_cmd(["openssl", "verify", "-CAfile", str(ca_crt), str(server_crt)]).strip()
    logger.info("  Server Certificate Verification: %s", verify_server)

    for site in sites:
        c_crt = cert_dir / f"client_{site}.crt"
        v_client = run_cmd(["openssl", "verify", "-CAfile", str(ca_crt), str(c_crt)]).strip()
        logger.info("  Client [%s] Verification: %s", site, v_client)

    logger.info("Successfully generated and verified all mTLS certificates in '%s'.", cert_dir)
    return cert_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mTLS certificates for Cross-Silo FL.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml.")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing certificates.")
    args = parser.parse_args()

    generate_certificates(config_path=args.config, force=args.force)

