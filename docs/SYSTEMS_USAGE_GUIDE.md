# Systems & Telemetry Usage Guide (Phase P4 & P5)
**Cross-Silo Federated Learning for Digital Mammography**

This guide documents the distributed systems architecture, network emulation, mutual TLS (mTLS) authentication, secure aggregation, and telemetry visualization across both **Docker Multi-Container Mesh** and **Native Localhost Sockets**.

---

## 1. Architecture & Network Topology

The systems evaluation models 4 heterogeneous digital mammography hospital silos communicating with a central aggregation server over an isolated bridge network (`fl-net`).

```
                              ┌────────────────────────────────────────┐
                              │         fl-server (Coordinator)        │
                              │   - Port 8080: Wire Socket / mTLS      │
                              │   - Port 9090: Prometheus Metrics      │
                              │   - Minimum-Cohort Guard (k = 3)       │
                              └───────────────────┬────────────────────┘
                                                  │
                      ┌───────────────────────────┼───────────────────────────┐
                      │                           │                           │
          Metro Silos (Low Latency)               │              District Silos (High Latency)
                      │                           │                           │
         ┌────────────┴────────────┐              │              ┌────────────┴────────────┐
         ▼                         ▼              │              ▼                         ▼
  ┌─────────────┐           ┌─────────────┐       │       ┌─────────────┐           ┌─────────────┐
  │client-vindr │           │ client-cmmd │       │       │client-inbr. │           │ client-cdd  │
  │ Siemens     │           │ GE          │       │       │ Siemens     │           │ GE          │
  │ Inspiration │           │ Senographe  │       │       │ Novation    │           │ Contrast DM │
  │ 5ms / 1ms   │           │ 5ms / 1ms   │       │       │ 120ms/35ms  │           │ 120ms/35ms  │
  └─────────────┘           └─────────────┘       │       └─────────────┘           └─────────────┘
                                                  ▼
                                      ┌───────────────────────┐
                                      │ fl-grafana (Port 3000)│
                                      │ Live Dashboards       │
                                      └───────────────────────┘
```

---

## 2. Docker Multi-Container Execution

The Docker environment runs all 5 nodes + Grafana inside lightweight Debian-based containers. Traffic shaping (`tc netem`) is applied inside client containers at startup via `entrypoint_client.sh`.

### Prerequisites
- Docker & Docker Compose (`docker compose version` $\ge$ 2.20)
- Linux Kernel support for `sch_netem` (enabled by default on Docker Desktop and standard Linux kernels).

### Step-by-Step Instructions

1. **Generate Cryptographic Mutual TLS Certificates**:
   ```bash
   python scripts/generate_tls_certs.py
   ```
   *Generates 4096-bit Root CA, Server Certificate, and individual client certificates (`client_vindr`, `client_cmmd`, `client_inbreast`, `client_cdd`) under `certs/`.*

2. **Launch the Distributed Mesh**:
   ```bash
   docker compose up --build
   ```
   *Spawns `fl-server`, 4 hospital clients, and `fl-grafana`.*

3. **Verify Network Emulation Inside a Container**:
   You can verify that `tc netem` was properly applied inside a district client container:
   ```bash
   docker compose exec client-inbreast tc qdisc show dev eth0
   ```
   *Expected output: `qdisc netem 8001: root refcnt 2 limit 1000 delay 120ms 35ms loss 2%`.*

4. **Monitor Client Logs & Injected Dropouts**:
   ```bash
   docker compose logs -f fl-server
   docker compose logs -f client-cdd
   ```

5. **Stop Containers**:
   ```bash
   docker compose down
   ```

---

## 3. Native Localhost Execution (No Docker Required)

If Docker is unavailable, the standalone orchestrator launches the server and clients over localhost TCP sockets, emulating the two-phase protocol and failure injection:

```bash
# Standard 3-round simulation with Secure Aggregation
python scripts/run_networked_fl.py --rounds 3

# Benchmark with custom cohort guard threshold (k=2)
python scripts/run_networked_fl.py --rounds 3 --min-k 2

# Measure bandwidth difference without SecAgg (Plaintext baseline)
python scripts/run_networked_fl.py --rounds 3 --no-secagg
```

---

## 4. Telemetry, Dashboards & Visualizations

### 1. Prometheus Telemetry Endpoint
While the server node is running, inspect metrics directly:
```bash
curl http://localhost:9090/metrics
```
Key metrics exposed:
- `fl_round_current`: Current communication round index.
- `fl_cohort_size`: Active surviving hospital clients.
- `fl_aborted_rounds_total`: Rounds aborted due to Minimum-Cohort Guard violation ($|U| < k$).
- `fl_dropouts_total`: Count of clients that failed during upload.
- `fl_round_duration_seconds`: Total round wall-clock duration.
- `fl_phase1_duration_seconds`: Time spent collecting `READY` signals.
- `fl_phase2_duration_seconds`: Time spent collecting and unmasking uploads.
- `fl_bytes_transferred_total`: Cumulative payload bytes exchanged.

### 2. Live Grafana Dashboard (Docker Mode)
If running via Docker Compose, access the pre-configured Grafana web dashboard:
- **URL**: [http://localhost:3000](http://localhost:3000)
- **Username**: `admin`
- **Password**: `admin`
- Dashboard file: `docker/grafana_dashboard.json`

### 3. Standalone Interactive HTML Dashboard
Open [`paper/figures/systems_dashboard.html`](file:///home/kc/breastFed/paper/figures/systems_dashboard.html) in any web browser. It requires zero setup and interactively renders:
- Live KPI status cards.
- Per-round MB payload transfer (SecAgg vs Plaintext).
- Phase 1 vs Phase 2 round duration breakdown.
- Client cohort dynamics and auto-abort boundaries.

---

## 5. Generating Publication Transfer Figures

To generate high-resolution PNG graphs for the paper and presentation slides:

```bash
python scripts/plot_systems_telemetry.py
```

Generated outputs:
1. `paper/figures/systems_transfer_bandwidth.png` (Per-round MB payload & cumulative GB).
2. `paper/figures/systems_latency_breakdown.png` (Latency decomposition & netem jitter).
3. `paper/figures/systems_cohort_privacy_guard.png` (Cohort dynamics, dropouts, and $k$-guard abort frontier).

---

## 6. Troubleshooting & Common Questions

- **Permission Denied on `tc netem`**: Ensure `cap_add: [NET_ADMIN]` is present in `docker-compose.yml`.
- **Port Conflicts (8080 or 9090 already in use)**: Override the ports in `configs/config.yaml` under `systems.server_port` and `systems.metrics_port`, or pass `--port` and `--metrics-port` CLI flags.
- **TLS Handshake Failures**: If running over IP addresses without DNS, set `check_hostname = False` in `client_node.py` or regenerate certificates with SANs using `scripts/generate_tls_certs.py`.

