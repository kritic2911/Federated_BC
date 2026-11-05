#!/bin/bash
set -e

# ==============================================================================
# entrypoint_client.sh: Hospital Client Container Startup & Traffic Shaping
# ==============================================================================
# Dynamically reads network latency, jitter, and packet loss from configs/config.yaml
# and applies Linux traffic control (tc netem) on eth0 prior to starting the client.
# ==============================================================================

CONFIG_FILE="${CONFIG_FILE:-configs/config.yaml}"
SITE_NAME="${SITE_NAME:-vindr}"
INTERFACE="${NETEM_INTERFACE:-eth0}"

echo "=========================================================="
echo " Starting FL Hospital Client: ${SITE_NAME}"
echo " Config Source: ${CONFIG_FILE}"
echo " Network Interface: ${INTERFACE}"
echo "=========================================================="

# Extract netem parameters dynamically from config.yaml using Python
NETEM_PARAMS=$(python3 -c "
import sys
from pathlib import Path
try:
    import yaml
    cfg_path = Path('${CONFIG_FILE}')
    if not cfg_path.exists():
        # Fallback to absolute /app/configs/config.yaml inside container
        cfg_path = Path('/app/configs/config.yaml')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}

    sys_cfg = cfg.get('systems', {})
    site_profiles = sys_cfg.get('site_profiles', {})
    netem_profiles = sys_cfg.get('netem_profiles', {})

    # Determine profile: explicit env var takes priority over site default
    profile_name = '${LATENCY_PROFILE}'
    if not profile_name:
        profile_name = site_profiles.get('${SITE_NAME}', 'metro_hospital')

    profile = netem_profiles.get(profile_name, {})
    delay = float(profile.get('latency_ms', 0.0))
    jitter = float(profile.get('jitter_ms', 0.0))
    loss = float(profile.get('loss_pct', 0.0))

    print(f'{profile_name} {delay}ms {jitter}ms {loss}')
except Exception as e:
    # Fail safe: no delay
    print(f'none 0ms 0ms 0')
")

PROFILE_NAME=$(echo "${NETEM_PARAMS}" | awk '{print $1}')
DELAY=$(echo "${NETEM_PARAMS}" | awk '{print $2}')
JITTER=$(echo "${NETEM_PARAMS}" | awk '{print $3}')
LOSS=$(echo "${NETEM_PARAMS}" | awk '{print $4}')

echo "[tc-netem] Resolved profile: '${PROFILE_NAME}' (Delay: ${DELAY}, Jitter: ${JITTER}, Loss: ${LOSS}%)"

# Apply traffic shaping if delay or loss is configured
if [ "${DELAY}" != "0.0ms" ] && [ "${DELAY}" != "0ms" ]; then
    echo "[tc-netem] Applying rule: delay ${DELAY} ${JITTER} loss ${LOSS}% on ${INTERFACE}..."
    # Clear any previous qdisc rule first
    tc qdisc del dev "${INTERFACE}" root 2>/dev/null || true
    if tc qdisc add dev "${INTERFACE}" root netem delay "${DELAY}" "${JITTER}" loss "${LOSS}%" 2>/dev/null; then
        echo "[tc-netem] Successfully configured network profile '${PROFILE_NAME}'."
    else
        echo "[tc-netem] WARNING: Unable to apply tc netem (requires CAP_NET_ADMIN). Running on native bridge network."
    fi
else
    echo "[tc-netem] Latency is 0ms. Running with native bridge latency."
fi

# Execute the main container process
exec "$@"
