"""
Network protocol and wire message serialization for Phase P4 distributed FL.
Handles length-prefixed binary/JSON wire framing over TLS or plain TCP sockets.
"""

import io
import json
import struct
from enum import StrEnum
from typing import Any, Dict, Tuple

import torch


class MessageType(StrEnum):
    ANNOUNCE = "ANNOUNCE"          # Server -> Client: Start of round, sends round ID
    READY = "READY"                # Client -> Server: Confirms readiness to participate
    START_UPLOAD = "START_UPLOAD"  # Server -> Client: Surviving cohort U confirmed; ready for upload
    UPLOAD = "UPLOAD"              # Client -> Server: Sends masked local update & metadata
    ABORT = "ABORT"                # Server -> Client: Round aborted (e.g. cohort < k or timeout)
    COMMIT = "COMMIT"              # Server -> Client: Round successful; aggregate committed
    HEARTBEAT = "HEARTBEAT"        # Ping/pong telemetry


def serialize_tensor_dict(tensors: Dict[str, torch.Tensor]) -> bytes:
    """Serializes a dictionary of torch Tensors (e.g. state_dict) to bytes."""
    buf = io.BytesIO()
    # Save with CPU mapping to ensure portability across different host/container devices
    cpu_tensors = {k: v.detach().cpu() for k, v in tensors.items()}
    torch.save(cpu_tensors, buf)
    return buf.getvalue()


def deserialize_tensor_dict(payload_bytes: bytes) -> Dict[str, torch.Tensor]:
    """Deserializes bytes back to a dictionary of torch Tensors."""
    buf = io.BytesIO(payload_bytes)
    return torch.load(buf, map_location="cpu", weights_only=True)


def pack_message(
    msg_type: MessageType,
    metadata: Dict[str, Any],
    payload_bytes: bytes | None = None,
) -> bytes:
    """
    Packs a protocol message with a 4-byte length prefix framing:
    [Total Length: 4B uint32]
    [JSON Metadata Length: 4B uint32]
    [JSON Metadata UTF-8 Bytes]
    [Binary Payload Bytes (optional)]
    """
    meta_with_type = {"msg_type": msg_type.value, **metadata}
    meta_json = json.dumps(meta_with_type).encode("utf-8")
    meta_len = len(meta_json)

    payload = payload_bytes if payload_bytes is not None else b""
    payload_len = len(payload)

    total_len = 4 + meta_len + payload_len

    # Pack: uint32 total_len, uint32 meta_len, meta_bytes, payload_bytes
    header = struct.pack(">II", total_len, meta_len)
    return header + meta_json + payload


def unpack_message(raw_bytes: bytes) -> Tuple[MessageType, Dict[str, Any], bytes]:
    """
    Unpacks a raw framed message buffer into (msg_type, metadata, payload_bytes).
    Note: raw_bytes should not include the outer 4-byte total_len if already stripped by socket reader.
    """
    if len(raw_bytes) < 4:
        raise ValueError("Buffer too short to contain message header.")

    meta_len = struct.unpack(">I", raw_bytes[:4])[0]
    meta_bytes = raw_bytes[4 : 4 + meta_len]
    payload_bytes = raw_bytes[4 + meta_len :]

    meta_dict = json.loads(meta_bytes.decode("utf-8"))
    msg_type = MessageType(meta_dict.pop("msg_type"))
    return msg_type, meta_dict, payload_bytes


def send_msg(
    sock: Any,
    msg_type: MessageType,
    metadata: Dict[str, Any],
    payload_bytes: bytes | None = None,
) -> int:
    """Sends a framed message over a socket. Returns total bytes sent."""
    packet = pack_message(msg_type, metadata, payload_bytes)
    sock.sendall(packet)
    return len(packet)


def recv_msg(sock: Any) -> Tuple[MessageType, Dict[str, Any], bytes]:
    """Receives a framed message from a socket, reading exactly total_len bytes."""
    header = _recv_exact(sock, 4)
    if not header:
        raise ConnectionResetError("Socket closed while waiting for message header.")
    total_len = struct.unpack(">I", header)[0]

    # Read the rest of the message (total_len bytes)
    body = _recv_exact(sock, total_len)
    if len(body) != total_len:
        raise ConnectionResetError(f"Premature EOF: expected {total_len} bytes, got {len(body)}.")

    return unpack_message(body)


def _recv_exact(sock: Any, n: int) -> bytes:
    """Helper to read exactly n bytes from a streaming socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)
