"""
Unit tests for wire protocol serialization and tensor transfer.
"""

import socket
import threading

import torch

from src.systems.protocol import (
    MessageType,
    deserialize_tensor_dict,
    pack_message,
    recv_msg,
    send_msg,
    serialize_tensor_dict,
    unpack_message,
)


def test_pack_unpack_metadata():
    msg_type = MessageType.READY
    metadata = {"client_id": "client_vindr", "num_samples": 1200}
    packed = pack_message(msg_type, metadata)

    # Strip leading 4-byte total_len to simulate reading body
    unpacked_type, unpacked_meta, payload = unpack_message(packed[4:])
    assert unpacked_type == MessageType.READY
    assert unpacked_meta["client_id"] == "client_vindr"
    assert unpacked_meta["num_samples"] == 1200
    assert payload == b""


def test_tensor_serialization_roundtrip():
    orig_tensors = {
        "conv.weight": torch.randn(16, 3, 3, 3),
        "fc.bias": torch.tensor([0.1, 0.2, 0.3]),
    }
    raw_bytes = serialize_tensor_dict(orig_tensors)
    recovered = deserialize_tensor_dict(raw_bytes)

    assert set(orig_tensors.keys()) == set(recovered.keys())
    for k in orig_tensors:
        assert torch.allclose(orig_tensors[k], recovered[k])


def test_socket_send_recv():
    orig_tensors = {"layer.weight": torch.ones(5, 5) * 4.2}
    payload = serialize_tensor_dict(orig_tensors)
    metadata = {"round": 1, "client_id": "client_cmmd"}

    # Server socket
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(1)
    port = server_sock.getsockname()[1]

    received_data = {}

    def server_thread():
        conn, _ = server_sock.accept()
        msg_type, meta, rec_payload = recv_msg(conn)
        received_data["msg_type"] = msg_type
        received_data["meta"] = meta
        received_data["tensors"] = deserialize_tensor_dict(rec_payload)
        conn.close()
        server_sock.close()

    th = threading.Thread(target=server_thread)
    th.start()

    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect(("127.0.0.1", port))
    send_msg(client_sock, MessageType.UPLOAD, metadata, payload)
    client_sock.close()

    th.join(timeout=3.0)

    assert received_data["msg_type"] == MessageType.UPLOAD
    assert received_data["meta"]["client_id"] == "client_cmmd"
    assert torch.allclose(received_data["tensors"]["layer.weight"], orig_tensors["layer.weight"])

