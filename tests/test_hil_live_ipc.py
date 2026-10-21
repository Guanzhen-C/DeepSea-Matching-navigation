from __future__ import annotations

import socket
import threading

import pytest

from standalone.hil.live_ipc import LiveProducer, read_packet, send_packet
from standalone.hil.protocol import pack, unpack


def test_live_producer_preserves_events_and_waits_for_all_consumption_acks(tmp_path):
    endpoint = tmp_path / "hil.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(endpoint))
    server.listen(1)
    observed = []
    errors = []

    def bridge():
        try:
            connection, _ = server.accept()
            with connection:
                for i in range(20):
                    event = unpack(read_packet(connection))
                    observed.append(event)
                    send_packet(connection, pack(dict(sequence=i, status="consumed", result={"value": i})))
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=bridge)
    worker.start()
    producer = LiveProducer(endpoint)
    for i in range(20):
        producer.send(pack(dict(sequence=i, sensor_value=i / 10.)), i)
    producer.finish()
    worker.join(timeout=3.)
    server.close()
    assert not worker.is_alive() and not errors
    assert [v["sequence"] for v in observed] == list(range(20))
    assert producer.last_ack == producer.last_sent == 19
    assert producer.last_result == {"value": 19}


def test_ipc_rejects_unconsumed_disconnect(tmp_path):
    endpoint = tmp_path / "hil.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(endpoint))
    server.listen(1)

    def bridge():
        connection, _ = server.accept()
        read_packet(connection)
        connection.close()

    worker = threading.Thread(target=bridge)
    worker.start()
    producer = LiveProducer(endpoint)
    try:
        producer.send(pack(dict(sequence=0)), 0)
        with pytest.raises(EOFError):
            producer.finish()
    except EOFError:
        pass  # Immediate consumption failure is equally valid.
    finally:
        producer.connection.close()
        worker.join(timeout=3.)
        server.close()
