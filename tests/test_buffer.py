from tvdi_industrial.edge.buffer import EdgeBuffer


def test_buffer_enqueue_and_sync(tmp_path):
    buf = EdgeBuffer(str(tmp_path / "buf.sqlite"))
    buf.enqueue("t/1", {"a": 1})
    buf.enqueue("t/2", {"a": 2})
    pending = buf.pending()
    assert len(pending) == 2

    sent = []
    buf.sync(lambda topic, payload: sent.append(topic) or True)
    assert sent == ["t/1", "t/2"]
    assert buf.pending() == []
