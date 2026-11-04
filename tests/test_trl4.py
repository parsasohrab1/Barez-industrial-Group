from tvdi_industrial.lab_validation import run_trl4


def test_laboratory_breadboard_reaches_trl4(tmp_path):
    report = run_trl4(
        out_dir=tmp_path / "lab",
        n_train=8,
        n_test=6,
        seed=42,
        publish=False,
    )
    assert report["trl"] == 4
    assert report["environment"] == "laboratory"
    assert report["metrics"]["macro_f1"] >= 0.8
    assert report["metrics"]["latency_ms"]["p95"] < 2000
    assert report["grading"]["critical_defects"] == 6
    assert report["grading"]["critical_defects_graded_c"] == report["grading"]["critical_defects"]
    assert report["traceability"]["causes"]
    assert report["integration"]["camera"] is True
    assert report["integration"]["offline_buffer"] is True
    assert report["integration"]["vision"] is True
    assert (tmp_path / "lab" / "trl4-report.json").exists()
