"""CLI for the TVDI package.

    python src.py generate --n 50 --out data
    python src.py train --data data --model models/defect_classifier.joblib
    python src.py serve
    python src.py test
"""
import argparse
import sys


def _cmd_generate(args):
    from tvdi_industrial.synthetic import generate_dataset

    frame = generate_dataset(args.out, n_per_class=args.n, seed=args.seed)
    print(f"generated {len(frame)} samples in {args.out}")


def _cmd_train(args):
    from tvdi_industrial.training import train

    train(args.data, args.model)


def _cmd_serve(args):
    import uvicorn

    uvicorn.run("tvdi_industrial.api:app", host=args.host, port=args.port, reload=args.reload)


def _cmd_test(_args):
    import pytest

    sys.exit(pytest.main(["tests", "-q"]))


def _cmd_validate_real(_args):
    from tvdi_industrial.real_xray import run_trl5

    report = run_trl5()
    metrics = report["metrics"]
    print(
        f"TRL {report['trl']} real X-ray check: "
        f"accuracy {metrics['accuracy']:.3f}, macro F1 {metrics['macro_f1']:.3f}, "
        f"gate {'passed' if report['gate_passed'] else 'not passed'}"
    )
    print("report: data/real/trl5-report.json")


def _cmd_validate_locate(_args):
    from tvdi_industrial.localize import run_localization

    report = run_localization()
    test = report["test"]
    print(
        f"localization gate {'passed' if report['gate_passed'] else 'not passed'}: "
        f"recall {test['recall_iou_0_30']:.3f}, precision {test['precision_iou_0_30']:.3f}"
    )
    print("report: data/real/localization-report.json")


def _cmd_measure_cycle(_args):
    import json
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from tvdi_industrial.cycle import software_cycle
    from tvdi_industrial.defect_model import DefectModel
    from tvdi_industrial.features import extract_features

    model = DefectModel.load()

    def score(canvas):
        return model.predict(extract_features(np.asarray(canvas)))

    image = np.zeros((128, 128), dtype=np.uint8)
    sample = next(Path("data/real/xray/test/images").glob("*.jpg"), None)
    if sample is not None:
        image = np.asarray(Image.open(sample).convert("L"))
    report = software_cycle(image, score)
    report["sample"] = sample.name if sample is not None else "blank"
    path = Path("data/lab/cycle-report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"software cycle {report['software_cycle_ms']:.1f} ms over {report['sectors']} sectors")
    print("motion measured: false")
    print("report: data/lab/cycle-report.json")


def _cmd_validate_lab(args):
    from tvdi_industrial.lab_validation import run_trl4

    report = run_trl4(n_train=args.n_train, n_test=args.n_test, seed=args.seed)
    metrics = report["metrics"]
    print(
        f"TRL {report['trl']} laboratory validation: "
        f"accuracy {metrics['accuracy']:.3f}, macro F1 {metrics['macro_f1']:.3f}, "
        f"p95 {metrics['latency_ms']['p95']:.1f} ms"
    )
    print("report: data/lab/trl4-report.json")


def main():
    parser = argparse.ArgumentParser(description="TVDI toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate")
    generate.add_argument("--n", type=int, default=50)
    generate.add_argument("--out", type=str, default="data")
    generate.add_argument("--seed", type=int, default=42)
    generate.set_defaults(func=_cmd_generate)

    train = sub.add_parser("train")
    train.add_argument("--data", type=str, default="data")
    train.add_argument("--model", type=str, default="models/defect_classifier.joblib")
    train.set_defaults(func=_cmd_train)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", type=str, default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=_cmd_serve)

    test = sub.add_parser("test")
    test.set_defaults(func=_cmd_test)

    real = sub.add_parser("validate-real", help="سنجش ایکس‌ری واقعی برای بلوغ ۵")
    real.set_defaults(func=_cmd_validate_real)

    locate = sub.add_parser("validate-locate", help="مکان‌یابی عیب روی ایکس‌ری واقعی")
    locate.set_defaults(func=_cmd_validate_locate)

    cycle = sub.add_parser("measure-cycle", help="زمان چرخه نرم‌افزاری ۱۶ قطاع")
    cycle.set_defaults(func=_cmd_measure_cycle)

    validate = sub.add_parser("validate-lab", help="سنجش آزمایشگاهی TRL 4")
    validate.add_argument("--n-train", type=int, default=30)
    validate.add_argument("--n-test", type=int, default=15)
    validate.add_argument("--seed", type=int, default=42)
    validate.set_defaults(func=_cmd_validate_lab)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
