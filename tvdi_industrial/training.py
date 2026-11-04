"""Train the fallback classifier from a generated dataset."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from tvdi_industrial.features import extract_features, features_to_vector


def train(data_dir="data", model_path="models/defect_classifier.joblib"):
    data_dir = Path(data_dir)
    frame = pd.read_csv(data_dir / "metadata.csv")
    vectors, labels = [], []
    for _, row in frame.iterrows():
        image = Image.open(data_dir / "images" / f"{row['tire_id']}.png").convert("L")
        vectors.append(features_to_vector(extract_features(image)))
        labels.append(row["defect_type"])

    x_train, x_test, y_train, y_test = train_test_split(
        np.array(vectors), np.array(labels), test_size=0.25, random_state=42, stratify=labels,
    )
    classifier = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    classifier.fit(x_train, y_train)
    print(classification_report(y_test, classifier.predict(x_test), zero_division=0))

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, model_path)
    print(f"model saved to {model_path}")
