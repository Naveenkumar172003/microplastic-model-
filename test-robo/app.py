import json
import os
import uuid
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from inference_sdk import InferenceHTTPClient
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.secret_key = "replace-this-with-a-random-secret"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="sMx93axJWDsX234ZDtuJ",
)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _collect_labels(obj):
    labels = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"class", "label", "name"} and isinstance(value, str):
                labels.append(value)
            labels.extend(_collect_labels(value))
    elif isinstance(obj, list):
        for item in obj:
            labels.extend(_collect_labels(item))
    return labels


def _collect_predictions(obj):
    predictions = []
    if isinstance(obj, dict):
        if any(k in obj for k in ["class", "label", "name"]):
            predictions.append(obj)
        for value in obj.values():
            predictions.extend(_collect_predictions(value))
    elif isinstance(obj, list):
        for item in obj:
            predictions.extend(_collect_predictions(item))
    return predictions


def summarize_microplastic_detection(result):
    labels = [label.lower() for label in _collect_labels(result)]
    has_microplastic = any("micro" in label and "plastic" in label for label in labels)

    predictions = _collect_predictions(result)
    microplastic_predictions = []
    for pred in predictions:
        label = pred.get("class") or pred.get("label") or pred.get("name")
        if isinstance(label, str) and ("micro" in label.lower() and "plastic" in label.lower()):
            microplastic_predictions.append(pred)

    return {
        "has_microplastic": has_microplastic,
        "microplastic_count": len(microplastic_predictions),
        "microplastic_predictions": microplastic_predictions,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("Please upload an image file.")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type. Upload JPG, PNG, BMP, or WEBP.")
        return redirect(url_for("index"))

    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    image_path = UPLOAD_DIR / unique_name
    file.save(image_path)

    try:
        result = client.run_workflow(
            workspace_name="face-n35x8",
            workflow_id="detect-count-and-visualize",
            images={"image": str(image_path)},
            use_cache=True,
        )
    except Exception as exc:
        flash(f"Inference failed: {exc}")
        return redirect(url_for("index"))

    summary = summarize_microplastic_detection(result)

    return render_template(
        "results.html",
        uploaded_image_url=url_for("uploaded_file", filename=unique_name),
        summary=summary,
        raw_result=json.dumps(result, indent=2),
    )


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True)
