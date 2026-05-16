#!/usr/bin/env python3
# Converts YOLOv10 to LiteRT (.tflite) format and runs inference on images or videos.
# Developed by Nitin Tiwari (github.com/NSTiwari)
#
# Usage:
#   # Export YOLOv10n to TFLite first:
#   python yolov10_litert.py --export --model yolov10n.pt
#
#   # Run inference on an image:
#   python yolov10_litert.py --input test_image.jpg --tflite yolov10n_saved_model/yolov10n_float32.tflite
#
#   # Run inference on a video:
#   python yolov10_litert.py --input test_video.mp4 --tflite yolov10n_saved_model/yolov10n_float32.tflite
#
# Install dependencies:
#   pip install -r requirements.txt

import os
import json
import random
import argparse

import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ai_edge_litert.interpreter import Interpreter


def export_to_litert(model_path):
    """Loads a YOLOv10 .pt checkpoint and exports it to TFLite via ONNX → SavedModel → TFLite."""
    print(f"Loading {model_path} ...")
    model = YOLO(model_path)
    export_path = model.export(format="tflite")
    print(f"Exported to: {export_path}")
    return export_path


def create_labelmap(export_folder, output_file="labels.json"):
    """Reads the metadata.yaml that ultralytics writes alongside the model and saves it as JSON."""
    metadata_path = os.path.join(export_folder, "metadata.yaml")
    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)

    names = metadata.get("names", {})
    with open(output_file, "w") as f:
        json.dump(names, f, indent=2)

    print(f"Labelmap saved to: {output_file}")
    return names


def load_labels(label_file):
    with open(label_file, "r") as f:
        return json.load(f)


def generate_color_map(labels):
    """Assigns a random BGR color to each label so boxes are visually distinct."""
    return {label: [random.randint(0, 255) for _ in range(3)] for label in labels.values()}


def load_and_preprocess(image_path, input_size):
    image = cv2.imread(image_path)
    original_h, original_w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_size, input_size))
    image = image / 255.0
    return image, (original_h, original_w)


def run_inference(interpreter, input_details, output_details, frame_or_path, is_video_frame=False):
    """Feeds one image/frame through the LiteRT interpreter and returns raw output tensors."""
    input_size = input_details[0]["shape"][1]

    if is_video_frame:
        original_h, original_w = frame_or_path.shape[:2]
        image = cv2.cvtColor(frame_or_path, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (input_size, input_size))
        image = image / 255.0
    else:
        image, (original_h, original_w) = load_and_preprocess(frame_or_path, input_size)

    interpreter.set_tensor(
        input_details[0]["index"],
        np.expand_dims(image, axis=0).astype(np.float32)
    )
    interpreter.invoke()

    output_data = [interpreter.get_tensor(d["index"]) for d in output_details]
    return output_data, (original_h, original_w)


def postprocess(output_data, original_dims, labels, confidence_threshold):
    """
    Converts raw model output to a list of detections.
    YOLOv10 outputs boxes in normalized [0,1] coordinates — we scale them back here.
    Output tensor shape is [1, 300, 6]: (x_min, y_min, x_max, y_max, confidence, class_id).
    """
    output_tensor = output_data[0]
    original_h, original_w = original_dims
    detections = []

    for i in range(output_tensor.shape[1]):
        box = output_tensor[0, i, :4]
        confidence = output_tensor[0, i, 4]
        class_id = int(output_tensor[0, i, 5])

        if confidence < confidence_threshold:
            continue

        x_min = int(box[0] * original_w)
        y_min = int(box[1] * original_h)
        x_max = int(box[2] * original_w)
        y_max = int(box[3] * original_h)

        detections.append({
            "box": [y_min, x_min, y_max, x_max],
            "score": float(confidence),
            "class": class_id,
            "label": labels.get(str(class_id), "Unknown"),
        })

    return detections


def draw_boxes(image_rgb, detections, color_map):
    """Draws bounding boxes and label badges onto an RGB image array."""
    for det in detections:
        y_min, x_min, y_max, x_max = det["box"]
        label = det["label"]
        score = det["score"]
        color = color_map.get(label, [0, 255, 0])

        cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), color, 3)

        text = f"{label}: {score:.2f}"
        font_scale = 1.0
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]

        # Fill a solid rectangle behind the label so it's readable on any background
        cv2.rectangle(
            image_rgb,
            (x_min, y_min - text_size[1] - 10),
            (x_min + text_size[0], y_min),
            color, -1
        )
        cv2.putText(
            image_rgb, text, (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2
        )

    return image_rgb


def infer_image(image_path, interpreter, input_details, output_details, labels, color_map, confidence_threshold):
    output_data, original_dims = run_inference(interpreter, input_details, output_details, image_path)
    detections = postprocess(output_data, original_dims, labels, confidence_threshold)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = draw_boxes(image_rgb, detections, color_map)

    plt.figure(figsize=(12, 8))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()

    output_path = "output_" + os.path.basename(image_path)
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved to: {output_path}")
    return output_path


def infer_video(video_path, interpreter, input_details, output_details, labels, color_map, confidence_threshold):
    cap = cv2.VideoCapture(video_path)
    output_path = "output_" + os.path.splitext(os.path.basename(video_path))[0] + ".avi"

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        output_path, fourcc, 20.0,
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_data, original_dims = run_inference(
            interpreter, input_details, output_details, frame, is_video_frame=True
        )
        detections = postprocess(output_data, original_dims, labels, confidence_threshold)

        # Draw on a copy so the original frame stays clean for the writer
        annotated = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated = draw_boxes(annotated, detections, color_map)
        out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  Processed {frame_count} frames ...")

    cap.release()
    out.release()
    print(f"Saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="YOLOv10 LiteRT inference")
    parser.add_argument("--export", action="store_true", help="Export YOLOv10 .pt to TFLite")
    parser.add_argument("--model", default="yolov10n.pt", help="Path to YOLOv10 .pt weights (for export)")
    parser.add_argument("--tflite", default=None, help="Path to the .tflite model file")
    parser.add_argument("--input", default=None, help="Path to input image or video")
    parser.add_argument("--labels", default="labels.json", help="Path to labels JSON file")
    parser.add_argument("--confidence", type=float, default=0.4, help="Confidence threshold (default: 0.4)")
    args = parser.parse_args()

    if args.export:
        export_path = export_to_litert(args.model)
        export_folder = os.path.dirname(export_path)
        create_labelmap(export_folder)
        print("\nExport done. Use --tflite and --input to run inference.")
        return

    if not args.tflite:
        parser.error("--tflite is required for inference")
    if not args.input:
        parser.error("--input is required for inference")

    print(f"Loading model: {args.tflite}")
    interpreter = Interpreter(model_path=args.tflite)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Input size: {input_details[0]['shape'][1]}x{input_details[0]['shape'][2]}")
    print(f"Output shape: {output_details[0]['shape']}")

    labels = load_labels(args.labels)
    color_map = generate_color_map(labels)

    # Decide image vs video based on file extension
    ext = os.path.splitext(args.input)[1].lower()
    if ext in (".mp4", ".avi", ".mov", ".mkv"):
        infer_video(args.input, interpreter, input_details, output_details, labels, color_map, args.confidence)
    else:
        infer_image(args.input, interpreter, input_details, output_details, labels, color_map, args.confidence)


if __name__ == "__main__":
    main()
