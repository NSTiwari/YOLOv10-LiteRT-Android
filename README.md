# YOLOv10-LiteRT on Android for Object Detection
Real-time on-device object detection on Android using YOLOv10 converted to LiteRT (.tflite) via Google AI Edge.

## Pipeline
<img src="https://github.com/NSTiwari/YOLOv10-LiteRT-Android/blob/main/assets/YOLOv10_LiteRT_Pipeline.png"/>

## How it works

The conversion goes through a few steps that aren't obvious from the outside. Ultralytics doesn't export directly to TFLite — it first converts the PyTorch model to ONNX, then uses `onnx2tf` to produce a TensorFlow SavedModel, and finally converts that to `.tflite`. The `yolov10_litert.py` script handles the full chain with a single `--export` flag.

Once on Android, inference runs through the LiteRT runtime (formerly TFLite) using the `Detector` class, which:
- Preprocesses camera frames to 640×640 float32 tensors
- Reads the model output tensor of shape `[1, 300, 6]` — 300 candidate detections, each with `(x_min, y_min, x_max, y_max, confidence, class_id)` in normalized coordinates
- Scales the box coordinates back to screen dimensions
- Filters by confidence threshold and passes the results to `OverlayView`

`OverlayView` draws bounding boxes and labels directly on the camera preview using Android's `Canvas` API, giving you a live overlay with no perceptible lag on modern devices.

The labelmap is extracted from the `metadata.yaml` that Ultralytics writes alongside the exported model and saved as a `labels.json` file that the Android app loads at startup.

Once the model is exported, you can also visualize its full neural network architecture interactively using Google AI Edge's Model Explorer:

```python
import model_explorer
model_explorer.visualize("yolov10n_saved_model/yolov10n_float32.tflite")
```

This opens a browser-based graph view where you can inspect every layer, see tensor shapes flowing through the network, and understand how YOLOv10's head produces the `[1, 300, 6]` detection output. Useful for debugging quantization issues or just understanding what's inside the model.

## Demo Output
<img src="https://github.com/NSTiwari/YOLOv10-LiteRT-Android/blob/main/assets/YOLOv10_LiteRT_Android.gif"/>

## How to run

### Python — export the model

```bash
pip install -r requirements.txt

# Export YOLOv10n to TFLite (downloads weights automatically)
python yolov10_litert.py --export --model yolov10n.pt

# Test inference on an image
python yolov10_litert.py --input test_image.jpg --tflite yolov10n_saved_model/yolov10n_float32.tflite

# Test inference on a video
python yolov10_litert.py --input test_video.mp4 --tflite yolov10n_saved_model/yolov10n_float32.tflite --confidence 0.4
```

### Android — deploy the model

1. Copy the exported `.tflite` file and `labels.json` into `Android_App/app/src/main/assets/`
2. Open `Android_App` in Android Studio
3. Build and run on a physical device (API 24+)
4. Grant camera permission on first launch

The app runs everything on-device — no network calls, no server.

## Resources

1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NSTiwari/YOLOv10-LiteRT-Android/blob/main/YOLOv10_LiteRT.ipynb)
2. [**Medium Blog**](https://tiwarinitin1999.medium.com/yolov10-to-litert-object-detection-on-android-with-google-ai-edge-2d0de5619e71) — step-by-step walkthrough
3. [YOLOv10 docs](https://docs.ultralytics.com/models/yolov10/) by Ultralytics
4. [Google AI Edge LiteRT](https://ai.google.dev/edge/litert)

## Acknowledgment
<img src="https://github.com/NSTiwari/YOLOv10-LiteRT-Android/blob/main/assets/google.png">

This project was developed during Google's ML Developer Programs AI Sprint. Thanks to the MLDP team for providing Google Cloud credits to support this project.
