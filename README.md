# YOLOv10-LiteRT on Android for Object Detection
This repository is an implementation of converting the YOLOv10 object detection model to LiteRT (.tflite) format using Google AI Edge and deploy it on Android for on-device inference.

<a target="_blank" href="https://colab.research.google.com/github/NSTiwari/YOLOv10-LiteRT-Android/blob/main/YOLOv10_LiteRT.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


## Pipeline:
<img src="https://github.com/NSTiwari/YOLOv10-LiteRT-Android/blob/main/assets/YOLOv10_LiteRT_Pipeline.png"/>


## Steps to run:

1. Clone the repository on your local machine.
2. Open the [Colab notebook](https://colab.research.google.com/github/NSTiwari/YOLOv10-LiteRT-Android/blob/main/YOLOv10_LiteRT.ipynb) to export YOLOv10-N to LiteRT (TF Lite) format.
3. Download the LiteRT model on your local machine and copy it in the [`assets`](https://github.com/NSTiwari/YOLOv10-LiteRT-Android/tree/main/Android_App/app/src/main/assets) folder.
4. Open the [`Android_App`](https://github.com/NSTiwari/YOLOv10-LiteRT-Android/tree/main/Android_App) project in Android Studio and let it build.
5. Install the app on your phone for real-time object detection.

