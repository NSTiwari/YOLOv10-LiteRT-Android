# YOLOv10-LiteRT on Android for Object Detection
This repository is an implementation of converting the YOLOv10 object detection model to LiteRT (.tflite) format and deploy it on Android using Google AI Edge for on-device inference.


## Pipeline
<img src="https://github.com/NSTiwari/YOLOv10-LiteRT-Android/blob/main/assets/YOLOv10_LiteRT_Pipeline.png"/>


## Demo Output
<img src="https://github.com/NSTiwari/YOLOv10-LiteRT-Android/blob/main/assets/YOLOv10_LiteRT_Android.gif"/>


## Steps to run:

1. Clone the repository on your local machine.
2. Open the [Colab notebook](https://colab.research.google.com/github/NSTiwari/YOLOv10-LiteRT-Android/blob/main/YOLOv10_LiteRT.ipynb) to export YOLOv10-N to LiteRT (TF Lite) format.
3. Download the LiteRT model on your local machine and copy it in the [`assets`](https://github.com/NSTiwari/YOLOv10-LiteRT-Android/tree/main/Android_App/app/src/main/assets) folder.
4. Open the [`Android_App`](https://github.com/NSTiwari/YOLOv10-LiteRT-Android/tree/main/Android_App) project in Android Studio and let it build.
5. Install the app on your phone and enjoy real-time object detection on Android.


## Resources

1. Colab Notebook: 
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NSTiwari/YOLOv10-LiteRT-Android/blob/main/YOLOv10_LiteRT.ipynb)
2. [**Medium Blog**](https://tiwarinitin1999.medium.com/yolov10-to-litert-object-detection-on-android-with-google-ai-edge-2d0de5619e71) for step-by-step implementation
3. Official [documentation](https://docs.ultralytics.com/models/yolov10/) of YOLOv10 by Ultralytics
4. [Google AI Edge LiteRT](https://ai.google.dev/edge/litert)

  
## Acknowledgment
<img src="https://github.com/NSTiwari/YOLOv10-LiteRT-Android/blob/main/assets/google.png">

This project was developed during Google's ML Developer Programs AI Sprint. Thanks to the MLDP team for providing Google Cloud credits to support this project.
