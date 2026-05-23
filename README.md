# Hand Gesture Recognition

This project runs a real-time hand gesture recognizer using MediaPipe for hand tracking and a trained TensorFlow model for classification.

## Setup

Install the Python dependencies with:

```powershell
pip install -r requirements.txt
```

The webcam script also expects a MediaPipe Hand Landmarker model file named `hand_landmarker.task` in the same folder as `realTimeTest.py`.

Download the model from MediaPipe and place it next to the script before running:

https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

This is a trained model artifact, so it cannot be created from your code or generated locally.

## Run

```powershell
python realTimeTest.py
```

## Test the trained model on sample dataset images

Use the script below to sample a few images from the local `09` folder, run the trained model, print the predictions, and show the images with the predicted labels:

```powershell
python test_model_samples.py --image-dir 09 --sample-count 4
```

If you want to save the plotted results, add:

```powershell
python test_model_samples.py --image-dir 09 --sample-count 4 --save-figure results.png
```

## Colab Notebook

Use the training notebook here:

https://colab.research.google.com/drive/1ZQ5SJYFUOtcCbCpjudcBxVt06fx5g2-x
