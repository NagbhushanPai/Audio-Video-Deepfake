# Lip-Reading Deepfake Detection Model

This project implements a deepfake detection model based on lip-reading techniques. It uses TensorFlow and a combination of convolutional neural networks (CNNs) and bidirectional long short-term memory networks (BiLSTMs) to process video frames and detect potential deepfakes. The model compares predicted text from video frames with audio transcription to identify discrepancies.

## Features

- **Lip-Reading Model**: Processes video frames to predict text from lip movements.
- **Deepfake Detection**: Compares lip-read text with audio transcription for validation.
- **Custom Metrics**: Computes Word Error Rate (WER), Character Error Rate (CER), and similarity scores to evaluate predictions.
- **Visualization**: Generates animations of processed video frames and displays intermediate results.
- **Audio Extraction**: Extracts audio from video for transcription using Google Speech-to-Text API.
- **Model Training and Testing**: Includes functionalities for training, validation, and testing with TensorFlow datasets.

---

## Prerequisites

- Python 3.8 or later
- GPU-enabled system with CUDA (optional for faster training)
- Required Python libraries:
  - OpenCV
  - TensorFlow
  - NumPy
  - Matplotlib
  - ImageIO
  - GDown
  - MoviePy
  - Pydub
  - SpeechRecognition
  - Jiwer

Install the dependencies using:

```bash
pip install opencv-python==4.6.0.66
pip install tensorflow==2.10.1
pip install matplotlib==3.6.2
pip install imageio==2.23.0
pip install gdown==4.6.0
pip install SpeechRecognition pydub moviepy jiwer



Project Workflow
1. Setup
Configure TensorFlow to use GPU.
Download and extract video data using gdown.
2. Data Preprocessing
Extract video frames and alignments.
Normalize frame data for input into the neural network.
3. Model
CNNs extract spatial features from video frames.
Bidirectional LSTMs process temporal dependencies.
CTC (Connectionist Temporal Classification) Loss is used for training the model.
4. Training
Dataset split into training and testing sets.
Model training with callbacks for checkpoints, learning rate scheduling, and prediction examples.
5. Deepfake Detection
Compare predicted text from video with transcribed audio text.
Normalize text and calculate similarity using metrics like SequenceMatcher.
6. Evaluation
Use CER and WER to evaluate the model's accuracy.
Usage
Run the notebook: Open Final_SIH.ipynb in Google Colab or Jupyter Notebook and execute cells sequentially.
Deepfake Detection: Provide video file paths and analyze using the pipeline.
Evaluate Model: Check CER and WER to validate the model.
Output
Predictions: Predicted text from the model for video inputs.
Speech Transcription: Text transcription from extracted audio.
Similarity Metrics: CER, WER, and overall similarity percentage.
Verdict: Determines whether the video is real or a deepfake.
Saved Models
Trained models are saved in the models directory and can be loaded for predictions.

Contributions
Feel free to contribute to this project by improving preprocessing, model architecture, or extending functionality.

References
TensorFlow documentation
Jiwer for WER/CER computation
OpenCV and MoviePy for video processing

Acknowledgments
This project was developed as part of the [Smart India Hackathon (SIH)] challenge. Special thanks to Nicholas Renotte for downloading Checkpoints.

License
This project is licensed under the MIT License. See LICENSE for more details.