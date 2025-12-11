# Project Documentation

## Introduction

This project implements an end-to-end steering controller for the Udacity self-driving car simulator. Using data collected from five manual laps of Track 1 (center/left/right cameras with steering corrections), images are cropped, converted to YUV, blurred, resized, and normalized before training. A CNN based on NVIDIA’s architecture (five conv layers followed by dense layers with dropout) is trained with MSE/Adam and early stopping, producing `model.h5`/`model_final.h5`. At runtime, `drive.py` pre-processes incoming simulator frames and uses the trained network plus a PI throttle controller to keep the car near 6 mph and drive around Track 1 autonomously, addressing drift, steering spikes, and class imbalance through data augmentation and controller tuning.

## How to Run the Project
### Environment Setup
# 1. Navigate to project folder
cd ComputerVisionFall2025

# 2. Create virtual environment
python -m venv cvi-final-env

# 3. Activate the environment
# Windows:
cvi-final-env\Scripts\activate

# Mac/Linux:
source cvi-final-env/bin/activate

# 4. Install all required dependencies
pip install -r requirements.txt

## Dependencies

The project requires the following packages:
Python 3.8+
TensorFlow (CPU)
NumPy
OpenCV
Flask
Eventlet
python-socketio
Pillow
scikit-learn
matplotlib

## Launch Testing / Inference (Autonomous Driving)

Start the autonomous driving agent:

python drive.py model.h5


To save all frames during driving:

python drive.py model.h5 run1/


Then:

Open the Udacity Self-Driving Car Simulator

Select Autonomous Mode

The simulator will connect to your server

The car will drive using your trained CNN model
