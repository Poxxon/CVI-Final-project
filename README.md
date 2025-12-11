# Project Documentation

## Introduction

This project implements an end-to-end steering controller for the Udacity self-driving car simulator. Using data collected from five manual laps of Track 1 (center/left/right cameras with steering corrections), images are cropped, converted to YUV, blurred, resized, and normalized before training. A CNN based on NVIDIA’s architecture (five conv layers followed by dense layers with dropout) is trained with MSE/Adam and early stopping, producing `model.h5`/`model_final.h5`. At runtime, `drive.py` pre-processes incoming simulator frames and uses the trained network plus a PI throttle controller to keep the car near 6 mph and drive around Track 1 autonomously, addressing drift, steering spikes, and class imbalance through data augmentation and controller tuning.

