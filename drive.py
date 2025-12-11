import argparse
import base64
from datetime import datetime
import os
import shutil
from io import BytesIO

import numpy as np
from PIL import Image
from flask import Flask
import socketio
import eventlet
import eventlet.wsgi

from tensorflow.keras.models import load_model
from data_preprocess import preprocess_image

# Socket.IO server + Flask
sio = socketio.Server(logger=False, async_mode="eventlet")
flask_app = Flask(__name__)
app = socketio.Middleware(sio, flask_app)

model = None
args = None


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.0
        self.error = 0.0
        self.integral = 0.0

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        self.error = self.set_point - measurement
        self.integral += self.error
        return self.Kp * self.error + self.Ki * self.integral


