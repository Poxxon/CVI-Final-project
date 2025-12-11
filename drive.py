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


controller = SimplePIController(0.1, 0.002)
SET_SPEED = 6.0
controller.set_desired(SET_SPEED)


@sio.on("telemetry")
def telemetry(sid, data):
    if not data:
        sio.emit("manual", data={}, skip_sid=True)
        return

    # speed from simulator
    speed = float(data["speed"])

    # image from simulator
    img_str = data["image"]
    image = Image.open(BytesIO(base64.b64decode(img_str)))
    image_array = np.asarray(image)

    # same preprocessing as training
    proc_img = preprocess_image(image_array)
    proc_img = np.expand_dims(proc_img, axis=0)

    steering_angle = float(model.predict(proc_img, batch_size=1))
    throttle = controller.update(speed)

    print(f"Steering: {steering_angle:.4f}, Throttle: {throttle:.3f}")

    send_control(steering_angle, throttle)

    # optional: record frames
    if args.image_folder != "":
        timestamp = datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
        img_name = os.path.join(args.image_folder, timestamp)
        image.save(f"{img_name}.jpg")


@sio.on("connect")
def connect(sid, environ):
    print("Connected:", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            "steering_angle": str(steering_angle),
            "throttle": str(throttle),
        },
        skip_sid=True,
    )


