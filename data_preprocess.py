import os
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "driving_log.csv")
IMG_DIR = os.path.join(DATA_DIR, "IMG")


def preprocess_image(img):
    """
    Preprocessing used both for training and driving:

    - Crop sky/hood -> keep road area
    - RGB -> YUV (better for lane features)
    - Gaussian blur (reduce noise)
    - Resize to 200x66
    - Normalize to [0,1]
    """
    img = img[60:135, :, :]                 # crop
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img


def load_samples():
    """
    Read driving_log.csv and create samples for
    center / left / right cameras with small steering offsets.
    """
    samples = []

    with open(CSV_PATH) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            center_path = line[0]
            left_path = line[1]
            right_path = line[2]
            steering = float(line[3])

            # basic camera correction
            correction = 0.2
            samples.append((center_path.strip(), steering))
            samples.append((left_path.strip(), steering + correction))
            samples.append((right_path.strip(), steering - correction))

    return samples


def train_val_split(test_size=0.2):
    samples = load_samples()
    train_samples, val_samples = train_test_split(
        samples, test_size=test_size, random_state=42
    )
    return train_samples, val_samples


def batch_generator(samples, batch_size=32, is_training=True):
    num_samples = len(samples)

    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []

            for img_path, steering in batch_samples:
                filename = os.path.basename(img_path)
                full_path = os.path.join(IMG_DIR, filename)

                if not os.path.exists(full_path):
                    # skip missing files
                    continue

                img = cv2.imread(full_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = preprocess_image(img)
                angle = steering

                if is_training:
                    # random horizontal flip
                    if np.random.rand() < 0.5:
                        img = np.fliplr(img)
                        angle = -angle

                images.append(img)
                angles.append(angle)

            if images:
                yield np.array(images), np.array(angles)


if __name__ == "__main__":
    train_samples, val_samples = train_val_split()
    total = len(train_samples) + len(val_samples)
    print(f"Total samples: {total}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
