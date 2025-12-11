import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from data_preprocess import train_val_split, batch_generator

EPOCHS = 10
BATCH_SIZE = 64

print("Loading samples...")
train_samples, val_samples = train_val_split()
print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

train_generator = batch_generator(train_samples, batch_size=BATCH_SIZE, is_training=True)
val_generator = batch_generator(val_samples, batch_size=BATCH_SIZE, is_training=False)


def build_nvidia_model():
    """
    Classic NVIDIA end-to-end driving model.
    Input: 66x200x3 (already preprocessed)
    """
    model = Sequential()
    model.add(Lambda(lambda x: x, input_shape=(66, 200, 3)))

    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))

    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    return model


model = build_nvidia_model()
model.compile(loss="mse", optimizer="adam")

checkpoint = ModelCheckpoint(
    "model.h5", monitor="val_loss", verbose=1, save_best_only=True, mode="min"
)
early = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
)

steps_per_epoch = math.ceil(len(train_samples) / BATCH_SIZE)
val_steps = math.ceil(len(val_samples) / BATCH_SIZE)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=val_steps,
    epochs=EPOCHS,
    callbacks=[checkpoint, early],
    verbose=1,
)

model.save("model_final.h5")
print("Training complete. Saved model.h5 and model_final.h5.")
