
from tensorflow import keras
import tensorflow_hub as hub




MODEL_PATH = "https://tfhub.dev/sayakpaul/convnext_tiny_1k_224_fe/1"
# "https://tfhub.dev/sayakpaul/convnext_large_1k_224_fe/1"


def get_model0(model_path=MODEL_PATH, res=224, num_classes=10):
    hub_layer = hub.KerasLayer(model_path, trainable=True)

    model = keras.Sequential(
        [
            keras.layers.InputLayer((res, res, 3)),
            hub_layer,
            keras.layers.Dense(num_classes, activation="sigmoid"),
        ]
    )
    return model
