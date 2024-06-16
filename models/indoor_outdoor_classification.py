import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def mobilenet_custom(weights):
        # Using models 
        base_model = tf.keras.applications.MobileNetV2(weights=None,input_shape=(112, 112, 3),include_top=False)
        base_model.trainable = False
        inputs = tf.keras.Input(shape=(112, 112, 3))
        x = base_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(32, activation = 'relu')(x)
        outputs = tf.keras.layers.Dense(1, activation = 'sigmoid' )(x)
        model = tf.keras.Model(inputs, outputs)
        model.load_weights(weights)
        return model

class indoor_outdoor_model():
    def __init__(self,w) -> None:
        self.model = mobilenet_custom(w)
        self.lg = ['Indoor', 'Outdoor']  # Categories of distribution 

    def find_environment(self,image):
        img = np.array(image[:, :, ::-1], dtype = 'float32')
        img = preprocess_input(img)
        img = img.reshape(1,112,112,3)
        y1 = self.model.predict(img)
        if y1[0][0]>0.5:
            return self.lg[1]
        else:
            return self.lg[0]

if __name__ == "__main__":
    print("hello")