# sketch/views.py
import logging
import os
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import SketchForm
#from keras import load_model
#import tensorflow.keras.models as models
import tensorflow as tf
#from tensorflow.keras.models import load_model

#from keras import img_to_array, load_img
#import tensorflow.keras.preprocessing.image as image

#from tensorflow.keras.preprocessing.image import img_to_array, load_img

from PIL import Image
import cv2
import matplotlib.pyplot as plt
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
#g_model = tf.keras.models.load_model('path/to/your/g_model.h5')
#g_model = tf.keras.models.load_model('C:/Users/dell/Downloads/Face-Sketch-to-Image-Generation-using-GAN-maste1r/Face-Sketch-to-Image-Generation-using-GAN-master/Models/Pixel[02]_Context[08]/g_model.h5',custom_objects={'InstanceNormalization':InstanceNormalization})
# Load the model
#from tensorflow.keras.layers import InstanceNormalization

#tf.keras.layers.InstanceNormalization
#g_model = tf.keras.models.load_model('C:/Users/dell/Downloads/Face-Sketch-to-Image-Generation-using-GAN-master/Models/Pixel[02]_Context[08]/g_1model.h5',custom_objects={'InstanceNormalization':InstanceNormalization})
# Load the model

import tensorflow as tf
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import tensorflow as tf

class CustomInstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, **kwargs):
        super(CustomInstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale',
                                     shape=(input_shape[-1],),
                                     initializer=self.gamma_initializer,
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint,
                                     trainable=self.scale)
        self.offset = self.add_weight(name='offset',
                                      shape=(input_shape[-1],),
                                      initializer=self.beta_initializer,
                                      regularizer=self.beta_regularizer,
                                      constraint=self.beta_constraint,
                                      trainable=self.center)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        inv_stddev = tf.math.rsqrt(variance + self.epsilon)
        normalized = (inputs - mean) * inv_stddev
        return normalized * self.scale + self.offset

    def get_config(self):
        config = super(CustomInstanceNormalization, self).get_config()
        config.update({
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': tf.keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer': tf.keras.initializers.serialize(self.gamma_initializer),
            'beta_regularizer': tf.keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': tf.keras.regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': tf.keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint': tf.keras.constraints.serialize(self.gamma_constraint),
        })
        return config

# Register InstanceNormalization as a custom object
custom_objects = {'InstanceNormalization':InstanceNormalization}

# Load the model with custom objects
g_model = tf.keras.models.load_model('C:/Users/dell/Downloads/Face-Sketch-to-Image-Generation-using-GAN-master/Models/Pixel[02]_Context[08]/g_model.h5', custom_objects=custom_objects)

"""with tf.keras.utils.custom_object_scope({'InstanceNormalization': InstanceNormalization}):
    g_model = tf.keras.models.load_model('C:/Users/dell/Downloads/Face-Sketch-to-Image-Generation-using-GAN-master/Models/Pixel[02]_Context[08]/g_1model.h5')
    g_model = tf.keras.models.load_model(
    'C:/Users/dell/Downloads/Face-Sketch-to-Image-Generation-using-GAN-master/Models/Pixel[02]_Context[08]/g_1model.h5',
    custom_objects={'BatchNormalization': tf.keras.layers.BatchNormalization}
)"""
#g_model = load_model('Models/Pixel[1]_Context[0]/g_model.h5', custom_objects={'InstanceNormalization':InstanceNormalization})

def index(request):
    if request.method == 'POST':
        form = SketchForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Process the uploaded image
                sketch = request.FILES['sketch']
                fs = FileSystemStorage()
                file_path = fs.save(sketch.name, sketch)
                file_path = fs.url(file_path)

                # Load and resize the image
                img_path = '.' + file_path  # The uploaded file path
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))

                # Convert to numpy array
                img = tf.keras.preprocessing.image.img_to_array(img)
                norm_img = (img.copy() - 127.5) / 127.5

                # Predict using the loaded model
                g_img = g_model.predict(np.expand_dims(norm_img, 0))[0]
                g_img = g_img * 127.5 + 127.5

                # Convert prediction to an image
                g_img = g_img.astype(np.uint8)
                g_img = Image.fromarray(g_img)

                # Ensure the directory exists
                output_dir = 'sketch/static/sketch/'
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, 'predicted_image.png')

                # Save the predicted image
                g_img.save(output_path)

                # Serve the predicted image URL to the template
                predicted_image_url = '/static/sketch/predicted_image.png'

                return render(request, 'result.html', {'predicted_image_url': predicted_image_url})
            except Exception as e:
                logging.error(f"Error processing the image: {e}")
                return render(request, 'index.html', {'form': form, 'error': 'Error processing the image.'})
    else:
        form = SketchForm()
    return render(request, 'index.html', {'form': form})
