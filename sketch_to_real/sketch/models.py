from django.db import models

# Create your models here.
# sketch/model.py
import tensorflow as tf

def load_model():
    model = tf.keras.models.load_model('C:/Users/dell/Downloads/Face-Sketch-to-Image-Generation-using-GAN-master/Models/Pixel[02]_Context[08]/g_model.h5')
    return model

model = load_model()

