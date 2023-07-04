import tensorflow as tf

path = "Abstract_gs_model/Abstract_tribrid_model/"
 
def load_model(model_path):
  model=tf.keras.models.load_model(model_path)
  return model

def get_model():
  return load_model(path)
