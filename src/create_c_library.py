import tensorflow as tf
import sagemaker
import os
# Function: Convert some hex value into an array for C programming
def hex_to_c_array(hex_data, var_name):

  c_str = ''

  # Create header guard
  c_str += '#ifndef ' + var_name.upper() + '_H\n'
  c_str += '#define ' + var_name.upper() + '_H\n\n'

  # Add array length at top of file
  c_str += '\nunsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'

  # Declare C variable
  c_str += 'unsigned char ' + var_name + '[] = {'
  hex_array = []
  for i, val in enumerate(hex_data) :

    # Construct string from hex
    hex_str = format(val, '#04x')

    # Add formatting so each line stays within 80 characters
    if (i + 1) < len(hex_data):
      hex_str += ','
    if (i + 1) % 12 == 0:
      hex_str += '\n '
    hex_array.append(hex_str)

  # Add closing brace
  c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'

  # Close out header guard
  c_str += '#endif //' + var_name.upper() + '_H'

  return c_str
  


def convert2lite(model_path, summary=False):
  
  if summary:
      model = tf.keras.models.load_model(model_path)
      print(model.summary())
  
  converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
  tflite_model = converter.convert()
  
  return tflite_model

def createCLibrary(model_path, tflite_model_name="tflite_model", c_model_name="tflite_model_library"):

  tflite_model = convert2lite(model_path) 
  pwd = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
  tflite_model_path = pwd + "/models/tflite_models/"+ tflite_model_name
  c_model_path = pwd + "/models/c_models/" + c_model_name
  open(tflite_model_path + '.tflite', 'wb').write(tflite_model)

  with open(c_model_path + '.h', 'w') as file:
      file.write(hex_to_c_array(tflite_model, c_model_name))

################## TEST FUNCTION ######################
"""
def main():
  model_path = "/home/konan/Desktop/edgeai/models/tf_models/model_test.h5"
  createCLibrary(model_path)
  print("C Library created..")
  

if __name__ =="__main__":

  main()
"""