import linebot
from linebot import LineBotApi
import tensorflow as tf
global graph,model
graph = tf.get_default_graph()

line_bot_api = LineBotApi('''iZ4eaM+6uM6TPdjBtGaF33vlH56gqR6v1BA3Z4UYdRHQP2wMNpS4/tG/87zuH+uh4B6v3zMk8rEfL0EVZp+C+aF8+HjO6mIgV5IgTQW8ILSOY2TqkjgbNExJOWcxhozOrexzB/M5JFJv2GPSfRIxTQdB04t89/1O/w1cDnyilFU=''')
file_path='./saved_img/downloaded_img1.jpg'
def load_line_img(MESSAGE_ID):
    message_content = line_bot_api.get_message_content(MESSAGE_ID)
    with open(file_path, 'wb') as fd:
        for chunk in message_content.iter_content():
            fd.write(chunk)

from keras.models import model_from_json
# load json and create model
json_file = open('my_model_weight/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("my_model_weight/keras_cifar10_trained_model.h5")

idx2label = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

import numpy as np
from keras.preprocessing import image
def predict_image(image_path):
    test_image = image.load_img(image_path,target_size = (32,32))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    with graph.as_default():
        result = loaded_model.predict(test_image)
    return idx2label[np.argmax(result)]


from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def root():
    return 'you just entered home page'

# Predict
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if(request.json is None):
        return '''enter key 'messageId' as json obj'''
    content = request.get_json()
    MESSAGE_ID = content['messageId']
    if(MESSAGE_ID == '12345'):
        return predict_image(file_path)
    else:
        load_line_img(MESSAGE_ID)
    return predict_image(file_path)


if __name__ == '__main__':
    a='pp.run'
app.run(host='127.0.0.1', port=12345, debug=True)