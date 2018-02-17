from flask import Flask, render_template, request
import os
from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
import re
from gtts import gTTS

app = Flask(__name__)

# extract features from each photo in the directory
def extract_features(filename):
	model = VGG16()
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	image = load_img(filename, target_size=(224, 224))
	image = img_to_array(image)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	feature = model.predict(image, verbose=0)
	return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	in_text = 'startseq'
	for i in range(max_length):
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text

@app.route('/')
def upload_file():
	html = '<html><link rel="stylesheet" type="text/css" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css"><body><form action = "http://localhost:5000/upload" method = "POST" enctype = "multipart/form-data"><center><input type = "file" name = "file" class="btn btn-danger"/><input type = "submit" class="btn btn-success"/></center></form></body></html>'
	return html

@app.route('/upload', methods = ['GET', 'POST'])
def process_file():
	if request.method == 'POST':
		f = request.files['file']
		if (f.filename.split('.')[1] == 'jpg' or f.filename.split('.')[1] == 'jpeg' or f.filename.split('.')[1] == 'png') :
			f.save(f.filename)
			print(f.filename)
			# load the tokenizer
			tokenizer = load(open('tokenizer.pkl', 'rb'))
			# pre-define the max sequence length (from training)
			max_length = 34
			# load the model
			model = load_model('model-ep004-loss1.809-val_loss4.163.h5')
			# load and prepare the photograph
			photo = extract_features(f.filename)
			# generate description
			description = generate_desc(model, tokenizer, photo, max_length)
			description = re.sub('startseq', "", description)
			tts = gTTS(text=description, lang='en')
			tts.save("C:/Users/Aayush/Desktop/listen.mp3")
			os.system("mpg321 C:/Users/Aayush/Desktop/listen.mp3")
			return description
		else:
			return "Only jpg, jpeg and png file are supported"

if __name__ == '__main__':
	app.run(debug=True, port=5000)