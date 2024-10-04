import os
import string
import numpy as np
from PIL import Image
from pickle import dump, load
import matplotlib.pyplot as plt
from keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

# Load text file into memory
def load_doc(filename):
    with open(filename, 'r') as file:
        return file.read()

# Get all images with their captions
def all_img_captions(filename):
    captions = load_doc(filename).split('\n')[:-1]
    descriptions = {}
    for caption in captions:
        img, caption = caption.split('\t')
        descriptions.setdefault(img[:-2], []).append(caption)
    return descriptions

# Data cleaning
def cleaning_text(captions):
    table = str.maketrans('', '', string.punctuation)
    for img, caps in captions.items():
        captions[img] = [
            ' '.join(
                [word.lower().translate(table) for word in img_caption.split() if len(word) > 1 and word.isalpha()]
            )
            for img_caption in caps
        ]
    return captions

def text_vocabulary(descriptions):
    return set(word for caps in descriptions.values() for desc in caps for word in desc.split())

def save_descriptions(descriptions, filename):
    with open(filename, 'w') as file:
        file.write('\n'.join(f"{key}\t{desc}" for key, desc_list in descriptions.items() for desc in desc_list))

def extract_features(directory):
    model = Xception(include_top=False, pooling='avg')
    features = {}
    for img in os.listdir(directory):
        filename = os.path.join(directory, img)
        image = Image.open(filename).resize((299, 299))
        feature = model.predict(preprocess_input(np.expand_dims(np.array(image), axis=0)))
        features[img] = feature
    return features

# Setup
dataset_text = 'Flickr8k_text'
dataset_images = 'Flicker8k_Dataset'
filename = os.path.join(dataset_text, 'Flickr8k.token.txt')
descriptions = cleaning_text(all_img_captions(filename))
save_descriptions(descriptions, 'descriptions.txt')

# Extract features
features = extract_features(dataset_images)
dump(features, open('features.p', 'wb'))

# Load and clean descriptions
def load_photos(filename):
    return load_doc(filename).split('\n')[:-1]

def load_clean_descriptions(filename, photos):
    file = load_doc(filename)
    return {
        line.split()[0]: [f"<start> {' '.join(line.split()[1:])} <end>"]
        for line in file.split('\n') if line.split()[0] in photos
    }

def load_features(photos):
    return {k: load(open('features.p', 'rb'))[k] for k in photos}

# Load training data
filename = os.path.join(dataset_text, 'Flickr_8k.trainImages.txt')
train_imgs = load_photos(filename)
train_descriptions = load_clean_descriptions('descriptions.txt', train_imgs)
train_features = load_features(train_imgs)

# Convert dictionary to list
def dict_to_list(descriptions):
    return [d for desc_list in descriptions.values() for d in desc_list]

# Tokenizer
def create_tokenizer(descriptions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(dict_to_list(descriptions))
    return tokenizer

# Tokenize and save
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.p', 'wb'))

# Calculate maximum length of descriptions
max_length = max(len(d.split()) for d in dict_to_list(train_descriptions))

# Create data generator
def data_generator(descriptions, features, tokenizer, max_length):
    while True:
        for key, desc_list in descriptions.items():
            feature = features[key][0]
            input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, desc_list, feature)
            yield [[input_image, input_sequence], output_word]

def create_sequences(tokenizer, max_length, desc_list, feature):
    X1, X2, y = [], [], []
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=len(tokenizer.word_index) + 1)[0]
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

# Define the model
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,))
    fe2 = Dense(256, activation='relu')(Dropout(0.5)(inputs1))
    inputs2 = Input(shape=(max_length,))
    se3 = LSTM(256)(Dropout(0.5)(Embedding(vocab_size, 256, mask_zero=True)(inputs2)))
    decoder1 = add([fe2, se3])
    outputs = Dense(vocab_size, activation='softmax')(Dense(256, activation='relu')(decoder1))
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

# Training
model = define_model(len(tokenizer.word_index) + 1, max_length)
epochs = 10
steps = len(train_descriptions)

# Create models directory
os.makedirs("models", exist_ok=True)

for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
    model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save(f"models/model_{i}.h5")

# Image captioning
img_path = input("Enter image path: ")

def extract_image_features(filename, model):
    try:
        image = Image.open(filename).resize((299, 299))
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        feature = model.predict(preprocess_input(np.expand_dims(np.array(image), axis=0)))
        return feature
    except Exception as e:
        print(f"ERROR: Couldn't open image! {str(e)}")

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for _ in range(max_length):
        sequence = pad_sequences([tokenizer.texts_to_sequences([in_text])[0]], maxlen=max_length)
        pred = np.argmax(model.predict([photo, sequence], verbose=0))
        word = word_for_id(pred, tokenizer)
        if not word or word == 'end':
            break
        in_text += f' {word}'
    return in_text

# Load model and extract features
model = load_model('models/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")
photo = extract_image_features(img_path, xception_model)
description = generate_desc(model, tokenizer, photo, max_length)
print("\n\nCaption:", description)
plt.imshow(Image.open(img_path))
plt.show()
