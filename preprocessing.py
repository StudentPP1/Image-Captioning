import io
import json
from PIL import Image
import os
from keras.src.utils import load_img, img_to_array
from tqdm import tqdm
import numpy as np
from keras.models import Model
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.applications.mobilenet import MobileNet, preprocess_input


class Preprocessing:
    def __init__(self):
        self._model = VGG16()
        self.start_seq = "startseq"
        self.end_seq = "endseq"

        self.captions_file = "Dataset/Flickr dataset/captions.txt"
        self.images_dir = "Dataset/Flickr dataset/Images"
        self.save_dir = "Saves"
        self.features_dir = "Saves/features"

        # leave the last layer, because we need features of image
        self._model = Model(inputs=self._model.inputs, outputs=self._model.layers[-2].output)
        self.model_output_size = self._model.layers[-2].output_shape[-1]

        with open(self.captions_file, 'r') as f:
            next(f)
            captions_doc = f.read()

        # create mapping of image to captions
        self.mapping = {}
        all_captions = []

        if not os.path.exists(self.features_dir):
            os.mkdir(self.features_dir)

        if not bool(os.listdir(self.features_dir)):
            directory = self.images_dir

            for img_name in tqdm(os.listdir(directory)):
                try:
                    # load the image from file
                    img_path = directory + '/' + img_name
                    # get image ID
                    image_feature_id = img_name.split('.')[0]
                    # image = load_img(img_path, target_size=(224, 224))
                    image = Image.open(img_path).resize((224, 224), Image.NEAREST)
                    # convert image pixels to numpy array
                    image = img_to_array(image)
                    # reshape data for model
                    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
                    # preprocess image for mobile net
                    image = preprocess_input(image)
                    # extract features
                    feature = self._model.predict(image, verbose=0)

                    np.save(f"{self.features_dir}/{image_feature_id}.npy", feature)

                except Exception as ex:
                    print(ex)
                    continue

        features_list = [img.split('.')[0] for img in os.listdir(self.features_dir)]

        # process lines
        for line in tqdm(captions_doc.split('\n')):
            if line != '':
                # split the line by comma(,)
                tokens = line.split(',')

                if len(line) < 2:
                    continue

                image_id, caption = tokens[0], tokens[1:]
                # remove extension from image ID
                image_id = image_id.split('.')[0]
                # convert caption list to string
                caption = " ".join(caption)
                # create list if needed
                if image_id in features_list:

                    if image_id not in self.mapping:
                        self.mapping[image_id] = []
                    #  store the caption
                    self.mapping[image_id].append(caption)

        print(len(self.mapping), len(os.listdir(self.features_dir)))
        print(self.mapping['1000268201_693b08cb0e'])

        for key, captions in self.mapping.items():
            for i in range(len(captions)):
                caption = captions[i]
                captions[i] = self._clean(caption)

        print(self.mapping['1000268201_693b08cb0e'])

        for key in self.mapping:
            for caption in self.mapping[key]:
                all_captions.append(caption)

        print(len(all_captions))
        print(all_captions[:10])

        if "tokenizer.json" in os.listdir(self.save_dir):
            with open(f"{self.save_dir}/tokenizer.json") as f:
                data = json.load(f)
                self._tokenizer = tokenizer_from_json(data)
        else:
            self._tokenizer = Tokenizer()
            self._tokenizer.fit_on_texts(all_captions)

            tokenizer_json = self._tokenizer.to_json()

            with io.open(f"{self.save_dir}/tokenizer.json", 'w', encoding="utf-8") as f:
                f.write(json.dumps(tokenizer_json, ensure_ascii=False))

        self.vocab_size = len(self._tokenizer.word_index) + 1
        self.max_caption_length = max(len(caption.split()) for caption in all_captions)

        print(self.vocab_size)
        print(self.max_caption_length)

    def preprocess_image(self, image_file):
        image = Image.open(image_file).resize((224, 224), Image.NEAREST)
        img_array = np.array(image)
        img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
        preprocess_image = preprocess_input(img_array)
        feature = self._model.predict(preprocess_image, verbose=0)
        return feature

    def get_feature(self, key):
        array = np.load(f"{self.features_dir}/{key}.npy")
        return array.reshape((array.shape[-1], ))

    def _clean(self, text: str) -> str:
        # clean the caption
        preprocess_caption = text.lower().replace('[^A-Za-z]', '').replace('\s+', ' ')
        ready_caption = f'{self.start_seq} ' + " ".join([word for word in preprocess_caption.split() if len(word) > 1]) \
                        + f' {self.end_seq}'
        return ready_caption

    def text_encoder(self, text: str) -> np.array:
        token_list = self._tokenizer.texts_to_sequences([text])[0]
        return token_list

    def idx_to_word(self, index: int) -> str or None:
        for word, ind in self._tokenizer.word_index.items():
            if ind == index:
                return word
        return None
