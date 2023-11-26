import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.utils import to_categorical
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from preprocessing import Preprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Network:
    def __init__(self, epochs=20, batch_size=16):
        self._preprocessing = Preprocessing()
        self._mapping = self._preprocessing.mapping
        self._max_caption_length = self._preprocessing.max_caption_length
        self._vocab_size = self._preprocessing.vocab_size

        self.epochs = epochs
        self.batch_size = batch_size

        image_ids = list(self._mapping.keys())
        split = int(len(image_ids) * 0.90)

        self._train_data = image_ids[:split]
        self._test_data = image_ids[split:]
        self._steps = len(self._train_data) // self.batch_size

        if not ("model.h5" in os.listdir(self._preprocessing.save_dir)):
            self._model = self._train()
        else:
            self._model = load_model(f"{self._preprocessing.save_dir}/model.h5")

    # create data generator to get data in batch (avoids session crash)
    def _data_generator(self, data_keys, mapping, max_length, vocab_size):
        # loop over images
        x1, x2, y = list(), list(), list()
        n = 0
        while 1:
            for key in data_keys:
                n += 1
                captions = mapping[key]
                # process each caption
                for caption in captions:
                    # encode the sequence
                    seq = self._preprocessing.text_encoder(caption)
                    # split the sequence into X, y pairs
                    for i in range(1, len(seq)):
                        # split into input and output pairs
                        in_seq, out_seq = seq[:i], seq[i]
                        # pad input sequence
                        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                        # encode output sequence
                        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                        # store the sequences
                        # x1.append(features[key][0])
                        x1.append(self._preprocessing.get_feature(key))
                        x2.append(in_seq)
                        y.append(out_seq)

                if n == self.batch_size:
                    x1, x2, y = np.array(x1), np.array(x2), np.array(y)
                    yield [x1, x2], y
                    x1, x2, y = list(), list(), list()
                    n = 0

    def _train(self):
        model = self._get_model()

        for i in range(self.epochs):
            # create data generator
            generator = self._data_generator(self._train_data,
                                             self._mapping,
                                             self._max_caption_length,
                                             self._vocab_size)
            # fit for one epoch
            model.fit(generator, epochs=1, steps_per_epoch=self._steps, verbose=1)

        # save the model
        model.save(f"{self._preprocessing.save_dir}/model.h5")
        return model

    def _get_model(self):
        # encoder model
        # image feature layers
        inputs1 = Input(shape=(self._preprocessing.model_output_size,))
        fe1 = Dropout(0.5)(inputs1)  # 0.4
        fe2 = Dense(256, activation="relu")(fe1)
        # sequence feature layers
        inputs2 = Input(shape=(self._max_caption_length, ))
        se1 = Embedding(self._vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)  # 0.4
        se3 = LSTM(256)(se2)

        # decoder model
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation="relu")(decoder1)
        outputs = Dense(self._vocab_size, activation="softmax")(decoder2)

        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss="categorical_crossentropy", optimizer="adam")

        return model

    def predict_caption(self, image_path, testing_mode=False, progress_bar=None, progress_label=None, window=None):
        if progress_bar and progress_label and window:
            progress_bar.step()
            progress_label.configure(text="Preprocessing image...")
            window.update()

        if not testing_mode:
            image = self._preprocessing.preprocess_image(image_path)
        else:
            image = image_path

        # add start tag for generation process
        in_text = self._preprocessing.start_seq

        if progress_bar and progress_label:
            progress_bar.step()
            progress_label.configure(text="Generating caption...")
            window.update()

        # iterate over the max length of sequence
        for i in range(self._max_caption_length):
            # encode input sequence
            sequence = self._preprocessing.text_encoder(in_text)
            # pad the sequence
            sequence = pad_sequences([sequence], self._max_caption_length)
            # predict next word
            image = image.reshape((1, self._preprocessing.model_output_size))

            yhat = self._model.predict([image, sequence], verbose=0)
            # get index with high probability
            yhat = int(np.argmax(yhat))
            # convert index to word
            word = self._preprocessing.idx_to_word(yhat)
            # stop if word not found
            if word is None:
                break
            # append word as input for generating next word
            in_text += " " + word
            # stop if we reach end tag
            if word == self._preprocessing.end_seq:
                break

        if progress_bar and progress_label:
            progress_bar.step()
            progress_label.configure(text="Cleaning caption...")
            window.update()

        if not testing_mode:
            in_text = in_text.replace(self._preprocessing.start_seq, '').replace(self._preprocessing.end_seq, '')

        return in_text

    @property
    def preprocessing(self):
        return self._preprocessing

    @property
    def mapping(self):
        return self._mapping

    @property
    def test_data(self):
        return self._test_data

