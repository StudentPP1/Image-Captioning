from model import Network
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from PIL import Image
import matplotlib.pyplot as plt
import os

model = Network()
actual, predicted = list(), list()

for key in tqdm(model.test_data):
    # get actual caption
    captions = model.mapping[key]
    # predict the caption for image
    y_pred = model.predict_caption(model.preprocessing.get_feature(key), True)
    # split into words
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    # append to the list
    actual.append(actual_captions)
    predicted.append(y_pred)

# calcuate BLEU score
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))


def generate_caption(image_name):
    image_id = image_name.split('.')[0]
    img_path = os.path.join(model.preprocessing.images_dir, image_name)
    image = Image.open(img_path)
    captions = model.mapping[image_id]

    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)

    # predict the caption
    y_pred = model.predict_caption(model.preprocessing.get_feature(image_id), True)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)
    plt.show()


generate_caption("1001773457_577c3a7d70.jpg")
generate_caption("1002674143_1b742ab4b8.jpg")
