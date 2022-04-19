from matplotlib import image
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps

X = np.load('Alphabet_Recognition.npz')
Y = pd.read_csv('Alphabet_Recognition.csv')
print(pd.Series(Y).value_counts())

classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses = len(classes)
xtrain, xtest, ytrain, ytest = train_test_split(random_state=9,test_size=3500,train_size=500)
lr = LogisticRegression(solver="saga", multi_class='multinomial').fit(xtrain,ytrain)

def get_prediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((22,30),Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized,pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,660)
    test_pred = lr.predict(test_sample)
    return test_pred[0]