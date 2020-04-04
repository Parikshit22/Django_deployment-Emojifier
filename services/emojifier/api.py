import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import *
model = Sequential()
model.add(LSTM(64,input_shape=(10,50),return_sequences = True))
model.add(Dropout(0.5))
model.add(LSTM(64,return_sequences = False))
model.add(Dense(5))
model.add(Activation("softmax"))
model.load_weights("services/emojifier/best_model.h5")
model._make_predict_function()
import emoji
emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":beaming_face_with_smiling_eyes:",
                    "3": ":downcast_face_with_sweat:",
                    "4": ":fork_and_knife:",
                   }
f = open("services/emojifier/glove.6B.50d.txt", encoding = 'utf-8')
word_embedding = {}
for line in f:
    values = line.split()
    val = values[0]
    coff = np.asarray(values[1:],dtype = float)
    word_embedding[val] = coff
f.close()
emb_dim = word_embedding["the"].shape
def sec2vec(x):
    maxlen = 10
    state_vec = np.zeros((x.shape[0],maxlen,emb_dim[0]))
    for i in range(x.shape[0]):
        x[i] = x[i].split()
        for j in range(len(x[i])):
            try:
                state_vec[i][j] = word_embedding[x[i][j].lower()]
            except:
                state_vec[i][j] = np.zeros((50,))
    return state_vec

def predict(text):
    data = pd.Series(text)
    pred = model.predict_classes(sec2vec(data))
    return emoji.emojize(emoji_dictionary[str(pred[0])])

if __name__=='__main__':
    print(predict("i want to fight"))