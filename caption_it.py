#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import add
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
import numpy as np
from tensorflow.keras.models import Model,load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt


# In[2]:


model = load_model("model_weights/model_9.h5")
# In[3]:
model_temp = ResNet50(weights="imagenet",input_shape=(224,224,3))

# In[7]:

model_resnet = Model(model_temp.input,model_temp.layers[-2].output)

# In[5]:

def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    #normalisation
    img = preprocess_input(img)
    return img

# In[9]:

def encode_img(img):
    img = preprocess_img(img)
    feature_vector = model_resnet.predict(img)
#     print(feature_vector.shape)
    feature_vector = feature_vector.reshape((1,-1))
#     print(feature_vector.shape)
    return feature_vector

# In[19]:

w2i= open("C:/Users/arpii/Desktop/Flask tutorials/word_2_idx.pkl","rb")
word_to_idx = pickle.load(w2i)

# In[20]:

i2w= open("C:/Users/arpii/Desktop/Flask tutorials/idx_2_word.pkl","rb")
idx_to_word = pickle.load(i2w)

# In[21]:

def predict_caption(photo):
    max_len=35
    in_text = "<s>"
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax() #Word with max prob always - Greedy Sampling
        word = idx_to_word[ypred]
        in_text += (' ' + word)
        
        if word == "<e>":
            break
    
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption


# In[25]:

def caption_image(image):
#     print("workong")
    enc = encode_img("C:/Users/arpii/Desktop/Flask tutorials/"+image)
    caption=predict_caption(enc)
    return caption

# In[ ]:




