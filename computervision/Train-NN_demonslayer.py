#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[2]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


# In[3]:


img = image.load_img("D:/computervision/training/kanao/1 (Custom).jpg")


# In[4]:


plt.imshow(img)


# In[5]:


cv2.imread("D:/computervision/training/kanao/1 (Custom).jpg")


# In[6]:


cv2.imread("D:/computervision/training/kanao/1 (Custom).jpg").shape


# In[7]:


train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)


# In[8]:


train_dataset = train.flow_from_directory('D:/computervision/training',
                                         target_size = (200,200),
                                         batch_size = 3,
                                         class_mode = 'categorical')

validation_dataset = train.flow_from_directory("D:/computervision/validation",
                                         target_size = (200,200),
                                         batch_size = 3,
                                         class_mode = 'categorical')


# In[9]:


train_dataset.class_indices


# In[10]:


train_dataset.classes


# In[11]:


import tensorflow as tf
model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3), activation = 'relu', input_shape = (200,200,3)),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   #
                                   tf.keras.layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (200,200,3)),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   #
                                   tf.keras.layers.Conv2D(64,(3,3), activation = 'relu', input_shape = (200,200,3)),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   ##
                                   tf.keras.layers.Flatten(),
                                   ##
                                   tf.keras.layers.Dense(512, activation = 'relu'),
                                   ##
                                   tf.keras.layers.Dense(3, activation = 'softmax')
                                   ])


# In[ ]:





# In[12]:


model.compile(loss = 'categorical_crossentropy',
             optimizer = RMSprop(lr=0.001),
             metrics = ['accuracy'])


# In[13]:


model_fit = model.fit(train_dataset,
                     steps_per_epoch = 3,
                     epochs = 100,
                     validation_data = validation_dataset)


# In[14]:


import numpy as np
import os
from tensorflow.keras.preprocessing import image

dir_path = 'D:/computervision/testing'

for i in os.listdir(dir_path):
    img = image.load_img(dir_path + '//' + i, target_size=(200, 200))
    plt.imshow(img)
    plt.show()

    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    images = np.vstack([X])
    val = model.predict(images)

    # Get the predicted class label
    label = np.argmax(val)

    if label == 0:
        print("Kanao")
    elif label == 1:
        print("Nezuko")
    elif label == 2:
        print("Tanjiro")
    else:
        print("kimetsu-no-yaiba")


# In[ ]:





# In[ ]:




