
# coding: utf-8

# In[68]:


import keras
from keras.models import Sequential
import math


# In[30]:


from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Dropout, Flatten


# In[36]:


import pandas as pd
import numpy as np


# In[5]:


data = pd.read_csv('data/train.csv')


# In[6]:


data.columns


# In[53]:


labels = data['label'].values
print(labels[:10])


# In[16]:


images = data.drop(['label'], axis=1).values


# In[23]:


NUM_CLASSES = 10
BATCH_SIZE = 32
N_EPOCHS = 20
KERNEL_SIZE = 3
ROWS = 28
COLUMNS = 28
CHANNELS = 1


# In[55]:


train_data = []
val_data = []
train_label = []
val_label = []

# populate training and validation datasets
for i,label in enumerate(labels):
    if (i+1)%5 != 0: # non-multiples of 5 to be in training set, i.e., 80%
        train_label.append([1 if label == i else 0 for i in range(10) ])
        train_data.append(images[i])
    else:
        val_label.append([1 if label == i else 0 for i in range(10)])
        val_data.append(images[i])
# print(train_label[:10])
# print(val_label[:5])
train_data = np.asarray(train_data).reshape(-1,ROWS, COLUMNS, CHANNELS)
val_data = np.asarray(val_data).reshape(-1,ROWS, COLUMNS, CHANNELS)
train_label = np.asarray(train_label)
val_label = np.asarray(val_label)


# In[56]:


# print(len(train_data), len(val_data))
# print(len(train_label), len(val_label))


# In[57]:


model = Sequential()
model.add(Conv2D(BATCH_SIZE, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), activation='relu', input_shape=(ROWS, COLUMNS, CHANNELS)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))


# In[58]:


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
metrics=['accuracy'])


# In[59]:


model.fit(train_data, train_label, batch_size=BATCH_SIZE, epochs=N_EPOCHS, verbose=1, validation_data=(val_data, val_label))
score = model.evaluate(val_data, val_label)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[60]:


test_data = pd.read_csv('data/test.csv')


# In[61]:


test_data.columns


# In[63]:


test_set = np.asarray(test_data.values).reshape(-1, ROWS, COLUMNS, CHANNELS)


# In[66]:


output_labels = model.predict(test_set, verbose=1)


# In[67]:


output_labels[:2]


# In[71]:


result = []
for output_arr in output_labels:
    result.append(np.argmax(output_arr))


# In[72]:


result[:2]


# In[73]:


len(result)


# In[91]:


output_dataframe = pd.DataFrame(result, index=[i+1 for i in range(len(result))], columns=['Label'])


# In[92]:


output_dataframe.index.name = 'ImageId'
output_dataframe


# In[93]:


output_dataframe.to_csv('keras_cnn.csv')

