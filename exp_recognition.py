
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')
import keras

from keras.models import Sequential 
from keras.layers import Dense, Activation, Convolution2D, Reshape, Flatten, MaxPooling2D, Dropout
from keras.utils import np_utils


# In[2]:


ds = pd.read_csv('fer2013.csv')
d = ds.values
d.shape


# In[4]:


arr = np.array([1,2,3])
b = np.array([4, 5, 6])

arr = np.vstack((arr, b))
c = np.array([7, 8 ,9])
arr = np.vstack((arr, c))
print arr

x = [1, 2, 3]
y = [4, 5, 6]

p = np.array(x)
print p

print x


# In[4]:


labels = d[:, 0]
print labels.shape

tem = d[:, 1]
pixels = np.zeros((d.shape[0], 48*48))
# t = d[0, 1]
# t = t.split()
# t = [ int(z) for z in t ]
# data1 = np.array(t)
# print data1

# for ix in range(1, d.shape[0]):
#     tem = d[ix, 1]
#     tem = tem.split()
#     tem = [ int(z) for z in tem ]
#     data = np.vstack((data, np.array(tem)))

for ix in range(pixels.shape[0]):
    t = tem[ix].split(' ')
    for iy in range(pixels.shape[1]):
        pixels[ix, iy] = int(t[iy])


# In[5]:


print pixels.shape
    
print pixels[:2, :10]

#normalize #data preprocessing
pixels -= np.mean(pixels, axis=0)
pixels /= np.std(pixels, axis=0)

print pixels[:2, :10]


# In[6]:


y = np_utils.to_categorical(labels)
print y.shape


# In[7]:


split = int(0.80*pixels.shape[0])

x_train = pixels[:split]
y_train = y[:split]

x_test = pixels[split:]
y_test = y[split:]

print x_train.shape, x_test.shape
print y_train.shape, y_test.shape


# In[8]:


x_train = x_train.reshape((x_train.shape[0], 1, 48, 48))
x_test = x_test.reshape((x_test.shape[0], 1, 48, 48))

print x_train.shape, x_test.shape
print y_train.shape, y_test.shape


# In[9]:


conv_model = Sequential()

conv_model.add(Convolution2D(64, 3, 3, input_shape=(1, 48, 48), activation='relu'))
conv_model.add(Convolution2D(64, 3, 3, activation='relu'))
conv_model.add(Convolution2D(64, 3, 3, activation='relu'))
conv_model.add(MaxPooling2D(pool_size=(2, 2)))

conv_model.add(Convolution2D(32, 3, 3, activation='relu'))
conv_model.add(Convolution2D(32, 3, 3, activation='relu'))
conv_model.add(Convolution2D(32, 3, 3, activation='relu'))
conv_model.add(MaxPooling2D(pool_size=(2, 2)))
conv_model.add(Dropout(0.5))

conv_model.add(Flatten())
conv_model.add(Dense(128, activation='relu'))
conv_model.add(Dense(64, activation='relu'))
conv_model.add(Dropout(0.5))
conv_model.add(Dense(7))
conv_model.add(Activation('softmax'))

conv_model.summary()
#keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
conv_model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])


# In[12]:


hist = conv_model.fit(x_train, y_train,
                     nb_epoch=4,
                     shuffle=True,
                      batch_size=256,
                     validation_data=(x_test, y_test))


# In[13]:


plt.figure(figsize=(14,3))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(hist.history['loss'], 'b', label='Training Loss')
plt.plot(hist.history['val_loss'], 'r', label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(hist.history['acc'], 'b', label='Training Accuracy')
plt.plot(hist.history['val_acc'], 'r', label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()


# In[14]:


test = np.zeros((d.shape[0], 48*48))
for ix in range(test.shape[0]):
    t = tem[ix].split(' ')
    for iy in range(test.shape[1]):
        test[ix, iy] = int(t[iy])


# In[15]:


fig = plt.figure(figsize=(10,10))

for ix in range(30):
    ax = plt.subplot(4,12,ix+1)
    ax.set_title = ix
    plt.imshow(test[split+ix].reshape((48, 48)), cmap='gray')
    plt.axis('off')


# In[298]:


pre = x_test[:100]
print pre.shape
ans = conv_model.predict(pre, batch_size=3)

correct=0
for ix in range(100):
    if np.argmax(ans[ix]) == np.argmax(y_test[ix]):
        correct += 1
        
print correct


# In[16]:


x_eval = x_test[343:3456]
y_eval = y[split+343:split+3456]

score = conv_model.evaluate(x_eval, y_eval, show_accuracy=True, verbose=0)


# In[17]:


print "Score : ", score[0]
print "Accuracy : ", score[1]*100


# In[22]:


import h5py
conv_model.save('face_reco.h5')  # creates a HDF5 file 'my_model.h5'
f = h5py.File('face_reco.h5', 'r+')
del f['optimizer_weights']
f.close()


# In[ ]:




