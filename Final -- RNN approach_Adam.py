#!/usr/bin/env python
# coding: utf-8

# In[6]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_pipeline import transformation_pipeline
from sklearn.model_selection import train_test_split
import tensorflow as tf


# In[7]:


data = pd.read_csv('content/preprocessed_train.csv')


# In[8]:


pipeline, data_cleaned = transformation_pipeline(
    data, building_id=122, meter=0, primary_use=99)


# In[ ]:


transformed_data = pipeline.fit_transform(data_cleaned)


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(transformed_data[:, 1:],
                                                  transformed_data[:, 0],
                                                  test_size=0.2,
                                                  shuffle=False,
                                                  random_state=2021)



train_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(x_train,
                                                                y_train,
                                                                length=6, sampling_rate=1,
                                                                stride=1, batch_size=32
                                                                )

val_gen = tf.keras.preprocessing.sequence.TimeseriesGenerator(x_val,
                                                              y_val,
                                                              length=6, sampling_rate=1,
                                                              stride=1, batch_size=32
                                                              )


# In[ ]:


model = tf.keras.Sequential([tf.keras.layers.SimpleRNN( 128, activation='relu',
                                                  return_sequences=False),
                            tf.keras.layers.Dense(1)])


# In[ ]:


model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.0001))

cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                      patience=15,
                                      restore_best_weights=True)
# Fitting the model
history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=100,
                    callbacks=[cb],
                    shuffle=False)


# In[ ]:


model.save('models/RNN_ADAM')


# In[13]:


predicted_batch_7 = model.predict(val_gen[7][0])


# In[14]:


_, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(32),
        predicted_batch_7,
        color='green', label='Predicted')

ax.plot(range(32),
        val_gen[7][1],
        color='red', label='Actual')
ax.legend()

plt.show()


# In[15]:


predicted = []
actual = []
for i in range(32):
    predicted.extend(model.predict(val_gen[i][0]))
    actual.extend(val_gen[i][1])


# In[16]:


print('Testing Loss= ', np.mean(tf.keras.losses.MSE(actual, predicted)))
# Testing Loss=  0.04267618


# In[17]:


fig, (ax1, ax2, ax) = plt.subplots(3, 1,  figsize=(30, 15), sharex=True)

ax1.plot(range(len(actual)),
         predicted,
         color='green', marker='o', linestyle='dashed', label='Predicted')
plt.legend()

ax2.plot(range(len(actual)),
         actual,
         color='red', marker='x', label='Actual')
plt.legend()

ax.plot(range(len(actual)),
        predicted,
        color='green', linestyle='dashed',
        label='Predicted')
plt.legend()
ax.plot(range(len(actual)),
        actual,
        color='red',
        label='actual')

plt.legend()

plt.title('Test_set', loc='center')

plt.show()


# In[18]:


predicted_t = []
actual_t = []
for i in range(32):
    predicted_t.extend(model.predict(train_gen[i][0]))
    actual_t.extend(train_gen[i][1])


# In[19]:


fig, (ax1, ax2, ax) = plt.subplots(3, 1,  figsize=(30, 15), sharex=True)

ax1.plot(range(len(actual_t)),
         predicted_t,
         color='green', marker='o', linestyle='dashed',
         label='Predicted')

ax2.plot(range(len(actual_t)),
         actual_t,
         color='red', marker='x', label='Actual')


ax.plot(range(len(actual_t)),
        predicted_t,
        color='green', linestyle='dashed',
        label='Predicted')

ax.plot(range(len(actual_t)),
        actual_t,
        color='red',
        label='actual')
plt.title('Train_set', loc='center')

plt.legend()

plt.show()


# In[20]:


model.summary()
# Total params: 18,433


# In[ ]:




