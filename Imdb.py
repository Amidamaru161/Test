from keras.datasets import imdb
from keras import models 
from keras import layers
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt


(train_data,train_laibels),(test_data,test_labels)=imdb.load_data(num_words=10000)

word_index=imdb.get_word_index()
reverse_word_index=dict([(value,key)for(key,value) in word_index.items()])
decoded_review=''.join([reverse_word_index.get(i-3,'?')for i in train_data[0]])

def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1
    return results


x_train=vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)

y_train=np.asarray(train_laibels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')
  




model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))


x_val=x_train[:10000]
partial_x_train=x_train[10000:]

y_val=y_train[:10000]
partial_y_train=y_train[10000:]

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

history=model.fit(partial_x_train,
                  partial_y_train,
                  epochs=30,
                  batch_size=512,
                  validation_data=(x_val,y_val))

history_dict=history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='Точность на обучающей выборке')
plt.plot(epochs,val_acc,'b',label='Точность на тестовой выборке')


plt.title('Тренировачная и тестовая ошибка')
plt.xlabel('Эпох')
plt.ylabel('Ошибка')
plt.legend()
plt.show()


