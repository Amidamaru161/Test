from keras.datasets import reuters
from keras import models 
from keras import layers
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical


(train_data,train_laibels),(test_data,test_labels)=reuters.load_data(num_words=10000)

word_index=reuters.get_word_index()
reverse_word_index=dict([(value,key)for(key,value) in word_index.items()])
decoded_newswire=''.join([reverse_word_index.get(i-3,'?')for i in train_data[0]])


def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1
    return results

def to_one_hot(labels,dimension=10000):
    results=np.zeros((len(labels),dimension))
    for i,label in enumerate(labels):
        results[i,label]=1
        return results

x_train=vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)

one_hot_train_labels=to_categorical(train_laibels)
one_hot_test_labels=to_categorical(test_labels)

# one_hot_train_labels=to_one_hot(train_laibels)
# one_hot_train_labels=to_one_hot(test_labels)

model=models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))    



model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'],
              run_eagerly = True)

x_val=x_train[:1000]
partial_x_train=x_train[1000:]

y_val=one_hot_train_labels[:1000]
partial_y_train=one_hot_train_labels[1000:]

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