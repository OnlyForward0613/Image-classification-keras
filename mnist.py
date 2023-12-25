#Step 2: Import Numpy and Keras 
import numpy as np

import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout

#Step 3: Load and Preprocess MNIST data
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = (X_train/255) - 0.5
X_test = (X_test/255) - 0.5

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Step 4: Design CNN Architecture
inp = Input(shape=(28,28,1))
i = Conv2D(32, kernel_size=(3,3), activation='relu', name='conv1')(inp)
i = MaxPooling2D(pool_size=(2,2))(i)
i = Conv2D(32, kernel_size=(3,3), activation='relu', name='conv2')(i)    
i = MaxPooling2D(pool_size=(2,2))(i)
i = Flatten()(i)
i = Dense(128, activation='relu', name='dense_1')(i)  
predictions = Dense(num_classes, activation='softmax', name='dense_last')(i)    

model = Model(inputs=inp, outputs=predictions)

#Step 5: Compile and Train The Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=32)

#Step 6: Print Result
scores = model.evaluate(X_test, y_test, verbose=0)
print("Test Accuracy: %.2f%%" % (scores[1]*100))