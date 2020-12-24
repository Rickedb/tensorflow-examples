import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#print(tf.__version__)
cols = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety', 'output']
cars = pd.read_csv('C:/Personal/tensorflow-examples/cars-classification/datasets/cars_dataset.csv', names=cols, header=None)

price = pd.get_dummies(cars.price, prefix='price')
maint = pd.get_dummies(cars.maint, prefix='maint')
doors = pd.get_dummies(cars.doors, prefix='doors')
persons = pd.get_dummies(cars.persons, prefix='persons')
lug_capacity = pd.get_dummies(cars.lug_capacity, prefix='lug_capacity')
safety = pd.get_dummies(cars.safety, prefix='safety')
labels = pd.get_dummies(cars.output, prefix='condition')

print(price)
x = pd.concat([price, maint, doors, persons, lug_capacity, safety], axis=1)
y = labels.values
print(x)
#print(y)
input_layer = Input(shape=(x.shape[1],))
dense_layer_1 = Dense(15, activation='relu')(input_layer)
dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)
output = Dense(y.shape[1], activation='softmax')(dense_layer_2)

model = Model(inputs=input_layer, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

history = model.fit(x_train, y_train, batch_size=8, epochs=50, verbose=0, validation_split=0.2)

score = model.evaluate(x_test, y_test, verbose=0)
print(history)
print("Score:", score[0])
print("Accuracy:", score[1])

test = x_test.iloc[:21]

predictions = model.predict(test)
index = np.argmax(predictions[0])
#print(cols[index])
#print(labels)
#print(labels.values[index])
#print(doors)
print(test)
print(test.columns[np.argmax(labels.values[index])])
print(predictions)
#print(predictions)
#print(model.summary())
