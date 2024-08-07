from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np

np.random.seed(9)
model = Sequential()
model.add(Dense(units=4, activation='sigmoid', input_dim=3))
model.add(Dense(units=1, activation='sigmoid'))

print(model.summary())
print('')

sgd = optimizers.SGD(learning_rate=1.0)
model.compile(loss='mean_squared_error', optimizer=sgd)

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])
model.fit(X, y, epochs=1500, verbose=False)
predictions = model.predict(X)

print(predictions)

