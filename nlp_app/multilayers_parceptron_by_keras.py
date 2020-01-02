from keras.layers import Dense
from keras.models import Sequential

model = Sequential()

model.add(Dense(units=2, activation='relu', input_dim=3))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

