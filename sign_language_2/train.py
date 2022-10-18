from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


# Loading Data
df = pd.read_csv("data.csv")


# Dividing Features & Labels Data
x = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])


# Spliting Data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.15)


# Normalized Data
x_train = x_train / 255         
x_test = x_test / 255           


# Storing Test Dataset
df_x = pd.DataFrame(x_test)
df_y = pd.DataFrame(y_test)
df_x.to_csv('x_test.csv',  index=False)
df_y.to_csv('y_test.csv',  index=False)


# Creating Brain
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='sigmoid'),
    keras.layers.Dense(50, activation='gelu'),
    keras.layers.Dense(25, activation='softmax')
])

# Compiling Model
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = 'accuracy'
)

# Saving Model
hist = model.fit(x_train, y_train, epochs=10)
model.save('model-1.h5', hist)


print('\nEnjoy!\nThe Brain has been created successfully...\n\n')