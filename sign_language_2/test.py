import numpy as np
import pandas as pd 
from keras.models import load_model


# Loading Model
model = load_model('model-1.h5')


# Loading Data
df_x_test = pd.read_csv("x_test.csv")
x_test = np.array(df_x_test)

df_y_test = pd.read_csv("y_test.csv")
y_test = np.array(df_y_test)


# Flattening Labels
y_test = y_test.flatten()


# Prediction According to Test Dataset
y_predicted = model.predict(x_test)
y_predicted_labels = [np.argmax(i) for i in y_predicted]


# Outputs
actual_labels = []
predicted_labels = []


for i in range(10):
    actual_labels.append(y_test[i])
    predicted_labels.append(y_predicted_labels[i])



print('\n======== American Sign Language ========')
print('========== Final Year Project ==========')
print('============= First  Trail =============\n')

print(f'Actual Labels => {actual_labels}')
print(f'Predicted Labels =>{predicted_labels}')
print('\n')
