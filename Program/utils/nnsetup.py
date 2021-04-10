from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split

def get_datasets():
  df = pd.read_csv("data/train.csv")
  
  X, Y = df.drop('label', axis=1), df['label']
  
  X_train, X_test, Y_train, Y_test = train_test_split(X.to_numpy(), Y.values.reshape(
      (Y.shape[0], 1)), test_size=0.2, random_state=10)

  X_train = X_train / 255
  X_test = X_test / 255
  
  encoder = OneHotEncoder()

  Y_train_e = encoder.fit_transform(Y_train).toarray()

  return X_train, Y_train, Y_train_e, X_test, Y_test
