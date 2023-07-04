# import numpy as np # linear algebra
# import pandas as pd # data processing

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# PATH = '../input/cropdata/'
# file_list = [PATH + f for f in os.listdir(PATH) if f.startswith('data_file')]
# csv_list = []

# for file in file_list:
#     csv_list.append(pd.read_csv(file))

# csv_merged = pd.concat(csv_list, ignore_index=True)
# csv_merged.to_csv('./' + 'data_file_crops.csv', index=False)

# df = pd.read_csv('./data_file_crops.csv')
# print(df.columns)

# df.shape
# df.tail()

# df.info()

# df.drop(df.iloc[:, 4:8], inplace=True, axis=1) # droping the unnecessary labels

# df.head()
# df.describe()

# print(df.isna().sum())
# import matplotlib.pyplot as plt
# import seaborn as sb
# correlation = df.corr()

# sb.heatmap(correlation)
# plt.show()

# df.CROP.unique()

# # data preprocessing
# # Cauliflower->1
# # Onion->2
# # Ginger->3
# # Garlic->4
# # Tomato->5
# df.loc[df['CROP'] == 'Cauliflower', "CROP"] = 1
# df.loc[df['CROP'] == 'Onion', "CROP"] = 2
# df.loc[df['CROP'] == 'Ginger', "CROP"] = 3
# df.loc[df['CROP'] == 'Garlic', "CROP"] = 4
# df.loc[df['CROP'] == 'Tomato', "CROP"] = 5

# data = df.values
# np.random.shuffle(data)
# row, col = data.shape

# Y = data[: , 0]
# X = data[: , 1 : ]
# print(Y.shape, X.shape)
# X = X.astype('float32')
# Y = Y.astype('float32')
# print(type(X[0][0]))

# split = int(0.8*row)
# print(split, row-split)

# Y_train = Y[ : split]
# Y_test = Y[split :]
# X_train = X[ : split, : ]
# X_test = X[split : , : ]

# print(Y_train.shape, Y_test.shape)
# print(X_train.shape, X_test.shape)

# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import SGD, Adam
# from keras.utils.np_utils import to_categorical
# Y_train = Y_train - 1
# Y_test = Y_test - 1
# Y_train = to_categorical(Y_train)
# Y_test = to_categorical(Y_test)

# print(Y_train.shape, Y_train[0]) # categorizes into vectors of size num of categories
# print(X_train.shape)

# # Model 
# model = Sequential()
# model.add(Dense(16, input_dim = 4,  activation='relu'))
# model.add(Dense(8, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(5, activation='softmax'))
# model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

# model.summary()
# X_train.shape, Y_train.shape

# model.fit(X_train, Y_train, verbose=1, epochs=10)

# from sklearn.metrics import classification_report, confusion_matrix

# y_test_arg = np.argmax(Y_test, axis=1)
# Y_pred = np.argmax(model.predict(X_test), axis=1)

# # print(Y_pred[1])
# # print(X_test[1])
# # print(y_test_arg[1])