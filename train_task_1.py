import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras import optimizers
from keras.backend.tensorflow_backend import set_session
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf
import model
from toolkit import tools

modelPath = "/mnt/HDD_1/yanshuo/Task_1/model/weights.hdf5"
logPath = "/mnt/HDD_1/yanshuo/Task_1/model/log.txt"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

data = np.load("/mnt/HDD_1/yanshuo/Task_1/data.npy")
data_topo = np.load("/mnt/HDD_1/yanshuo/Task_1/data_topo.npy")
label = np.load("/mnt/HDD_1/yanshuo/Task_1/label.npy")

# normalize data
data_scaler = StandardScaler()
data_topo_scaler = StandardScaler()
train_data = data_scaler.fit_transform(data).reshape((data.shape[0], data.shape[1], 1))
train_topo_data = data_topo_scaler.fit_transform(data_topo.reshape((data_topo.shape[0], data_topo.shape[1]**2)))
train_topo_data = train_topo_data.reshape((data_topo.shape[0], data_topo.shape[1], data_topo.shape[2], 1))
train_label = to_categorical(label, num_classes=6)
indices = np.arange(data.shape[0])
X_train_ind, X_test_ind, y_train_ind, y_test_ind = train_test_split(indices, indices,
                                                                    stratify=label, 
                                                                    test_size=0.2)
print("data shape:", train_data.shape)
print("data_topo shape:", train_topo_data.shape)
print("label shape:", train_label.shape)

# training
lr = 0.001
batch_size = 64
ad = optimizers.Adam(lr=lr)
callbacks = tools.get_callbacks(modelPath, logPath)
model_structure = model.structure(6, 1, 21, 21)
model_structure.compile(loss='categorical_crossentropy', optimizer=ad, metrics=['accuracy'])
train_history = model_structure.fit(x=[train_data[X_train_ind], train_topo_data[X_train_ind]], y=train_label[y_train_ind],
                            validation_split=0.2, epochs=50
                            ,batch_size=batch_size,
							callbacks=callbacks, verbose=1, shuffle=True)

validation_prediction = model_structure.predict([train_data[X_test_ind], train_topo_data[X_test_ind]], batch_size=1)
pred = [i.argmax() for i in validation_prediction]
score = f1_score(label[y_test_ind], pred, average=None)
print(score)

print("")