import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint


class TrainFER():
    def __init__(self, data_path='./data/fer2013.csv', model_path='./models/model.h5'):
        self.data_path = data_path
        self.model_path = model_path

        self.num_features = 128
        self.num_labels = 7
        self.batch_size = 64
        self.epochs = 100
        self.width, self.height = 48, 48

        self.data = pd.read_csv(self.data_path)

        self.model = self.build_model()

        self.X_train, self.X_val,self.X_test,\
        self.y_train, self.y_val, self.y_test = self.load_data()

    def load_data(self):
        # get label
        emotions = pd.get_dummies(self.data['emotion']).values

        pixels = self.data['pixels'].tolist()

        faces = []
        for pixel_row in pixels:
            face = [int(pixel) for pixel in pixel_row.split(' ')]
            face = np.asarray(face).reshape(self.width, self.height)
            faces.append(face.astype('float32'))

        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)

        X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=15)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=24)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def build_model(self):
        model = Sequential()

        model.add(Conv2D(self.num_features, kernel_size=(3, 3), activation='relu', input_shape=(self.width, self.height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
        model.add(Conv2D(self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(2*self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(2*self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(2*2*self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(2*2*self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(2**3*self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(2**3*self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())

        model.add(Dense(2**3*self.num_features, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(2*2*self.num_features, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(2*self.num_features, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(self.num_labels, activation='softmax'))

        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                      metrics=['accuracy'])

        print(model.summary())

        self.lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
        self.tensorboard = TensorBoard(log_dir='./logs')
        self.early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
        self.checkpointer = ModelCheckpoint(self.model_path, monitor='val_loss', verbose=1, save_best_only=True)

        return model

    def train(self):
        self.model.fit(np.array(self.X_train), np.array(self.y_train),
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1,
                  validation_data=(np.array(self.X_val), np.array(self.y_val)),
                  shuffle=True,
                  callbacks=[self.lr_reducer, self.tensorboard, self.early_stopper, self.checkpointer])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='Data path', required=False)
    parser.add_argument('-m', '--model', help='Model path', required=False)
    args = vars(parser.parse_args())

    if args['data'] and args['model']:
        tfer = TrainFER(args['data'], args['model'])
    elif args['data']:
        tfer = TrainFER(args['data'])
    elif args['model']:
        tfer = TrainFER(args['model'])
    else:
        tfer = TrainFER()

    tfer.train()