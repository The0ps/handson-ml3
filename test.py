import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
def preprocess_data():
    df = pd.read_json('ETH_USDT-1h-futures.json')
    df['label'] = df[4].shift(-1)
    df.drop(df[-1:].index,inplace=True)
    # train, test = train_test_split(df, test_size=0.2, random_state=8675309, shuffle=False)
    X = df[[1,2,3,4,5]]
    y = df.label

    X_pipeline = make_pipeline(StandardScaler())
    y_pipeline = make_pipeline(StandardScaler())
    X = X_pipeline.fit_transform(X)
    y = y_pipeline.fit_transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42,shuffle=False)

    params = {
        "batch_size": 20,  # 20<16<10, 25 was a bust
        "epochs": 300,
        "lr": 0.00010000,
        "time_steps": 8
    }
    from keras.models import Sequential, load_model
    from keras.layers import Dense, Dropout
    from keras.layers import LSTM
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
    from keras import optimizers

    X_train, y_train = pd.DataFrame(X_train),pd.DataFrame(y_train)
    def build_timeseries(X_train,y_train):
        X, y = [], []
        for i in range(params["time_steps"], X_train.shape[0]):
            X.append(X_train.iloc[i-params["time_steps"]:i, :])
            y.append(y_train.iloc[i])
        return np.array(X), np.array(y)

    X_train,y_train = build_timeseries(X_train, y_train)
    X_test, y_test = pd.DataFrame(X_test),pd.DataFrame(y_test)
    X_test, y_test = build_timeseries(X_test, y_test)
def create_model():

    lstm_model = Sequential()
    # (batch_size, timesteps, data_dim)
    lstm_model.add(LSTM(100, batch_input_shape=(params['batch_size'], params['time_steps'], X_train.shape[2]),
                        dropout=0.0,  stateful=True,recurrent_dropout=0.0, return_sequences=True,
                        kernel_initializer='random_uniform'))
    lstm_model.add(Dropout(0.4))
    lstm_model.add(LSTM(60, dropout=0.0))
    lstm_model.add(Dropout(0.4))
    lstm_model.add(Dense(20,activation='relu'))
    lstm_model.add(Dense(1,activation='sigmoid'))
    # optimizer = optimizers.RMSprop(lr=60)
    optimizer = optimizers.SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
    return lstm_model
def lstm():
    
def train_model(model):
    trimmer = X_test.shape[0] % params['batch_size']
    X_test = X_test[:-trimmer]
    y_test = y_test[:-trimmer]

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                        patience=10)

    # mcp = ModelCheckpoint(os.path.join(
    #                         "best_model.h5"), monitor='val_loss', verbose=1,
    #                         save_best_only=True, save_weights_only=False, mode='min', period=1)
    r_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, 
                                    verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    history = model.fit(X_train, y_train, epochs=500, verbose=2, batch_size=20,shuffle=False,validation_data=(X_test, y_test), callbacks=[es])
    
def plot():
    from matplotlib import pyplot as plt
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')








