from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np


def train_model(X_train, y_train, black_box="MLP", random_seed=42):
    # Fits a given black-box model to the training set
    if black_box == "MLP":
        model = MLPRegressor(random_state=random_seed)
    elif black_box == "KNN":
        model = KNeighborsRegressor(random_state=random_seed)
    elif black_box == "SVM":
        model = SVR(random_state=random_seed)
    elif black_box == "XGB":
        model = XGBRegressor(objective='reg:squarederror', random_state=random_seed)
    elif black_box == "Tree":
        model = DecisionTreeRegressor(random_state=random_seed)
    elif black_box == "RF":
        model = RandomForestRegressor(random_state=random_seed)
    else:
        raise NameError("black-box model type unknown")
    model.fit(X_train, y_train)
    return model


from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from .keras_utils import *

def evaluate_model(dataX, 
                    dataY, 
                    n_folds=5, 
                    define_model=None, 
                    batch_size=5, 
                    epochs=50):

    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)

    for train_ix, test_ix in kfold.split(dataX):
        model = define_model()
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX, testY), verbose=0)
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        scores.append(acc)
        histories.append(history)
    return model, scores, histories


def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
    plt.show()

    
def summarize_performance(scores):
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
    plt.boxplot(scores)
    plt.show()
    

def train_keras_model(load_dataset=None, 
                        define_model=None,
                        folds=2,
                        batch_size = 32,
                        epochs = 50):
    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    
    # evaluate model
    model, scores, histories = evaluate_model(trainX, trainY, 
                                                n_folds=folds, 
                                                define_model = define_model,
                                                 batch_size=batch_size,
                                                 epochs=epochs)
    summarize_diagnostics(histories)
    summarize_performance(scores)

    return model


class CModel():
    def __init__(self, model):
        self.model = model
    def predict(self, x):
        return np.argmax(self.model.predict(x), 1)


def simple_1Dmodel(indim=2, nclasses=3, lr=0.01):
    model = Sequential()
    model.add(Dense(8, activation='relu', kernel_initializer='he_uniform', input_shape=(indim,)))
    model.add(Dense(nclasses, activation='softmax'))
    # compile model
    opt = SGD(lr=lr, momentum=0.99)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def simple_CNNmodel(indim=(28, 28, 1), nclasses=10, lr=0.01):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=indim))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=lr, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model