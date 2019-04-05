import os
import time
import random
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
from math import log
from math import ceil
from keras.optimizers import Adam
from keras.models import save_model, load_model, model_from_json
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.activations import softmax, relu
from keras.optimizers import Adam

curdir = os.getcwd()


# ----------------------------------------------------------------------------------------------------------------------
# DEFINE FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------

def load_models(path):
    '''
    :return: a dictionairy with model names as keys, and the models themselves as values
    '''
    os.chdir(path)
    import pathlib
    files = os.listdir()
    nr_models = 0
    for file in files:
        if 'model' in file:
            nr_models += 1

    models = dict()
    for i in range(nr_models):
        this_model = dict()
        with open('model{}'.format(i), 'rb') as pickle_in:
            model_dic = pickle.load(pickle_in)
        model = model_from_json(model_dic['json_model'])
        model.set_weights(model_dic['weights'])
        this_model['model'] = model
        # add validation accuracy if applicable
        try:
            acc = model_dic['val_acc']
            this_model['val_acc'] = acc
        except KeyError:
            pass
        # add Beta for boosted models if applicable
        try:
            Beta = model_dic['Beta']
            this_model['Beta'] = Beta
        except KeyError:
            pass

        models['model{}'.format(i)] = this_model
    print('models loaded')
    return models


def plot_results(results, title):
    # linestyles = ['-', '--', ':']
    linestyles = ['-', '-', '-', '-']

    fig = plt.figure()
    axes = plt.gca()
    axes.set_ylim([0.81, 0.875])
    locs, labels = xticks()
    xticks(np.arange(1, 26, 1.0), [i for i in range(1, 26)])
    plt.xticks(fontsize=8)

    labels = ['weighted_absolute', 'mean', 'weighted_ranked']
    if title == 'Boosting':
        labels = ['Beta based weights']
    elif title == 'Different sampling methods':
        labels = ['Bagging', 'No subset', 'Boosting']
    elif title == 'Trained with subsetting':
        labels = ['weighted_absolute', 'mean', 'weighted_ranked', 'n.a']

    for i, result in enumerate(results):
        test_acc = [r[1]['test_acc'] for r in result.items()]
        ensemble_size = [i for i in range(1, 26)]
        plt.plot(ensemble_size, test_acc, label=labels[i], linestyle=linestyles[i])
    plt.xlabel('ensemble size')
    plt.ylabel('test accuracy')
    plt.title('{}'.format(title))
    if title == 'Different sampling methods':
        plt.legend(title='sampling methods')
    else:
        plt.legend(title='Aggregation methods')
    plt.savefig('{}'.format(title))
    return


def plot_time_results(results):
    linestyles = ['-', '--', ':']
    colors = ['blue', 'red', 'green', 'red']
    labels = [['no subset weighted absolute', 'no subset mean', 'no subset weighted rank'],
              ['bagging weighted absolute', 'bagging mean', 'bagging weighted rank'],
              ['boosting']]
    fig = plt.figure()

    for i, result in enumerate(results):
        for j, res in enumerate(result):
            if i == 2 and j == 0:
                res = result
            if type(res) == dict:
                time = [r[1]['pred_time'] for r in res.items()]
                this_color = colors[i]
                this_linestyle = linestyles[j]
                ensemble_size = [i for i in range(1, 26)]
                plt.plot(ensemble_size, time, label=labels[i][j], linestyle=this_linestyle, color=this_color)
                if i == 2 and j == 0:
                    plt.xlabel('ensemble size')
                    plt.ylabel('prediction time')
                    plt.title('Prediction time for ensemble type')
                    plt.legend(title='ensemble type')
                    plt.savefig('pred_time')
                    return


def experiment(dir, X_train, X_test, method):
    # create a dictionairy to store the experiment results in
    experiment_results = dict()
    # import all models
    models = load_models(dir)

    # Create a list of all model names
    files = os.listdir(dir)
    model_names = [file for file in files if 'model' in file]
    # Construct the maximum number of ensembles. Starting with 1, up to max_n_ensembles
    for i, el in enumerate(model_names):
        # create a dictionairy to store experiment result in
        this_experiment_results = dict()
        # define the amount of models in this ensemble. Due to indexing starting at 0, we do index + 1
        sample_size = i + 1
        sample = random.sample(model_names, sample_size)
        print(sample_size)
        print(sample)
        print()
        # --------------------------------------------------------------------------------------------------------------
        # GET TEST RESULTS
        # --------------------------------------------------------------------------------------------------------------
        # start timing
        time0 = time.time()

        total_y_pred = np.empty((X_test.shape[0], 2), dtype=float)
        # loop over all models in the ensemble
        if method == 'no_subset_mean' or method == 'bagging_mean':
            for m in sample:
                total_y_pred += models[m]['model'].predict(X_test).reshape((total_y_pred.shape[0], 2))
        elif method == 'no_subset_weighted_rank' or method == 'bagging_weighted_rank':
            val_ranking = sorted([models[m]['val_acc'] for m in sample])
            for m in sample:
                val_acc = models[m]['val_acc']
                this_rank = val_ranking.index(val_acc) + 1
                this_weight = this_rank / sample_size
                total_y_pred += models[m]['model'].predict(X_test).reshape((total_y_pred.shape[0], 2)) * this_weight
        elif method == 'no_subset_weighted_absolute' or method == 'bagging_weighted_absolute':
            for m in sample:
                total_y_pred += models[m]['model'].predict(X_test).reshape((total_y_pred.shape[0], 2)) * models[m]['val_acc']
        elif method == 'boosting':
            sample = model_names[:sample_size]
            for m in sample:
                Beta = models[m]['Beta']
                total_y_pred += models[m]['model'].predict(X_test).reshape((total_y_pred.shape[0], 2)) * log(1 / Beta)


        # get average y_pred
        y_pred = total_y_pred / sample_size
        # reshape y_pred to be able to fit into check_accuracy
        y_pred = np.asarray([[1 if np.argmax(row) == i else 0 for i in range(2)] for row in y_pred]).reshape((total_y_pred.shape[0], 2))
        # check the time it took to make predictions
        pred_time = time.time() - time0
        # get test accuracy
        test_acc = check_accuracy(y_pred, y_test)

        this_experiment_results['pred_time'] = pred_time
        this_experiment_results['test_acc'] = test_acc

        # --------------------------------------------------------------------------------------------------------------
        # GET TRAIN RESULTS
        # --------------------------------------------------------------------------------------------------------------
        total_y_train = np.empty((X_train.shape[0], 2), dtype=float)
        # loop over all models in the ensemble
        if method == 'no_subset_mean':
            for m in sample:
                total_y_train += models[m]['model'].predict(X_train).reshape((total_y_train.shape[0], 2))
        elif method == 'no_subset_weighted_rank':
            val_ranking = sorted([models[m]['val_acc'] for m in sample])
            for m in sample:
                val_acc = models[m]['val_acc']
                this_rank = val_ranking.index(val_acc) + 1
                this_weight = this_rank / sample_size
                total_y_train += models[m]['model'].predict(X_train).reshape((total_y_train.shape[0], 2)) * this_weight
        elif method == 'no_subset_weighted_absolute':
            for m in sample:
                total_y_train += models[m]['model'].predict(X_train).reshape((total_y_train.shape[0], 2)) * models[m]['val_acc']
        elif method == 'boosting':
            sample = model_names[:sample_size]
            for m in sample:
                Beta = models[m]['Beta']
                total_y_train += models[m]['model'].predict(X_train).reshape((total_y_train.shape[0], 2)) * log(1 / Beta)

        # get average y_pred
        y_pred_train = total_y_train / sample_size
        # reshape y_pred to be able to fit into check_accuracy
        y_pred_train = np.asarray([[1 if np.argmax(row) == i else 0 for i in range(2)] for row in y_pred_train]).reshape((X_train.shape[0], 2))

        # get train accuracy
        train_acc = check_accuracy(y_pred_train, y_train)
        this_experiment_results['train_acc'] = train_acc

        experiment_results['{}'.format(sample_size)] = this_experiment_results

    # change directory
    os.chdir('D:/thijs/Github/chess position classification/results/')
    # and save experiment results
    with open('results-{}'.format(method), 'wb') as pickle_out:
        pickle.dump(experiment_results, pickle_out)

    return


def load_test_data():
    os.chdir(curdir + '/data/')

    with open('X_test.npy', 'rb') as pickle_in:
        X_test = pickle.load(pickle_in)

    with open('y_test.npy', 'rb') as pickle_in:
        y_test = pickle.load(pickle_in)

    return X_test, y_test


def train_boosted_models(X_train, y_train, validation_split, nr_models, epoch, path=(curdir + '/models/boosted/')):
    # change directory
    os.chdir(path)

    # Initialize probability distribution Dt
    Dt_dict = dict()
    Dt_dict['0'] = np.full((X_train.shape[0],), 1/X_train.shape[0])

    # train t models
    for i in range(nr_models):
        model = Sequential()
        model.add(Dense(130, input_shape=(769,), activation=relu))
        model.add(Dropout(0.1))
        model.add(Dense(130, activation=relu))
        model.add(Dropout(0.1))
        model.add(Dense(260, activation=relu))
        model.add(Dropout(0.1))
        model.add(Dense(260, activation=relu))
        model.add(Dropout(0.1))
        model.add(Dense(130, activation=relu))
        model.add(Dropout(0.1))
        model.add(Dense(130, activation=relu))
        model.add(Dropout(0.25))
        model.add(Dense(65, activation=relu))
        model.add(Dense(2, activation=softmax))

        Optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=Optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
        hist = model.fit(X_train, y_train,
                         epochs=epoch,
                         verbose=1,
                         batch_size=150,
                         sample_weight=Dt_dict['{}'.format(i)],
                         validation_split=validation_split)

        # predict training samples
        train_pred = model.predict(X_train)
        # transform predictions to the right format (shape of (2,))
        y_pred = np.asarray([[1 if np.argmax(row) == j else 0 for j in range(2)] for row in train_pred])
        # create an array containing booleans: each 1 represents a missclasification
        error_array = np.asarray([0 if np.array_equal(y_train[j], pred) else 1 for j, pred in enumerate(y_pred)])
        # compute error: error is the sum of distribution probabilities for misclassified samples.
        error = np.sum(np.asarray([Dt_dict['{}'.format(i)][j] if el == 1 else 0 for j, el in enumerate(error_array)]))

        if error > 0.5:
            print('error bigger than 0.5 at iteration {}, ABORTING'.format(i))
            return

        # set Beta to be error / 1 - error
        Beta = error / (1 - error)
        # update sampling distribution
        new_Dt = []
        for j, el in enumerate(error_array):
            old_weight = Dt_dict['{}'.format(i)][j]
            # if this sample is misclassified
            if el == 1:
                product = 1
            else:
                product = Beta
            # compute new weight
            new_weight = (old_weight) * product
            # add new weight to the final Dt+1 weight list
            new_Dt.append(new_weight)

        Zt = np.sum(new_Dt)
        Dt_dict['{}'.format(i + 1)] = new_Dt / Zt
        # print('sum of new probability distribution = ', (np.sum(new_Dt / Zt)))

        # save model in dictionairy format
        model_dic = dict()
        model_dic['json_model'] = model.to_json()
        model_dic['weights'] = model.get_weights()
        model_dic['Beta'] = Beta
        with open('model{}'.format(i), 'wb') as pickle_out:
            pickle.dump(model_dic, pickle_out)

    # save probability distribution Dt for all iterations in a dictionairy format
    with open('Dt_dict', 'wb') as pickle_out:
        pickle.dump(Dt_dict, pickle_out)

    return


def train_bagging_models(X_train, y_train, nr_models, validation_split, path=(curdir + '/models/bootstrapped'), sample_size=0.8):
    os.chdir(path)
    # train t models
    for i in range(nr_models):
        # Sample from train set
        n_samples = ceil(sample_size * X_train.shape[0])
        new_X_train = np.empty((n_samples, X_train.shape[1]), dtype='uint8')
        new_y_train = np.empty((n_samples, y_train.shape[1]), dtype='uint8')
        sample_index = [random.randint(0, n_samples) for n in range(n_samples)]

        for j, ind in enumerate(sample_index):
            new_X_train[j] = X_train[ind]
            new_y_train[j] = y_train[ind]

        model = Sequential()
        model.add(Dense(130, input_shape=(769,), activation=relu, kernel_initializer='random_uniform'))
        model.add(Dropout(0.1))
        model.add(Dense(130, activation=relu, kernel_initializer='random_uniform'))
        model.add(Dropout(0.1))
        model.add(Dense(260, activation=relu, kernel_initializer='random_uniform'))
        model.add(Dropout(0.1))
        model.add(Dense(260, activation=relu, kernel_initializer='random_uniform'))
        model.add(Dropout(0.1))
        model.add(Dense(130, activation=relu, kernel_initializer='random_uniform'))
        model.add(Dropout(0.1))
        model.add(Dense(130, activation=relu, kernel_initializer='random_uniform'))
        model.add(Dropout(0.25))
        model.add(Dense(65, activation=relu, kernel_initializer='random_uniform'))
        model.add(Dense(2, activation=softmax))

        Optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=Optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
        hist = model.fit(X_train, y_train,
                         epochs=15,
                         verbose=1,
                         batch_size=150,
                         validation_split=validation_split)

        model_dic = dict()
        model_dic['json_model'] = model.to_json()
        model_dic['weights'] = model.get_weights()
        model_dic['val_acc'] = hist.history['val_binary_accuracy'][-1]

        with open('model{}'.format(i), 'wb') as pickle_out:
            pickle.dump(model_dic, pickle_out)

    return


def train_models(X_train, y_train, nr_models, validation_split=0, path=(curdir + '/models/no_subsetting')):
    os.chdir(path)

    for i in range(nr_models):
        model = Sequential()
        model.add(Dense(130, input_shape=(769,), activation=relu, kernel_initializer='random_uniform'))
        model.add(Dropout(0.1))
        model.add(Dense(130, activation=relu, kernel_initializer='random_uniform'))
        # model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.1))
        model.add(Dense(260, activation=relu, kernel_initializer='random_uniform'))
        model.add(Dropout(0.1))
        model.add(Dense(260, activation=relu, kernel_initializer='random_uniform'))
        model.add(Dropout(0.1))
        model.add(Dense(130, activation=relu, kernel_initializer='random_uniform'))
        model.add(Dropout(0.1))
        model.add(Dense(130, activation=relu, kernel_initializer='random_uniform'))
        model.add(Dropout(0.25))
        model.add(Dense(65, activation=relu, kernel_initializer='random_uniform'))
        model.add(Dense(2, activation=softmax))
        # callback = EarlyStopping(monitor='val_loss', patience=5)
        Optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=Optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

        hist = model.fit(X_train, y_train, epochs=15, verbose=1, batch_size=150, validation_split=validation_split, shuffle=True)
        print('val accuracy = ', hist.history['val_binary_accuracy'])

        model_dic = dict()
        model_dic['json_model'] = model.to_json()
        model_dic['weights'] = model.get_weights()
        model_dic['val_acc'] = hist.history['val_binary_accuracy'][-1]

        with open('model{}'.format(i), 'wb') as pickle_out:
            pickle.dump(model_dic, pickle_out)
    return


# ----------------------------------------------------------------------------------------------------------------------
# GET DATA
# ----------------------------------------------------------------------------------------------------------------------
X_test, y_test = load_test_data()

os.chdir(curdir + '/data/')

with open('X_train.npy', 'rb') as pickle_in:
    X_train = pickle.load(pickle_in)
with open('y_train.npy', 'rb') as pickle_in:
    y_train = pickle.load(pickle_in)


# ----------------------------------------------------------------------------------------------------------------------
# DO EXPERIMENTS ADN STORE RESULTS
# ----------------------------------------------------------------------------------------------------------------------
# get results of boosted models
experiment(curdir + '/models/boosted/', X_train, X_test, method='boosting')
# get results of no subset models
experiment(curdir + '/models/no_subsetting', X_train, X_test, method='no_subset_mean')
experiment(curdir + '/models/no_subsetting', X_train, X_test, method='no_subset_weighted_rank')
experiment(curdir + '/models/no_subsetting', X_train, X_test, method='no_subset_weighted_absolute')
# get results of bootstrapped models
experiment(curdir + '/models/bootstrapped', X_train, X_test, method='bagging_mean')
experiment(curdir + '/models/bootstrapped', X_train, X_test, method='bagging_weighted_rank')
experiment(curdir + '/models/bootstrapped', X_train, X_test, method='bagging_weighted_absolute')


# ----------------------------------------------------------------------------------------------------------------------
# LOAD RESULTS
# ----------------------------------------------------------------------------------------------------------------------
os.chdir(curdir + '/results')

with open('results-no_subset_weighted_rank', 'rb') as pickle_in:
    no_subset_weighted_rank = pickle.load(pickle_in)
with open('results-no_subset_mean', 'rb') as pickle_in:
    no_subset_mean = pickle.load(pickle_in)
with open('results-no_subset_weighted_absolute', 'rb') as pickle_in:
    no_subset_weighted_absolute = pickle.load(pickle_in)
with open('results-bagging_weighted_rank', 'rb') as pickle_in:
    bagging_weighted_rank = pickle.load(pickle_in)
with open('results-bagging_mean', 'rb') as pickle_in:
    bagging_mean = pickle.load(pickle_in)
with open('results-bagging_weighted_absolute', 'rb') as pickle_in:
    bagging_weighted_absolute = pickle.load(pickle_in)
with open('results-boosting', 'rb') as pickle_in:
    boosted = pickle.load(pickle_in)


# ----------------------------------------------------------------------------------------------------------------------
# CREATE PLOTS AND SAVE THEM
# ----------------------------------------------------------------------------------------------------------------------
plot_results([no_subset_weighted_absolute, no_subset_mean, no_subset_weighted_rank], 'No Subsetting')
plot_results([bagging_weighted_absolute, bagging_mean, bagging_weighted_rank, boosted], 'Trained with subsetting')
plot_results([boosted], 'Boosting')
plot_results([bagging_weighted_absolute, no_subset_weighted_absolute, boosted], 'Different sampling methods')
plot_time_results([[no_subset_weighted_absolute, no_subset_mean, no_subset_weighted_rank],
                   [bagging_weighted_absolute, bagging_mean, bagging_weighted_rank],
                   boosted])