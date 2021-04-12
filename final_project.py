import os
import cv2
import glob
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense  # , Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from tensorflow import keras

# choose a number of time steps
time_steps = 20
n_features = 1
stride = 1


# create train
def split_sequence(inner_img, steps, mask, skip=1):
    X, y = [], []
    for row in range(inner_img.shape[0]):
        sequence = inner_img[row, :]
        for i in range(skip * steps, len(sequence), skip):
            curr_mask, curr_y = mask[row][i - skip * steps:i:skip], mask[row][i]
            if (0 not in curr_mask) and (curr_y != 0):  # data don't contain black pixels
                seq_x, seq_y = sequence[i - skip * steps:i:skip], sequence[i]
                X.append(seq_x)
                y.append(seq_y)
    return np.array(X), np.array(y)


# create test
def split_sequence_test(inner_img, steps, mask, skip=1):
    X_test_, y_pos = [], []
    for row in range(inner_img.shape[0]):
        sequence = inner_img[row, :]
        for i in range(skip * steps, len(sequence), skip):
            curr_mask, curr_y = mask[row][i - skip * steps:i:skip], mask[row][i]
            if (0 not in curr_mask) and (curr_y == 0):  # only first y=0
                seq_x = sequence[i - skip * steps:i:skip]
                X_test_.append(seq_x)
                y_pos.append((row, i))
    return np.array(X_test_), np.array(y_pos)


# show side-by-side images
def show_images(dmg, org=None):
    if org is None:
        cv2.imshow('', dmg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        numpy_horizontal = np.hstack((dmg, org))
        cv2.imshow('', numpy_horizontal)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
# https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
# https://keras.io/api/layers/convolution_layers/convolution1d/
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
def create_cnn_lstm_model(X_train_, y_train_, name_='temp'):
    model_ = Sequential()
    # define CNN model
    model_.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(time_steps, 1)))
    model_.add(MaxPooling1D(pool_size=2))
    # define LSTM model
    model_.add(LSTM(100, activation="relu", recurrent_activation="sigmoid"))
    model_.add(Dense(56, activation='sigmoid'))
    model_.add(Dense(1, activation='sigmoid'))
    model_.compile(optimizer='adam', loss='mse')
    model_.fit(X_train_, y_train_, epochs=10, verbose=1)
    # model_.save(name_)
    return model_


def predict_model(model_, in_img, mask_, time_steps_, stride_):
    inner_img = np.copy(in_img)
    X_test, y_test_poses = split_sequence_test(inner_img, time_steps_, mask_, stride_)
    while len(y_test_poses) > 0:
        num_of_samples_ = X_test.shape[0]
        x_shaped_t = np.reshape(X_test, newshape=(num_of_samples_, time_steps_, n_features))
        pred_y = model_.predict(x_shaped_t, verbose=1)
        for i, img_ in enumerate(y_test_poses):
            inner_img[img_[0], img_[1]] = pred_y[i]
            mask_[img_[0], img_[1]] = 1
        X_test, y_test_poses = split_sequence_test(inner_img, time_steps_, mask_, stride_)
    return inner_img


def save_img(img_, name_):
    img2 = (img_ * 255).astype(int)
    img1 = np.stack([img2, img2, img2], axis=2)
    cv2.imwrite(name_ + '.png', img1)


def rotate_img(in_img, pos):
    return np.rot90(in_img, pos)


def prep_img(in_img, pos):
    img_ = rotate_img(in_img, pos)
    mask_ = 1 - np.array(img_ == 0).astype(int)
    X_train_, y_train_ = split_sequence(img_, time_steps, mask_, 1)
    num_of_samples = X_train_.shape[0]
    x_shaped = np.reshape(X_train_, newshape=(num_of_samples, time_steps, n_features))
    return x_shaped, y_train_, mask_, img_


# cnn
def cnn_model_img(n, in_img, org_img_):
    pred_imgs = []
    for i in range(4):
        print('\nimage', n, 'iter', i, ':')
        x_shaped, y_train_, mask_, img_ = prep_img(in_img, i)
        model = create_cnn_lstm_model(x_shaped, y_train_, 'model_' + str(n) + '_' + str(i))
        # model = keras.models.load_model('model_' + str(i))
        pred_img = predict_model(model, img_, mask_, time_steps, 1)
        pred_img = rotate_img(pred_img, 4 - i)
        # save_img(pred_img, 'img_' + str(n) + '_' + str(i))
        pred_imgs.append(pred_img)
        # show_images(lr_cnn_img, org_img)

    print("combined", n)
    comb_img = np.average(pred_imgs, axis=0)
    # comb_img2 = np.sum(pred_imgs, axis=0) / len(pred_imgs)
    save_img(comb_img, 'comb_' + str(n))
    # show_images(comb_img, org_img_)


if __name__ == '__main__':
    # paths
    dmg_path = r'.\Damaged Gray small'
    org_path = r'.\ORG Gray small'

    # read images
    dmg_imgs = [cv2.imread(file)[:, :, 0] for file in glob.glob(os.path.join(dmg_path, "*.png"))]
    org_imgs = [cv2.imread(file)[:, :, 0] for file in glob.glob(os.path.join(org_path, "*.png"))]

    # first image
    # img = np.array(dmg_imgs[0])

    for i in range(len(dmg_imgs)):
        # resize image
        resize_img = 1
        img = np.array(dmg_imgs[i])
        img = img[::resize_img, ::resize_img]
        img = img / 255.
        org_img = np.array(org_imgs[i]) / 255.
        org_img = org_img[::resize_img, ::resize_img]

        cnn_model_img(i, img, org_img)
