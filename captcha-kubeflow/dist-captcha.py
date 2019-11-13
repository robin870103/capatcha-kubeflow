import string,os,cv2,numpy as np
import argparse,random
import sys

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.keras.optimizers import *

FLAGS = None

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

class CaptchaSequence(Sequence):
    def __init__(self, characters, batch_size, steps, img_dir_list, local_path, n_len=4, width=128, height=64,
                 input_length=16, label_length=4):
        self.characters = characters
        self.batch_size = batch_size
        self.steps = steps
        self.n_len = n_len
        self.width = width
        self.height = height
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        self.img_dir_list = img_dir_list
        self.local_path = local_path

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        y = np.zeros((self.batch_size, self.n_len), dtype=np.uint8)
        input_length = np.ones(self.batch_size)*self.input_length
        label_length = np.ones(self.batch_size)*self.label_length
        for i in range(self.batch_size):
            ramdom_img = random.choice(self.img_dir_list)
            random_str = ramdom_img[0:4]
            X[i] =  cv2.resize(cv2.imread(self.local_path + ramdom_img),(self.width,self.height))/ 255.0
            y[i] = [self.characters.find(x) for x in random_str]
        return [X, y, input_length, label_length], np.ones(self.batch_size)


def train():
    characters = string.digits + string.ascii_uppercase + string.ascii_lowercase
    width, height, n_len, n_class = 128, 64, 4, len(characters) + 1


    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)

    input_tensor = Input((height, width, 3))
    x = input_tensor
    for i, n_cnn in enumerate([2, 2]):
        for j in range(n_cnn):
            x = Conv2D(32*2**min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        x = MaxPooling2D(2 if i < 3 else (2, 1))(x)

    x = Permute((2, 1, 3))(x)
    x = TimeDistributed(Flatten())(x)

    rnn_size = FLAGS.rnn_size
    x = Bidirectional(SimpleRNN(rnn_size, return_sequences=True))(x)
    x = Bidirectional(SimpleRNN(rnn_size, return_sequences=True))(x)
    x = Dense(n_class, activation='softmax')(x)

    base_model = Model(inputs=input_tensor, outputs=x)
    base_model.summary()

    labels = Input(name='the_labels', shape=[n_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=loss_out)

    def evaluate(model, batch_size=128, steps=20):
        batch_acc = 0
        local_path = FLAGS.data_dir
        img_list = []
        for img in os.listdir(local_path):
            img_list.append(img)

        valid_data = CaptchaSequence(characters, batch_size, steps,local_path = local_path, img_dir_list=img_list)
        for [X_test, y_test, _, _], _ in valid_data:
            y_pred = base_model.predict(X_test)
            shape = y_pred.shape
            out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(shape[0])*shape[1])[0][0])[:, :4]
            if out.shape[1] == 4:
                batch_acc += (y_test == out).all(axis=1).mean()
        return batch_acc / steps

    class Evaluate(Callback):
        def __init__(self):
            self.accs = []

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            acc = evaluate(base_model)
            logs['val_acc'] = acc
            self.accs.append(acc)

    local_path = FLAGS.data_dir
    img_list = []
    for img in os.listdir(local_path):
        img_list.append(img)

    train_data = CaptchaSequence(characters, batch_size=128, steps=100, local_path=local_path, img_dir_list=img_list)
    valid_data = CaptchaSequence(characters, batch_size=128, steps=10, local_path=local_path, img_dir_list=img_list)
    callbacks = [EarlyStopping(patience=5), Evaluate(),
                 CSVLogger('captcha3.csv'), ModelCheckpoint('captcha3-cpu_best.h5', save_best_only=True)]

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(1e-3, amsgrad=True))
    model.fit_generator(train_data, epochs=36, validation_data=valid_data, workers=4, use_multiprocessing=True,
                        callbacks=callbacks)
    model.save(FLAGS.save_model_path)

def main(_):
    train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnn_size",type =int, default = 128,
                        help = "RNN Size")
    parser.add_argument("--data_dir",type=str,
                        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                             'tensorflow/captcha/input_data'),
                        help = "directory for inputting data")
    parser.add_argument("--log_dir",type= str,
                        default = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),'tensorflow/captcha/logs/captcha'),
                        help = "Summaries log directory")

    parser.add_argument("--save_model_path",type= str,
                        default = os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),'tensorflow/captcha/model'),
                        help = "directory for saving model")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
