import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.layers import Input, Concatenate, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal

set_random_seed(42)
warnings.filterwarnings('ignore')
initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=42)

wave_shape = 256
input_shape_spec = (256,1)
num_classes = 9

def SPEC_CNN():

    input_shape = input_shape_spec
    inputs = Input(shape=input_shape)
    x = Dense(wave_shape,kernel_regularizer=tf.keras.regularizers.L1(0.01), kernel_initializer=initializer)(inputs)
    # Block 1
    x = Conv1D(32, 3, strides=2, padding='same', activation='relu')(inputs)
    x = Conv1D(32, 3, strides=1, padding='same', activation='relu')(x)
    x = Conv1D(32, 3, strides=1, padding='same', activation='relu')(x)
    skip1 = MaxPooling1D(pool_size=2, strides=1)(x)
    x = Dropout(0.25)(skip1)

    # Block 2
    x = Conv1D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv1D(64, 3, strides=1, padding='same', activation='relu')(x)
    x = Conv1D(64, 3, strides=1, padding='same', activation='relu')(x)
    skip2 = MaxPooling1D(pool_size=2, strides=1)(x)
    x = Dropout(0.25)(skip2)

    # Block 3
    x = Conv1D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv1D(128, 3, strides=1, padding='same', activation='relu')(x)
    x = Conv1D(128, 3, strides=1, padding='same', activation='relu')(x)
    skip3 = MaxPooling1D(pool_size=2, strides=1)(x)
    x = Dropout(0.25)(skip3)

    # Block 4
    x = Conv1D(256, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv1D(256, 3, strides=1, padding='same', activation='relu')(x)
    x = Conv1D(256, 3, strides=1, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2, strides=1)(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Concatenate()([x, Flatten()(skip1)])
    x = Concatenate()([x, Flatten()(skip2)])
    x = Concatenate()([x, Flatten()(skip3)])

    x = Dense(512, activation='linear')(x)
    x = Dropout(0.25)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
