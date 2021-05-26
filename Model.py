import keras
# from keras.utils.vis_utils import plot_model
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
import pydot
import graphviz

##################################################################################################

def My_Model(weights_path=None):
    input = Input(shape=(64, 64, 3))
    X = BatchNormalization()(input)
    X = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(X)
    X = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    X = BatchNormalization()(X)
    X = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(X)
    X = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    X = BatchNormalization()(X)
    X = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(X)
    X = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(X)
    X = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)

    X = BatchNormalization()(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(X)
    # X = Conv2D(filters=256 , kernel_size=(3,3) , activation ='relu' , padding='same')(X)
    # X = Conv2D(filters=256 , kernel_size=(3,3) , activation ='relu' , padding='same')(X)
    # X = MaxPooling2D(pool_size=(2,2))(X)
    # X = Flatten()(X)

    X = BatchNormalization()(X)
    X = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(X)
    X = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(X)
    # X = Conv2D(filters=512 , kernel_size=(3,3) , activation ='relu' , padding='same')(X)
    # X = Conv2D(filters=256 , kernel_size=(3,3) , activation ='relu' , padding='same')(X)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = Flatten()(X)

    pred_gender = Dense(2, activation='softmax')(X)
    pred_age = Dense(117, activation='softmax')(X)
    pred_race = Dense(5, activation='softmax')(X)

    model = Model(inputs=input, outputs=[pred_gender, pred_age, pred_race])

    if weights_path is not None:
        model.load_weights(weights_path)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                  loss=["categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy"],
                  metrics=['accuracy'])

    # plot_model(model, to_file='model.png')

    print('Model Created')
    print(model.summary())
    return model
