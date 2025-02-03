import tensorflow as tf
from tensorflow.keras.models import load_model,save_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten, Activation, GlobalAveragePooling2D,Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def define_model(width, height):
    input_tensor = Input(shape=(224, 224, 3))
    base_model = DenseNet201(input_tensor=input_tensor, weights='imagenet', include_top=False)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(128,activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64,activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(5, activation='softmax')
    ])
    return model

def define_generators():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=10,
    vertical_flip = True,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.0
    )

    train_generator = train_datagen.flow_from_directory(
        directory='./archive',
        target_size=(width, height),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode="categorical",
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        directory='./archive',
        target_size=(width, height),
        batch_size=batch_size,
        color_mode='rgb',
class_mode="categorical",
        subset='validation'
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
        directory='./archive',
        target_size=(width, height),
        color_mode='rgb',
        shuffle=False,
        class_mode='categorical')

    return train_generator, validation_generator, test_generator

nb_epoch     = 30
batch_size   = 32
width        = 224
height       = 224

model = define_model(width, height)
model.summary()
train_generator, validation_generator, test_generator = define_generators()


model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy'])


model.fit(
    train_generator,
    epochs=nb_epoch
)

model.evaluate(test_generator)

save_model(model, "./model_v1_ALL.h5")

model = load_model('./model_v1_ALL.h5')

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_generator,
    epochs=15
)

train_generator, validation_generator, test_generator = define_generators()
model = load_model('./kmodel_v1_ALL.h5')
model.evaluate(test_generator)

save_model(model, "./model_v1_ALL.h5")