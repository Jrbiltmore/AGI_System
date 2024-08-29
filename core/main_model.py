
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import HeNormal

class AGIModel:
    def __init__(self, input_shape, output_shape, learning_rate=1e-4):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=self.input_shape)
        
        # First Dense Block
        x = Dense(512, kernel_initializer=HeNormal())(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        # Second Dense Block
        x = Dense(256, kernel_initializer=HeNormal())(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        # Third Dense Block
        x = Dense(128, kernel_initializer=HeNormal())(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        # Output Layer
        outputs = Dense(self.output_shape, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=100):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping, model_checkpoint, reduce_lr]
        )

        return history

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def summary(self):
        return self.model.summary()

    def plot_model(self, filename):
        plot_model(self.model, to_file=filename, show_shapes=True, show_layer_names=True)

