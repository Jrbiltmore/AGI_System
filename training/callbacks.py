
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomCallback(Callback):
    def __init__(self, patience=5):
        super(CustomCallback, self).__init__()
        self.patience = patience
        self.best_weights = None
        self.wait = 0
        self.best = float('inf')

    def on_train_begin(self, logs=None):
        self.best = float('inf')
        self.wait = 0
        self.best_weights = self.model.get_weights()
        logging.info("Training started. Best weights initialized.")

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss')
        if current_loss < self.best:
            self.best = current_loss
            self.wait = 0
            self.best_weights = self.model.get_weights()
            logging.info(f"Improvement in validation loss at epoch {epoch}.")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
                logging.info(f"No improvement for {self.patience} epochs. Stopping training.")

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)
        logging.info("Training ended. Best weights restored.")

class LearningRateScheduler(Callback):
    def __init__(self, schedule):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            scheduled_lr = self.schedule(epoch, lr)
        except Exception as e:
            logging.error(f"Error getting scheduled learning rate: {str(e)}")
            raise

        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        logging.info(f"Epoch {epoch}: Learning rate set to {scheduled_lr}.")

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    def schedule(epoch, lr):
        new_lr = initial_lr * (decay_factor ** np.floor(epoch / step_size))
        return new_lr
    return schedule

def main():
    scheduler = LearningRateScheduler(step_decay_schedule())
    logging.info("Custom learning rate scheduler initialized.")

if __name__ == "__main__":
    main()
