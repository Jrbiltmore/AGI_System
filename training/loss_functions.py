
import tensorflow as tf
from tensorflow.keras import backend as K

class CustomLossFunctions:
    @staticmethod
    def focal_loss(gamma=2.0, alpha=0.25):
        def focal_loss_fixed(y_true, y_pred):
            y_true = tf.convert_to_tensor(y_true, tf.float32)
            y_pred = tf.convert_to_tensor(y_pred, tf.float32)

            alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
            p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
            fl = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
            return K.mean(fl, axis=-1)
        
        return focal_loss_fixed

    @staticmethod
    def dice_loss(smooth=1):
        def dice_loss_fixed(y_true, y_pred):
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        
        return dice_loss_fixed

    @staticmethod
    def tversky_loss(alpha=0.5, beta=0.5):
        def tversky_loss_fixed(y_true, y_pred):
            y_true_pos = K.flatten(y_true)
            y_pred_pos = K.flatten(y_pred)
            true_pos = K.sum(y_true_pos * y_pred_pos)
            false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
            false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
            return 1 - (true_pos + 1) / (true_pos + alpha * false_neg + beta * false_pos + 1)
        
        return tversky_loss_fixed

def main():
    # Example usage
    loss_function = CustomLossFunctions.focal_loss(gamma=2.0, alpha=0.25)
    print("Custom loss function initialized: Focal Loss")

if __name__ == "__main__":
    main()
