
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class PostProcessor:
    def __init__(self):
        pass

    def compute_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        report = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm
        }
        return metrics

    def plot_confusion_matrix(self, cm, class_names):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_roc_curve(self, y_true, y_score, class_names):
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        from itertools import cycle

        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        n_classes = y_true_bin.shape[1]
        
        plt.figure()
        for i, color in zip(range(n_classes), cycle(['aqua', 'darkorange', 'cornflowerblue'])):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'ROC curve of class {class_names[i]} (area = {roc_auc:0.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

    def save_results(self, metrics, file_path):
        with open(file_path, 'w') as f:
            for key, value in metrics.items():
                f.write(f'{key}: {value}
')

    def load_results(self, file_path):
        with open(file_path, 'r') as f:
            results = f.read()
        return results
