import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def calculate_metrics(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'accuracy': np.mean(y_true == y_pred)
    }

def save_metrics(metrics, filepath):
    # Save confusion matrix
    np.save(f"{filepath}_cm.npy", metrics['confusion_matrix'])
    
    # Save classification report
    pd.DataFrame(metrics['classification_report']).transpose().to_csv(f"{filepath}_report.csv")