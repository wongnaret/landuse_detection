#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
 File name: accuracy.py
 Date Create: 14/9/2021 AD 18:34
 Author: Wongnaret Khantuwan 
 Email: wongnaet.khantuwan@nectec.or.th, wongnaret@gmail.com
 Python Version: 3.9
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from pycm import *
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
from numpy import sqrt

def accuracy_assesment(y_test, y_pred, labels=None):
    confusion = confusion_matrix(y_test, y_pred)
    print(confusion)

    #disp = ConfusionMatrixDisplay(confusion, display_labels=labels)
    #disp.plot()

    # importing accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

    print('Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    print('Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    print('F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='weighted')))

    from sklearn.metrics import classification_report
    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, target_names=labels))

    return confusion

def export_confusion_matrix(y_actu, y_pred, output_file, verbose=False):
    cm = ConfusionMatrix(y_actu, y_pred)
    cm.plot(cmap=plt.cm.Greens, number_label=True, plot_lib="matplotlib", normalized=True)
    # plt.show()
    plt.savefig(output_file.replace('.png', '_nomalrized.png'))

    cm.plot(cmap=plt.cm.Greens, number_label=True, plot_lib="matplotlib")
    # plt.show()
    plt.savefig(output_file)

    #print("ACC:", cm.Overall_ACC)
    #print("Precision", cm.OP[0], cm.OP[1])
    #print("Recall", cm.TPR[0], cm.TPR[1])

    #print("TPR:", cm.TPR[1])
    #print("FPR:", cm.FPR[1])
    #print("G-mean:", sqrt(cm.TPR[1] * (1 - cm.FPR[1])))

    if(verbose):
        print(cm)
