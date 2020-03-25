import argparse
import numpy as np
import sys
sys.path.append("./learning")
from metrics import *

parser = argparse.ArgumentParser(description='Evaluation function for LPMT')

parser.add_argument('--odir', default='./results', help='Directory to store results')

args = parser.parse_args()

C = ConfusionMatrix
C.number_of_labels = 13
C.confusion_matrix=np.zeros((C.number_of_labels, C.number_of_labels))

class_map = {0:'ceiling', 1:'floor', 2:'terrain', 3:'column', 4:'beam', 5:'window', 6:'door', 7:'table', 8:'debris', 9:'bookcase', 10:'sofa', 11:'board', 12:'clutter'}

C.confusion_matrix+=np.load(args.odir+'/pointwise_cm.npy')  
    
print("\nOverall accuracy : %3.2f %%" % (100 * np.mean(ConfusionMatrix.get_overall_accuracy(C))))
print("Mean accuracy    : %3.2f %%" % (100 * np.mean(ConfusionMatrix.get_mean_class_accuracy(C))))
print("Mean IoU         : %3.2f %%\n" % (100 * np.mean(ConfusionMatrix.get_intersection_union_per_class(C))))
print("     Classe :  mIoU")
for c in range(0,C.number_of_labels):
    print ("   %8s : %6.2f %%" %(class_map[c],100*ConfusionMatrix.get_intersection_union_per_class(C)[c]))
