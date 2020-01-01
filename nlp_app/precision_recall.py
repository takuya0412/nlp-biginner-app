from sklearn.metrics import precision_score, recall_score


y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]

print('Precision of each class:{}'.format(precision_score(y_true, y_pred, average=None)))
print('Averaged precision: {}'.format(precision_score(y_true, y_pred, average='macro')))
print('Recall of each class: {}'.format(recall_score(y_true, y_pred, average=None)))
print('Averaged reacall: {}'.format(recall_score(y_true, y_pred, average='macro')))
