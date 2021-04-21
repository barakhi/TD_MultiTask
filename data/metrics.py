
import numpy as np


class RunningMetric(object):
    def __init__(self, metric_type, n_classes =None, binary=False):
        self._metric_type = metric_type
        self._binary = binary

        if metric_type == 'ACC':
            self.accuracy = 0.0
            self._n_classes = n_classes
            self.num_updates = 0.0
            self.confusion_matrix = np.zeros((n_classes, n_classes))


    def reset(self):

        if self._metric_type == 'ACC':
            self.accuracy = 0.0
            self.num_updates = 0.0
            self.confusion_matrix = np.zeros((self._n_classes, self._n_classes))


    def _fast_hist(self, pred, gt):
        mask = (gt >= 0) & (gt < self._n_classes)
        hist = np.bincount(
            self._n_classes * gt[mask].astype(int) +
            pred[mask], minlength=self._n_classes**2).reshape(self._n_classes, self._n_classes)
        return hist

    def update(self, pred, gt):

        if self._metric_type == 'ACC':
            predictions = pred.data.max(1, keepdim=True)[1]
            self.accuracy += (predictions.eq(gt.data.view_as(predictions)).cpu().sum())
            if self._binary:
                self.confusion_matrix += self._fast_hist(predictions.view_as(gt).cpu().numpy(), gt.cpu().numpy())
            self.num_updates += predictions.shape[0]
    

        
    def get_result(self):

        if self._metric_type == 'ACC':
            try:
                return { 'num': self.num_updates, 'acc': self.accuracy.float()/self.num_updates}
            except:
                print(self.num_updates)
                print(self.accuracy)
                return {'num': self.num_updates, 'acc': float(self.accuracy) / (self.num_updates+0.000001)}



def get_metrics(params):
    met = {}
    if 'rightof' in params['dataset']:
        for t in range(10):
            met[t] = RunningMetric(metric_type='ACC', n_classes=10)
            met['tsk'] = RunningMetric(metric_type='ACC', n_classes=10)
    if 'mnist' in params['dataset']:
        for t in params['tasks']:
            met[t] = RunningMetric(metric_type = 'ACC', n_classes=10)
            met['tsk'] = RunningMetric(metric_type='ACC', n_classes=10)
    if 'clevr' in params['dataset']:
        for t in params['tasks']:
            met[t] = RunningMetric(metric_type='ACC', n_classes=8)
    return met
