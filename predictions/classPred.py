import sys
import argparse
import os
from predictions.utils import loader, processor
from predictions.net import classifier


class Pred:
    def __init__(self, model_path, data_path):
        self.base_path = os.path.dirname(os.path.realpath(__file__))
        self.data_path = os.path.join(self.base_path, '..\\data\\') + data_path
        self.model_path = os.path.join(self.base_path, '..\\model\\') + model_path
        self.coords = 3
        self.joints = 16
        self.cycles = 1
        self.num_classes = 4
        self.device = 'cuda:0'
        self.test_size = 0.1
        self.graph_dict = {'strategy': 'spatial'}
        self.model = classifier.Classifier(3, 4, self.graph_dict)
        self.args = self.init_args()
        self.pr = self.generate_processor()
        self.emotions = ['Angry', 'Neutral', 'Happy', 'Sad']

    def generate_processor(self):
        pr = processor.Processor(self.args, None, self.coords, self.num_classes, self.graph_dict, device=self.device, verbose=False)
        pr.load_best_model(self.model_path)
        pr.model.eval()
        return pr

    #def load_model(self):
        #self.model.load_state_dict(torch.load(self.model_path))
        #self.model.apply(weights_init)
        #self.model.cuda(torch.device("cuda:0"))
        #self.model.eval()

    def generate_data(self):
        data = loader.load_data(self.data_path, self.coords, self.joints, cycles=1)
        return data

    def generate_predictions(self, data):
        labels_pred = self.pr.generate_predictions(data, self.num_classes,self.joints, self.coords)
        labels = []
        for idx in range(labels_pred.shape[0]):
            #print('{:d}.{:s}'.format(idx, self.emotions[int(labels_pred[idx])]))
            labels.append(self.emotions[int(labels_pred[idx])])
        return labels

    def init_args(self):
        parser = argparse.ArgumentParser(description='Gait Gen')

        parser.add_argument('--train', type=bool, default=False, metavar='T',
                            help='train the model (default: True)')
        parser.add_argument('--smap', type=bool, default=False, metavar='S',
                            help='train the model (default: True)')
        parser.add_argument('--save-features', type=bool, default=False, metavar='SF',
                            help='save penultimate layer features (default: True)')
        parser.add_argument('--batch-size', type=int, default=6, metavar='B',
                            help='input batch size for training (default: 6)')
        parser.add_argument('--num-worker', type=int, default=4, metavar='W',
                            help='input batch size for training (default: 4)')
        parser.add_argument('--start_epoch', type=int, default=0, metavar='SE',
                            help='starting epoch of training (default: 0)')
        parser.add_argument('--num_epoch', type=int, default=100, metavar='NE',
                            help='number of epochs to train (default: 500)')
        parser.add_argument('--optimizer', type=str, default='Adam', metavar='O',
                            help='optimizer (default: Adam)')
        parser.add_argument('--base-lr', type=float, default=0.1, metavar='L',
                            help='base learning rate (default: 0.1)')
        parser.add_argument('--step', type=list, default=[0.5, 0.75, 0.875], metavar='[S]',
                            help='fraction of steps when learning rate will be decreased (default: [0.5, 0.75, 0.875])')
        parser.add_argument('--nesterov', action='store_true', default=True,
                            help='use nesterov')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                            help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='D',
                            help='Weight decay (default: 5e-4)')
        parser.add_argument('--eval-interval', type=int, default=1, metavar='EI',
                            help='interval after which model is evaluated (default: 1)')
        parser.add_argument('--log-interval', type=int, default=100, metavar='LI',
                            help='interval after which log is printed (default: 100)')
        parser.add_argument('--topk', type=list, default=[1], metavar='[K]',
                            help='top K accuracy to show (default: [1])')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--pavi-log', action='store_true', default=False,
                            help='pavi log')
        parser.add_argument('--print-log', action='store_true', default=False,
                            help='print log')
        parser.add_argument('--save-log', action='store_true', default=False,
                            help='save log')
        parser.add_argument('--work-dir', type=str, default=self.model_path, metavar='WD',
                            help='path to save')
        args = parser.parse_args()
        return args

