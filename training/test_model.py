from detectors import DETECTOR
import torch
from torch.utils.data import DataLoader
import numpy as np
from log_utils import Logger
import torch.backends.cudnn as cudnn
from dataset.datasets_train import ImageDataset_Test
import csv
import argparse
from tqdm import tqdm
import time
import os


from transform import get_albumentations_transforms

parser = argparse.ArgumentParser("Example")

parser.add_argument('--test_batchsize', type=int, default=32, help="test batch size")
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--datapath', type=str,
                    default='../dataset/')
parser.add_argument("--checkpoints", type=str, default='',
                    help="continue train model path")
parser.add_argument("--model", type=str, default='xception',
                    help="detector name[xception, fair_df_detector,daw_fdd, efficientnet,...]")

#################################test##############################
# in intersectional attribute skintone1 represent Light (tone1-3)
parser.add_argument("--inter_attribute", type=str,
                    default='nomale,skintone1-nomale,skintone2-nomale,skintone3-male,skintone1-male,skintone2-male,skintone3-child-young-adult-middle-senior')
parser.add_argument("--test_datapath", type=str,
                        default='../dataset/test.csv', help="test data path")
parser.add_argument("--savepath", type=str,
                        default='../results')

args = parser.parse_args()

if args.model=='srm' or args.model=='core':
    from fairness_metrics_srm import acc_fairness_softmax
else:
    from fairness_metrics import acc_fairness

###### import data transform #######
from transform import default_data_transforms as data_transforms
test_transforms = get_albumentations_transforms([''])


device = torch.device(args.device)

# prepare the model (detector)
model_class = DETECTOR[args.model]


def cleanup_npy_files(directory):
    """
    Deletes all .npy files in the given directory.
    :param directory: The directory to clean up .npy files in.
    """
    for item in os.listdir(directory):
        if item.endswith(".npy"):
            os.remove(os.path.join(directory, item))
    print("Cleaned up .npy files in directory:", directory)


# evaluation
def evaluate(model):
    interattributes = args.inter_attribute.split('-')
    model.eval()

    for eachatt in interattributes:
        test_dataset = ImageDataset_Test(args.test_datapath, eachatt, test_transforms)

        test_dataloader = DataLoader(
            test_dataset, batch_size=args.test_batchsize,
            shuffle=False,num_workers=32, pin_memory=True)

        print('Testing: ', eachatt)
        print('-' * 10)

        pred_list = []
        label_list = []

        for idx, data_dict in enumerate(tqdm(test_dataloader)):
            imgs, labels = data_dict['image'], data_dict['label']
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')  # remove the specific label

            data_dict['image'], data_dict['label'] = imgs.to(
                device), labels.to(device)
                
            with torch.no_grad():
                output = model(data_dict, inference=True)
                pred = output['cls']
                pred = pred.cpu().data.numpy().tolist()
    
                pred_list += pred
                label_list += labels.cpu().data.numpy().tolist()

        label_list = np.array(label_list)
        pred_list = np.array(pred_list)
        savepath = args.savepath + '/' + eachatt
        np.save(savepath+'labels.npy', label_list)
        np.save(savepath+'predictions.npy', pred_list)

        print()
        
    acc_fairness(args.savepath + '/', [['nomale', 'male'],
                                       ['skintone1', 'skintone2', 'skintone3'],
                                       ['child', 'young', 'adult','middle','senior']])
    cleanup_npy_files(args.savepath)

    return


def main():
    torch.manual_seed(args.seed)
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    model = model_class()
    model.to(device)

    state_dict = torch.load(args.checkpoints)
    model.load_state_dict(state_dict)

    evaluate(model)


if __name__ == '__main__':
    main()
