import argparse
from utils.utils import *
import seaborn as sns
import matplotlib.pyplot as plt

def eval(args):
    logger = get_logger('Evaluate F3')
    logger.propagate = False
    PREDICATION_PATH = args.predication_path
    cut_index = PREDICATION_PATH.rfind('/')
    LOG_PATH = PREDICATION_PATH[:cut_index]
    PREDICT_NAME = PREDICATION_PATH[cut_index + 1:-4]
    logger.info(f'{PREDICT_NAME}')
    CLASSES = args.classes
    test1_labels = np.load(args.test1_labels_path)
    test2_labels = np.load(args.test2_labels_path)
    train_labels = np.load(args.train_labels_path)
    # xyz->zxy
    labels_data = np.concatenate([np.concatenate([test1_labels, train_labels], axis=0), test2_labels], axis=1).transpose(2, 1, 0)

    y_list = list(range(labels_data.shape[2]))
    ex_list = [0,100,200,300,400,500,600]
    for e in ex_list:
        y_list.remove(e)

    predication = np.load(PREDICATION_PATH)['prediction']
    labels_data = labels_data[:, :, y_list]
    predication = predication[:,:,y_list]
    eval = runningScore(CLASSES)
    eval.update(labels_data,predication)
    eval_score, eval_class_iou = eval.get_scores()

    xticks = ['Upper N.S.', 'Middle N.S.', 'Low N.S.', 'Rijnland/Chalk', 'Scruff', 'Zechstein']
    yticks = ['Upper N.S.', 'Middle N.S.', 'Low N.S.', 'Rijnland/Chalk', 'Scruff', 'Zechstein']
    plt.figure(figsize=(9, 9), dpi=300)
    ax = sns.heatmap(eval.confusion_matrix/eval.confusion_matrix.sum(axis=1),
                     annot=True,
                     fmt=".3f",
                     linewidths=2,
                     square=True,
                     xticklabels=xticks,
                     yticklabels=yticks,annot_kws={"fontsize":13})
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    ax.set_title('Class Accuracy', family='Arial')
    plt.tight_layout()
    plt.savefig(LOG_PATH+'/heatmap.png', dpi=300)

    eval_txt = open(LOG_PATH+'/eval_score.txt','w')
    for key in eval_score.keys():
        eval_txt.write(key+':'+str(eval_score[key])+'\n')
        logger.info(f'{key} {eval_score[key]}')
    eval_txt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval')
    parser.add_argument('--predication_path', nargs='?', type=str,
                        default='/volume/seis_facies_ident/runs/Oct12_091353_f3_semi_sup_sparse/best_model_prediction_f3.npz',
                        help='Predication path')
    parser.add_argument('--train_data_path', nargs='?', type=str,
                        default='/volume/dataset/f3/train/train_seismic.npy',
                        help='Train data path')
    parser.add_argument('--train_labels_path', nargs='?', type=str,
                        default='/volume/dataset/f3/train/train_labels.npy',
                        help='Train labels path')
    parser.add_argument('--test1_data_path', nargs='?', type=str,
                        default='/volume/dataset/f3/test_once/test1_seismic.npy',
                        help='Test1 data path')
    parser.add_argument('--test1_labels_path', nargs='?', type=str,
                        default='/volume/dataset/f3/test_once/test1_labels.npy',
                        help='Test1 labels path')
    parser.add_argument('--test2_data_path', nargs='?', type=str,
                        default='/volume/dataset/f3/test_once/test2_seismic.npy',
                        help='Test2 data path')
    parser.add_argument('--test2_labels_path', nargs='?', type=str,
                        default='/volume/dataset/f3/test_once/test2_labels.npy',
                        help='Test2 labels path')
    parser.add_argument('--classes', nargs='?', type=int, default=6,
                        help='Classes')
    args = parser.parse_args()
    eval(args)