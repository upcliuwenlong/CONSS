import argparse
import cv2
from utils.utils import *
import warnings
import os
import torch.nn.functional as F
warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def predict(args):
    logger = get_logger('Predict SEAM AI')
    logger.propagate = False
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ClASSES = args.classes
    SLIDE_WINDOW = args.slide_window
    SLICE_WIDTH = args.slice_width
    MODEL_PATH = args.model_path
    cut_index = MODEL_PATH.rfind('/')
    LOG_PATH = MODEL_PATH[:cut_index]
    MODEL_NAME = MODEL_PATH[cut_index+1:-4]
    logger.info(f'Model {MODEL_NAME}')
    logger.info(f'Loading data...')
    train_data = np.load(args.train_data_path)['data']
    # label：1-6 -> 0-5
    labels_data = np.load(args.labels_data_path)['labels'] - 1
    all_data = train_data
    val_pos = 0
    all_labels_data = np.ones_like(all_data) * -1

    # label range is [:,:val_pos,:] other -1
    all_labels_data[:, :val_pos, :] = labels_data[:, :val_pos, :]
    # czxy
    shape = tuple([ClASSES] + list(all_labels_data.shape))
    prob_labels_data = np.zeros(shape)
    # one-hot
    prob_labels_data[:, :, :val_pos, :] = np.stack([labels_data[:, :val_pos, :] == cl for cl in range(ClASSES)],
                                                                      axis=0).astype(np.float32)

    logger.info(f'Data loaded...')
    # load model:
    model = torch.load(MODEL_PATH)
    total = sum([param.nelement() for param in model.parameters()])
    logger.info(f"Model total parameters {total}")

    # Send to GPU if available
    logger.info(f"Sending the model to {DEVICE}")
    model = model.to(DEVICE)

    explt_origin_x = val_pos
    explt_x_length = all_labels_data.shape[1] - explt_origin_x
    logger.info(f'Predict start...')

    SHOW_IMG_FLAG = True
    for step in range(int(explt_x_length / SLIDE_WINDOW)):
        input_x_start = explt_origin_x + SLIDE_WINDOW * step
        # end positon not more than shape[1]
        input_x_end = min(input_x_start + SLICE_WIDTH, all_labels_data.shape[1])
        explt_x_start = input_x_start
        for y in range(labels_data.shape[2]):
            logger.info(f'step: {str(step)}, y: {str(y)}')
            img = all_data[:, input_x_start:input_x_end, y]
            img = (img - 0.6766) / 390.3082
            input = cv2.copyMakeBorder(img, 9, 9, 0, 0, cv2.BORDER_REPLICATE)
            # hw->chw
            input = np.expand_dims(input, 0)
            # chw->bchw
            input = np.expand_dims(input, 0)
            input = torch.from_numpy(input).to(DEVICE).float()
            output, _ = model(input)
            output = output[0, :, 9:-9, :]
            output = F.softmax(output,dim=0)
            output = output.detach().cpu().numpy()
            if SHOW_IMG_FLAG:
                show_img = np.argmax(output,axis=0)
                cv2.imwrite(LOG_PATH+'/'+MODEL_NAME+'_prediction_seam_ai.png',show_img*32)
                SHOW_IMG_FLAG = False
            prob_labels_data[:, :, explt_x_start:explt_x_start + SLIDE_WINDOW, y] = \
                output[:, :, : SLIDE_WINDOW]
        all_labels_data[:, explt_x_start:explt_x_start + SLIDE_WINDOW, :] = \
            np.argmax(
                prob_labels_data[:, :, explt_x_start:explt_x_start + SLIDE_WINDOW, :],
                axis=0)
    for y in range(labels_data.shape[2]):
        input_x_start = labels_data.shape[1] - SLICE_WIDTH
        explt_x_start = input_x_start
        input_x_end = labels_data.shape[1]
        logger.info(f'step: {str(step + 1)}, y: {str(y)}')
        img = all_data[:, input_x_start:input_x_end, y]
        img = (img - 0.6766) / 390.3082
        input = cv2.copyMakeBorder(img, 9, 9, 0, 0, cv2.BORDER_REPLICATE)
        # hw->chw
        input = np.expand_dims(input, 0)
        # chw->bchw
        input = np.expand_dims(input, 0)
        input = torch.from_numpy(input).to(DEVICE).float()
        output, _ = model(input)
        output = output[0, :, 9:-9, :]
        output = F.softmax(output, dim=0)
        output = output.detach().cpu().numpy()
        prob_labels_data[:, :, explt_x_start:explt_x_start + SLIDE_WINDOW, y] = \
            output[:, :, : SLIDE_WINDOW]

    all_labels_data[:, explt_x_start:explt_x_start + SLIDE_WINDOW, :] = \
        np.argmax(
            prob_labels_data[:, :, explt_x_start:explt_x_start + SLIDE_WINDOW, :],
            axis=0)
    np.savez_compressed(LOG_PATH+'/'+MODEL_NAME+'_prediction_seam_ai.npz', prediction=all_labels_data.astype(np.uint8))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--slice_width', nargs='?', type=int, default=256,
                        help='Slice width')
    parser.add_argument('--slide_window', nargs='?', type=int, default=256,
                        help='Slide window')
    parser.add_argument('--classes', nargs='?', type=int, default=6,
                        help='Classes')
    parser.add_argument('--model_path', nargs='?', type=str,
                        default='/volume/seis_facies_ident_113/runs/seam_ai_semi_sup_std/best_model.pkl')
    parser.add_argument('--train_data_path', nargs='?', type=str,
                        default='/volume/dataset/seam_ai/data_train.npz',
                        help='Train data path')
    parser.add_argument('--labels_data_path', nargs='?', type=str,
                        default='/volume/dataset/seam_ai/labels_train.npz',
                        help='Labels data path')
    args = parser.parse_args()
    predict(args)