import yaml
import os
import segmentation_models_pytorch as smp
from tensorboardX import SummaryWriter
from datetime import datetime
from dataset.loader_builder import *
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models
from net.deeplabv3plus import DeepLabv3Plus
from utils.loss import *
import torch
import shutil
warnings.filterwarnings('ignore')
# global settings
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

cfg_path = '/volume/seis_facies_ident/conf/seam_ai_sup_sparse.yaml'
cfg = yaml.load(open(cfg_path, "r"), Loader=yaml.Loader)

logger = get_logger('seis_facies_ident')
logger.propagate = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sce_loss = smp.losses.SoftCrossEntropyLoss(smooth_factor=cfg["train"]["smooth_factor"],ignore_index=-1)

CURRENT_TIME = datetime.now().strftime('%b%d_%H%M%S')
LOG_DIR = os.path.join(os.getcwd()+'/runs', CURRENT_TIME + "_{}".format(cfg["train"]["dir_suffix"]))
writer = SummaryWriter(log_dir=LOG_DIR)
shutil.copy(cfg_path,LOG_DIR)

ROUND_NUM = 6
BEST_METRIC = 0


def init_model(cfg):
    global logger,DEVICE
    cfg_train = cfg["train"]
    CLASSES = cfg_train["classes"]
    model = DeepLabv3Plus(models.resnet101(pretrained=True), num_classes=CLASSES, output_dim=128).to(DEVICE)
    model_total_param = sum([param.nelement() for param in model.parameters()])
    logger.info(f"Model total parameters {model_total_param}")

    # use DataParallel if more than 1 GPU available
    if torch.cuda.device_count() > 1 and not DEVICE.type == 'cpu':
        model = torch.nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for training')
    logger.info(f"Sending the model to {DEVICE}")
    model = model.to(DEVICE)
    return model

def main():
    global cfg
    logger.info(cfg)
    LR = cfg["train"]["lr"]
    model = init_model(cfg)
    optimizer = torch.optim.Adam(params=model.parameters(),lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.2)
    sup_loader, semi_loader, valid_loader= build_loader(cfg)
    # smp
    scaler = GradScaler()
    for epoch in range(cfg["train"]["max_epoch"]):
        train(cfg,model,optimizer,scaler,lr_scheduler,sup_loader,epoch)
        validate(cfg,model,valid_loader,epoch)

def train(cfg,model,optimizer,scaler,lr_scheduler,sup_loader,epoch):
    model.train()
    cfg_train = cfg["train"]
    MAX_EPOCH = cfg_train["max_epoch"]
    CLASSES = cfg_train["classes"]
    SDA = cfg_train["sda"]
    CONTRA_LOSS = cfg_train["contra_loss"]
    STRONG_TH = cfg_train["contrastive"]["strong_threshold"]
    TEMP = cfg_train["contrastive"]["temperature"]
    NUM_NEG = cfg_train["contrastive"]["num_negatives"]
    NUM_QE = cfg_train["contrastive"]["num_queries"]
    sup_running_metrics_train = runningScore(CLASSES)
    sup_running_metrics_train.reset()
    sup_loader_iter = iter(sup_loader)
    for i_train in range(len(sup_loader)):
        sup_images, sup_labels = sup_loader_iter.next()
        sup_images, sup_labels = sup_images.to(DEVICE).to(torch.float32), sup_labels.to(DEVICE).to(torch.int64)
        if SDA:
            sup_images,sup_labels,_ = \
                generate_aug_data(sup_images, sup_labels, sup_labels, mode=SDA)

        optimizer.zero_grad()
        with autocast():
            pred_sup, rep_sup = model(sup_images)
            sup_loss = sce_loss(pred_sup,sup_labels)

        if CONTRA_LOSS:
            with torch.no_grad():
                mask_all = (sup_labels.unsqueeze(1)>=0).float()
                label_all = label_onehot(sup_labels,CLASSES)
                prob_sup = torch.softmax(pred_sup, dim=1)
            with autocast():
                con_loss = contra_loss(rep_sup,label_all,mask_all,prob_sup,STRONG_TH,TEMP,NUM_QE,NUM_NEG)
        else:
            con_loss = torch.tensor(0.0)

        loss = sup_loss + con_loss

        sup_running_metrics_train.update(sup_labels.detach().cpu().numpy(), pred_sup.detach().max(1)[1].cpu().numpy())
        writer.add_scalar('sup_loss', round(sup_loss.item(), ROUND_NUM), i_train+epoch*len(sup_loader))
        writer.add_scalar('con_loss', round(con_loss.item(), ROUND_NUM), i_train + epoch * len(sup_loader))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        logger.info(
            f'Train: epoch [{epoch}/{MAX_EPOCH - 1}] iteration [{i_train}] sup_loss {round(sup_loss.item(), ROUND_NUM)} '
            f'con_loss {round(con_loss.item(), ROUND_NUM)} '
            f'lr {round(optimizer.param_groups[0]["lr"],ROUND_NUM)}')

    lr_scheduler.step()

    train_sup_score, train_sup_class_iou = sup_running_metrics_train.get_scores()
    for key in train_sup_score.keys():
        if key != 'Class Accuracy':
            writer.add_scalar('train_sup/' + key, train_sup_score[key], epoch)
            logger.info(f'Train_sup: {key} {round(train_sup_score[key], ROUND_NUM)}')

def validate(cfg,model,valid_loader,epoch):
    global DEVICE, BEST_METRIC, LOG_DIR, writer
    cfg_train = cfg["train"]
    CLASSES = cfg_train["classes"]
    MAX_EPOCH = cfg_train["max_epoch"]
    running_metrics_val = runningScore(CLASSES)
    running_metrics_val.reset()
    METRIC = cfg["valid"]["metric"]
    with torch.no_grad():
        model.eval()
        for i_val, (images_val, labels_val) in enumerate(valid_loader):
            images_val, labels_val = images_val.to(DEVICE).to(torch.float32), labels_val.to(DEVICE).to(torch.int64)
            outputs_val,_ = model(images_val)
            loss = sce_loss(outputs_val, labels_val)
            pred = outputs_val.detach().max(1)[1].cpu().numpy()
            gt = labels_val.detach().cpu().numpy()
            running_metrics_val.update(gt, pred)

            logger.info(
                f'Validation epoch [{epoch }/{MAX_EPOCH - 1}] iteration [{i_val}] sup_loss: {round(loss.item(), ROUND_NUM)}')

        val_score, val_class_iou = running_metrics_val.get_scores()

        for key in val_score.keys():
            if key != 'Class Accuracy':
                writer.add_scalar('val/' + key, val_score[key], epoch)
                logger.info(f'Validation: {key} {round(val_score[key], ROUND_NUM)}')

        MODEL_PATH = os.path.join(LOG_DIR, f"epoch" + str(epoch) + "_model.pkl")
        torch.save(model, MODEL_PATH)
        # save checkpoint
        LAST_MODEL_PATH = os.path.join(LOG_DIR, f"last_model.pkl")
        torch.save(model, LAST_MODEL_PATH)
        logger.info(f'Save checkpoint {LAST_MODEL_PATH}')
        if val_score[METRIC] >= BEST_METRIC:
            BEST_METRIC = val_score[METRIC]
            BEST_MODEL_PATH = os.path.join(LOG_DIR, f"best_model.pkl")
            torch.save(model, BEST_MODEL_PATH)
            logger.info(f'Save checkpoint {BEST_MODEL_PATH}')

if __name__ == '__main__':
    main()
