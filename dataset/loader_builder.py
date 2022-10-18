from torch.utils.data import DataLoader
from dataset.dataset import *
from utils.utils import *
warnings.filterwarnings('ignore')

def build_loader(cfg):
    logger = get_logger('seis_facies_ident')
    logger.propagate = False
    dataset = cfg["dataset"]
    if dataset["name"] == "seam_ai_semi":
        train_data = np.load(dataset["data_path"])['data']
        # label：1-6 -> 0-5
        train_labels = np.load(dataset["labels_path"])['labels'] - 1
        SLICE_WIDTH = dataset["slice_width"]
        BATCH_SIZE = dataset["batch_size"]

        logger.info(f"Make semi dataset...")
        semi_dataset = UnlabelDataset(data=train_data,labels=train_labels,
                                      augmentations=Compose([PadIfNeeded(1024, SLICE_WIDTH, p=1)]),slice_width=SLICE_WIDTH,
                                      sampling_pos=dataset["sampling_pos"])


        logger.info(f"Semi dataset: {len(semi_dataset)}")

        logger.info(f"Make sup dataset...")
        sup_dataset = FewLabelDataset(data=train_data, labels=train_labels,
                                         augmentations=Compose([PadIfNeeded(1024, SLICE_WIDTH, p=1)]),slice_width=SLICE_WIDTH,
                                         sampling_pos=dataset["sampling_pos"],sparse=dataset["sparse"])
        sup_dataset.over_sample(len(semi_dataset))
        logger.info(f"Sup dataset: {len(sup_dataset)}")

        sup_loader = DataLoader(sup_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers=0,drop_last=True)
        semi_loader = DataLoader(semi_dataset,batch_size=BATCH_SIZE, shuffle=True, num_workers=0,drop_last=True)
        validate_loader = DataLoader(semi_dataset,batch_size=BATCH_SIZE*4, shuffle=False, num_workers=0,drop_last=True)
        return sup_loader,semi_loader,validate_loader
    elif dataset["name"] == "f3_semi":
        # xyz: 401*701*255
        train_labels = np.load(dataset["train_labels_path"])
        # 200*701*255
        test1_labels = np.load(dataset["test1_labels_path"])
        # 601*200*255
        test2_labels = np.load(dataset["test2_labels_path"])
        # 601*701*255 + 601*200*255 ->255*901*601
        train_labels = np.concatenate([np.concatenate([test1_labels, train_labels], axis=0),test2_labels],axis=1).transpose(2,1,0)
        test1_data = np.load(dataset["test1_data_path"])
        test2_data = np.load(dataset["test2_data_path"])
        train_data = np.load(dataset["train_data_path"])
        train_data = np.concatenate([np.concatenate([test1_data, train_data], axis=0),test2_data],axis=1).transpose(2,1,0)
        SLICE_WIDTH = dataset["slice_width"]
        BATCH_SIZE = dataset["batch_size"]
        logger.info(f"Make semi dataset...")
        semi_dataset = UnlabelDataset(data=train_data, labels=train_labels,
                                      augmentations=Compose([PadIfNeeded(256, SLICE_WIDTH, p=1)]),
                                      slice_width=SLICE_WIDTH,
                                      sampling_pos=dataset["sampling_pos"])

        logger.info(f"Semi dataset: {len(semi_dataset)}")

        logger.info(f"Make sup dataset...")
        sup_dataset = FewLabelDataset(data=train_data, labels=train_labels,
                                         augmentations=Compose([PadIfNeeded(256, SLICE_WIDTH, p=1)]),
                                         slice_width=SLICE_WIDTH,
                                         sampling_pos=dataset["sampling_pos"],sparse=dataset["sparse"])
        sup_dataset.over_sample(len(semi_dataset))
        logger.info(f"Sup dataset: {len(sup_dataset)}")

        sup_loader = DataLoader(sup_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
        semi_loader = DataLoader(semi_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
        validate_loader = DataLoader(semi_dataset, batch_size=BATCH_SIZE * 4, shuffle=False, num_workers=0,
                                     drop_last=True)

        return sup_loader, semi_loader, validate_loader