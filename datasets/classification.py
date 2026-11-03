"""Multi-label classification of VID frames using all annotated object categories."""
import random
import torch
from torchvision.transforms import functional as TF
from datasets.transforms import resize_detection_views
from datasets.paired_frames import stack_frames


class ClassificationTransform:
    def __init__(self, backbone, training=True):
        self.interpolation = 'bicubic' if backbone in ('timm_mobilenet_v2', 'tinynet_d', 'efficientnet_b0') else 'bilinear'
        self.paired = backbone in ('timm_mobilenet_v2', 'tinynet_d', 'efficientnet_b0')
        self.training = training

    def __call__(self, images, targets):
        # Keep the full frame: crops could remove objects while retaining their image labels.
        views = []
        for image in images:
            image = image.convert('RGB')
            if self.training and random.random() < .5:
                image = TF.hflip(image)
            views.append(resize_detection_views(image, 320, self.interpolation, self.paired))
        return stack_frames(views), targets


class VIDClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, detection_dataset, backbone, training=True):
        if detection_dataset.clip_length != 1 or detection_dataset.num_classes != 30:
            raise ValueError('Classification requires single frames with all 30 VID categories.')
        self.detection_dataset = detection_dataset
        self.detection_dataset.transform = ClassificationTransform(backbone, training)

    def __len__(self):
        return len(self.detection_dataset)

    def __getitem__(self, index):
        frames, targets = self.detection_dataset[index]
        labels = targets[0]['labels'].long()
        if ((labels < 1) | (labels > 30)).any():
            raise ValueError('VID labels must be foreground indices 1..30.')
        target = torch.zeros(30, dtype=torch.float32)
        target[labels.unique() - 1] = 1.
        return frames[0], target


def classification_collate(batch):
    images, labels = zip(*batch)
    return stack_frames(images), torch.stack(labels)
