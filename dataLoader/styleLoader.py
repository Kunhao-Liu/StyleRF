from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T


def getDataLoader(dataset_path, batch_size, sampler, image_side_length=256, num_workers=2):
    transform = T.Compose([
                T.Resize(size=(image_side_length*2, image_side_length*2)),
                T.RandomCrop(image_side_length),
                T.ToTensor(),
            ])

    train_dataset = datasets.ImageFolder(dataset_path, transform=transform)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler(len(train_dataset)), num_workers=num_workers)

    return dataloader