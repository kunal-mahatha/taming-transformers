class COCODataset(CustomBase):
    def __init__(self, size, annotation_file, image_dir, split, transform=None):
        super().__init__()
        self.size = size
        self.annotation_file = annotation_file
        self.image_dir = image_dir
        self.split = split
        self.transform = transform
        self.coco = datasets.CocoCaptions(root=self.image_dir,
                                         annFile=self.annotation_file,
                                         split=self.split)
    def __len__(self):
        return len(self.coco)

    def __getitem__(self, i):
        example = self.coco[i]
        if self.transform:
            example = self.transform(example)
        return example

class COCOTrain(COCODataset):
    def __init__(self, size, annotation_file, image_dir, transform=None):
        super().__init__(size, annotation_file, image_dir, 'train', transform)

class COCOTest(COCODataset):
    def __init__(self, size, annotation_file, image_dir, transform=None):
        super().__init__(size, annotation_file, image_dir, 'val', transform)
        
transforms = albumentations.Compose([albumentations.Resize(256, 256),
                                      albumentations.RandomCrop(224,224),
                                      albumentations.ToTensor()])

train_dataset = COCOTrain(224, 'path/to/annotations.json', 'path/to/COCO/', transform = transforms)
test_dataset = COCOTest(224, 'path/to/annotations.json', 'path/to/COCO/', transform = transforms)

dataloader = torch.utils.data.DataLoader(train_dataset,
                                         batch_size=32,
                                         shuffle=True,
                                         num_workers=4)
