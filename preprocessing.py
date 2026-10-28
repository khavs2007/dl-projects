"""STEP 4: image preprocessing (shared by ALL experiments so comparisons stay fair)."""
from torchvision import transforms

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # ResNet50 was pretrained with these statistics
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool, img_size: int = IMG_SIZE):
    steps = [
        # 1) Grayscale -> 3 channels: removes the colour shortcut (e.g. every 'stain'
        #    image was grayscale) but keeps 3 channels so ResNet50 still works.
        transforms.Grayscale(num_output_channels=3),
        # 2) Plain resize (no crop, no padding): keeps the whole image. Padding bars
        #    would reveal which source an image came from, cropping could cut a defect.
        transforms.Resize((img_size, img_size)),
    ]
    if train:
        steps += [
            # Flips keep a horizontal defect horizontal and a vertical defect vertical.
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            # Tiny brightness/contrast change = lighting variation, not a new defect.
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            # NOT used on purpose: 90-degree rotation (would turn 'horizontal' into
            # 'Vertical'), big crops/zoom (could delete the defect), strong distortions.
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)
