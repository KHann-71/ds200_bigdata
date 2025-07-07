from trainer import Trainer
from models import SVM
from transforms import Transforms, Normalize, RandomHorizontalFlip

if __name__ == "__main__":
    transforms = Transforms([
        RandomHorizontalFlip(p=0.5),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    model = SVM(n_components=100)  # MLP + PCA
    trainer = Trainer(model=model, host="localhost", port=6100, transforms=transforms)
    trainer.train()
