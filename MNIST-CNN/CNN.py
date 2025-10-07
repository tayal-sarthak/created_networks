# MNIST digit recognition
# citations: DataCamp, https://www.youtube.com/watch?v=PYnrWyuEEa4, https://www.youtube.com/watch?v=e1HqOjLCvms, https://www.youtube.com/watch?v=L2cAjgc1-bo
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

# ----------------------------
# 0) Reproducibility settings
# ----------------------------
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(RANDOM_SEED)


# ---------------------------------
# 1) Transforms and dataset objects
# ---------------------------------
# We normalize MNIST using the dataset's standard mean and std.
# This makes optimization easier while gradients output stronger results.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))

# Compose transforms in a readable way (step by step)
train_transform = transforms.Compose([
    to_tensor,
    normalize,
])

test_transform = transforms.Compose([
    to_tensor,
    normalize,
])

# Download (first run) and prepare datasets
train_full_dataset = datasets.MNIST(
    root="./cnn_data",
    train=True,
    download=True,
    transform=train_transform,
)

test_dataset = datasets.MNIST(
    root="./cnn_data",
    train=False,
    download=True,
    transform=test_transform,
)

# Create a fresh train/validation split each run so we truly restart
def split_datasets(random_seed):
    generator = torch.Generator().manual_seed(random_seed)
    validation_size = 5000
    train_size = len(train_full_dataset) - validation_size
    train_dataset, validation_dataset = random_split(
        train_full_dataset,
        [train_size, validation_size],
        generator=generator,
    )
    return train_dataset, validation_dataset


# -----------------------
# 2) Data loader builders
# -----------------------
# Batch size controls how many images we process at once.
BATCH_SIZE = 64

def build_data_loaders(train_dataset, validation_dataset, batch_size):
    # Create training, validation, and test data loaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,   # shuffle only training data
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, validation_loader, test_loader


# -----------------------
# 3) Visualization helpers
# -----------------------

def unnormalize_for_display(tensor_batch):
    
    #Undo MNIST normalization so we can display images correctly.
    #The formula is: original = normalized * std + mean.
    # We also clamp to [0, 1] so matplotlib is happy. 0 to 1 reminds me of softmax/sigmoid :(

    mean = MNIST_MEAN
    std = MNIST_STD
    original = tensor_batch * std + mean
    original = original.clamp(0.0, 1.0)
    return original


def show_image_grid_from_loader(data_loader, images_to_show):
    # Display a small grid of images pulled from a single batch.
    batch_images, batch_labels = next(iter(data_loader))
    batch_images = batch_images[:images_to_show]

    batch_images = unnormalize_for_display(batch_images)

    grid = make_grid(batch_images, nrow=4)

    plt.figure(figsize=(4, 4))
    if grid.shape[0] == 1:
        plt.imshow(grid[0], cmap="gray", vmin=0.0, vmax=1.0)
    else:
        plt.imshow(grid.permute(1, 2, 0), cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ----------------------
# 4) Model architecture
# ----------------------
class SmallCnnClassifier(nn.Module):

    #A very small, readable CNN for 28x28 grayscale images.
    #Layers are kept simple and sequential for clarity.

    def __init__(self):
        super(SmallCnnClassifier, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),  # downsample to 14x14

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),  # downsample to 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),  # 10 digits: 0..9
        )

    def forward(self, inputs):
        features = self.feature_extractor(inputs)
        logits = self.classifier(features)
        return logits  # raw scores; CrossEntropyLoss will handle softmax internally


# 5) training and evaluation utilities


def train_one_epoch(model, data_loader, optimizer, loss_function, device):
    # run one full pass over the training data and return average loss.
    model.train()
    running_loss_total = 0.0
    running_items = 0

    for batch_images, batch_targets in data_loader:
        batch_images = batch_images.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        batch_logits = model(batch_images)
        batch_loss = loss_function(batch_logits, batch_targets)
        batch_loss.backward()
        optimizer.step()

        running_loss_total += batch_loss.item() * batch_targets.size(0)
        running_items += batch_targets.size(0)

    average_loss = running_loss_total / float(running_items)
    return average_loss


def evaluate_accuracy(data_loader, model, device):
    # compute simple classification accuracy on the given data loader.
    model.eval()
    correct_predictions = 0
    total_examples = 0

    with torch.no_grad():
        for batch_images, batch_targets in data_loader:
            batch_images = batch_images.to(device)
            batch_targets = batch_targets.to(device)

            batch_logits = model(batch_images)
            batch_predictions = batch_logits.argmax(dim=1)

            correct_predictions += (batch_predictions == batch_targets).sum().item()
            total_examples += batch_targets.size(0)

    accuracy = correct_predictions / float(total_examples)
    return accuracy


def predict_top1_for_samples(model, data_loader, device, number_of_images):
    # take a few images, compute class probabilities with softmax YAY, and return 
    # predicted classes and their top-1 probabilities.
    model.eval()
    images, targets = next(iter(data_loader))
    images = images[:number_of_images]

    with torch.no_grad():
        logits = model(images.to(device))

    probabilities = torch.softmax(logits.cpu(), dim=1)
    predictions = probabilities.argmax(dim=1)
    top1_probabilities = probabilities.max(dim=1).values

    return predictions.tolist(), top1_probabilities.tolist()


if __name__ == "__main__":
    # determinism knobs; set to true for identical results each run
    FULLY_DETERMINISTIC = False

    print("original_train:", len(train_full_dataset), "test:", len(test_dataset))
    label_counts = torch.bincount(train_full_dataset.targets, minlength=10)
    print({i: int(count) for i, count in enumerate(label_counts.tolist())})

    # show one sample grid (do this once so doesn't block five times)
    # build temporary split just for the preview
    preview_train, preview_val = split_datasets(RANDOM_SEED)
    preview_train_loader, preview_val_loader, _ = build_data_loaders(preview_train, preview_val, BATCH_SIZE)
    show_image_grid_from_loader(preview_train_loader, images_to_show=16)

    runs = 5
    all_test_acc = []

    for run_index in range(1, runs + 1):
        run_seed = RANDOM_SEED + run_index
        torch.manual_seed(run_seed)
        if FULLY_DETERMINISTIC and DEVICE == "cuda":
            cudnn.deterministic = True
            cudnn.benchmark = False

        # fresh split + loaders each run
        train_dataset, validation_dataset = split_datasets(run_seed)
        train_loader, validation_loader, test_loader = build_data_loaders(train_dataset, validation_dataset, BATCH_SIZE)

        model = SmallCnnClassifier().to(DEVICE)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        total_epochs = 3
        for epoch_index in range(total_epochs):
            average_train_loss = train_one_epoch(
                model=model,
                data_loader=train_loader,
                optimizer=optimizer,
                loss_function=loss_function,
                device=DEVICE,
            )
            validation_accuracy = evaluate_accuracy(validation_loader, model, DEVICE)
            print("run", run_index, "epoch", epoch_index + 1,
                  "train_loss:", round(average_train_loss, 4),
                  "val_acc:", round(validation_accuracy, 4))

        test_accuracy = evaluate_accuracy(test_loader, model, DEVICE)
        all_test_acc.append(test_accuracy)
        print("run", run_index, "test_acc:", round(test_accuracy, 4))

    # summary across runs
    avg_test = sum(all_test_acc) / float(len(all_test_acc))
    print("avg_test_acc_over_", runs, "_runs:", round(avg_test, 4))