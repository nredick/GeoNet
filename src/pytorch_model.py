import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Define Focal Loss function
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(focal_loss)
        elif self.reduction == "sum":
            return torch.sum(focal_loss)
        else:
            return focal_loss


def intersection_over_union(predicted, target, epsilon=1e-8):
    intersection = (predicted * target).sum(
        dim=(1, 2)
    )  # Calculate the intersection for each sample
    union = (predicted + target).sum(
        dim=(1, 2)
    ) - intersection  # Calculate the union for each sample

    iou = (intersection + epsilon) / (
        union + epsilon
    )  # Calculate IoU for each sample, adding epsilon to avoid division by zero

    # Return the mean IoU across all samples
    return iou.mean()


# Convert numpy arrays to tensors with permuted dimensions (N, C, H, W)
X_train_tensor = torch.from_numpy(X_train).permute(0, 3, 1, 2)
Y_train_tensor = torch.from_numpy(Y_train).permute(0, 3, 1, 2)

# Create a TensorDataset
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)

# Create a train dataloader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# print shapes
print("X_train_tensor:", X_train_tensor.shape)
print("Y_train_tensor:", Y_train_tensor.shape)

# repeat for validation data loader
X_val_tensor = torch.from_numpy(X_val).permute(0, 3, 1, 2)
Y_val_tensor = torch.from_numpy(Y_val).permute(0, 3, 1, 2)

val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("\nX_val_tensor:", X_val_tensor.shape)
print("Y_val_tensor:", Y_val_tensor.shape)

# repeat for test data loader
X_test_tensor = torch.from_numpy(X_test).permute(0, 3, 1, 2)
Y_test_tensor = torch.from_numpy(Y_test).permute(0, 3, 1, 2)

test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("\nX_test_tensor:", X_test_tensor.shape)
print("Y_test_tensor:", Y_test_tensor.shape)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, filters=32):
        super(UNet, self).__init__()
        self.float()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters, filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(filters, out_channels, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2
