import torch
from torch.utils.data import Dataset,DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
# from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os

class RoadsRegionsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Populate image paths and labels
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                self.image_paths.append(os.path.join(cls_folder, img_name))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # print("idx",idx)
        img_path = self.image_paths[idx]
        # print("image path",img_path)
        label = self.labels[idx]
        # print("label",type(label))

        image = Image.open(img_path).convert('RGB')  # Open as grayscale
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.float32)

        return image, label

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=3),
    transforms.Resize((1000, 1000)),  # Resize image to 500 * 500
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

model = torchvision.models.wide_resnet101_2(weights=torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V2)
# model.to(device)
# print("model fully conneted features",model.fc.in_features)
num_classes = 2
in_features = 2048
model.fc = torch.nn.Linear(in_features, num_classes)
print("model fully conneted features",model.fc.out_features)
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

# summary(model,(3,1000,1000))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, 
                num_epochs=50, save_interval=10, log_dir="runs/experiment2", 
                checkpoint_path=None, start_epoch=0):

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    model.train()  # Set the model to training mode

    # Load checkpoint if provided (resume training)
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch + 1}")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # Iterate over the DataLoader
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)  # Move data to the GPU
            model.to(device)
            # print("images",images)
            # print("labels",labels)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels.long())  # Ensure labels are long type for CrossEntropyLoss

            # Backward pass and optimization
            optimizer.zero_grad()  # Zero the gradients
            loss.backward()        # Backpropagation
            optimizer.step()       # Update the weights

            # Calculate loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # Get the index of the max log-probability
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels.long()).sum().item()

        # Calculate average loss and accuracy for the epoch
        avg_loss = running_loss / len(train_loader)
        # print("average loss",avg_loss)
        accuracy = (correct_predictions / total_predictions) * 100
        # print("accuracy",accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train", avg_loss, epoch + 1)
        writer.add_scalar("Accuracy/Train", accuracy, epoch + 1)

        # Save checkpoint every 'save_interval' epochs
        if (epoch + 1) % save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model': model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            save_path = f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved at {save_path}")

    # Close the TensorBoard writer
    writer.close()

train_dir = r"/media/usama/SSD/Roads_Regions_Classification/Roads_and_regions_dataset/train/"
checkpoint_path = "checkpoint_epoch_20.pth"
train_dataset = RoadsRegionsDataset(root_dir=train_dir,transform=transform)
print("length of train dataset",len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)

train_model(model, train_loader, criterion, optimizer)
    

