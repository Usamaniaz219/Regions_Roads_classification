import torchvision
import torch 
from PIL import Image
from torchvision import transforms
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torchvision.models.wide_resnet101_2(weights=torchvision.models.Wide_ResNet101_2_Weights.IMAGENET1K_V2)
model.to(device)
checkpoint_path = 'checkpoint_epoch_20.pth'
# model.load_state_dict(torch.load(checkpoint_path))
# model.eval()
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((1000, 1000)),  # Resize image to 500 * 500
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

def isRoads(image):
        # Define transformations (Resize, Tensor conversion, and Normalization)
    transform = transforms.Compose([
        transforms.Resize((1000, 1000)),  # Resize image to 500 * 500
        transforms.ToTensor(),          # Convert image to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
    ])
        # Load the image
    Image.MAX_IMAGE_PIXELS = None  # Disable the limit check
    # image = Image.open(image_path).convert('RGB')
    # Apply the transformations
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    # Forward pass through the model
    with torch.no_grad():
        model = load_checkpoint(checkpoint_path)

        output = model(image)
        _, predicted = torch.max(output, 1)  # Get the predicted class
        label = predicted.cpu().item()
        print("label:",label)
    if label == 1:
        return True
    else:
        return False
    
# image_path = "Roads_and_regions_dataset/train/regions/demo141_1_mask_demo141_57.jpg"
road_dir = "/media/usama/SSD/Roads_Regions_Classification/Roads_and_regions_dataset/Failed_Cases/"
road_images = [f for f in os.listdir(road_dir)]
for road_image in road_images:
    print("Region image:",road_image)
    road_image_path = os.path.join(road_dir, road_image)
    image = Image.open(road_image_path).convert('RGB')
    label = isRoads(image)
    print("label:",label)


