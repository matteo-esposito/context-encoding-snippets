##### 1 BREAK

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import timm

# Define the context encoding module
class ContextEncodingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContextEncodingModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

# Define the main model with context encoding
class CustomModel(nn.Module):
    def __init__(self, base_model, context_encoding_channels=64):
        super(CustomModel, self).__init__()

        # Load pre-trained base model
        self.base_model = timm.create_model(base_model, pretrained=True)

        # Add context encoding module
        in_channels = self.base_model.num_features
        self.context_encoding = ContextEncodingModule(in_channels, context_encoding_channels)

    def forward(self, x):
        # Base model forward pass
        base_model_features = self.base_model(x)

        # Apply context encoding
        context_encoded_features = self.context_encoding(base_model_features)

        # Append context-encoded features to the input
        x = torch.cat([x, context_encoded_features], dim=1)

        # Continue with the rest of the base model forward pass or additional processing

        return x

# Example usage
base_model_name = 'resnet18'  # You can choose a different base model
model = CustomModel(base_model_name)

# Print the model architecture
print(model)


##### 2 BREAK

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

class ContextEncodingModel(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super(ContextEncodingModel, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, output_channels),
            nn.Sigmoid()  # Sigmoid activation for binary classification
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Example usage
def process_and_encode_images(image_paths, context_model):
    context_model.eval()

    # Transformations for the input images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.ToTensor(),
    ])

    for image_path in image_paths:
        # Load and preprocess the original image
        original_image = Image.open(image_path).convert('RGB')
        input_image = transform(original_image).unsqueeze(0)  # Add batch dimension

        # Forward pass through the context encoding model
        output_prob = context_model(input_image)

        # Use the output probability as needed
        print(f"Image: {image_path}, Probability: {output_prob.item()}")

# Example usage
image_paths = ["path/to/not_totaled_car.jpg", "path/to/totaled_car.jpg"]
context_model = ContextEncodingModel(input_channels=3, output_channels=1)
process_and_encode_images(image_paths, context_model)



#### 3 BREAK (panoptic segmentation then timm)

# Certainly! To achieve this, you can follow these steps:
#    Perform panoptic segmentation using a pre-trained Panoptic FPN model.
#    Aggregate the segmentation masks into a single channel mask.
#    Concatenate the new mask to the original image.
#    Use the resulting image as input to a pre-trained timm model.

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torchvision.models.detection import panoptic_segmentation
import timm
import matplotlib.pyplot as plt

# Load a pre-trained Panoptic FPN model
def load_panoptic_fpn_model():
    model = models.detection.panoptic_segmentation.panoptic_fpn_resnet50(pretrained=True)
    return model

# Load a pre-trained timm model
def load_pretrained_timm_model(model_name):
    model = timm.create_model(model_name, pretrained=True)
    return model

# Function to perform panoptic segmentation and concatenate the masks to the original image
def panoptic_segmentation_and_concat(image_path, panoptic_fpn_model, timm_model_name):
    panoptic_fpn_model.eval()

    # Load and preprocess the original image
    original_image = Image.open(image_path).convert('RGB')

    # Transformations for the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.ToTensor(),
    ])

    input_image = transform(original_image).unsqueeze(0)  # Add batch dimension

    # Forward pass through the Panoptic FPN model
    with torch.no_grad():
        panoptic_segmentation_result = panoptic_fpn_model(input_image)[0]

    # Extract instance segmentation masks
    instance_masks = panoptic_segmentation_result['instances'].numpy()

    # Aggregate the instance masks into a single channel mask (sum over all instances)
    aggregated_mask = instance_masks.sum(axis=0, keepdims=True)

    # Normalize the aggregated mask to the range [0, 1]
    aggregated_mask = aggregated_mask / aggregated_mask.max()

    # Convert the aggregated mask to a tensor
    aggregated_mask_tensor = torch.FloatTensor(aggregated_mask)

    # Concatenate the aggregated mask to the original image as a new channel
    input_with_mask = torch.cat([input_image, aggregated_mask_tensor], dim=1)

    # Load a pre-trained timm model
    timm_model = load_pretrained_timm_model(timm_model_name)

    # Forward pass through the timm model using the image with the concatenated mask
    with torch.no_grad():
        output_timm_model = timm_model(input_with_mask)

    # Use the output as needed
    print("Timm Model Output:", output_timm_model)

    # Optionally, visualize the results
    visualize_result(original_image, aggregated_mask, input_with_mask)

def visualize_result(original_image, aggregated_mask, input_with_mask):
    # Plot original image, aggregated mask, and input image with the concatenated mask
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(aggregated_mask[0], cmap='gray')
    plt.title("Aggregated Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(input_with_mask.squeeze().permute(1, 2, 0))
    plt.title("Input with Concatenated Mask")

    plt.show()

# Example usage
image_path = "path/to/image.jpg"
timm_model_name = "resnet18"  # Replace with your preferred timm model
panoptic_fpn_model = load_panoptic_fpn_model()
panoptic_segmentation_and_concat(image_path, panoptic_fpn_model, timm_model_name)

##### 4 BREAK, adapting timm to use 4 channel output which comes from the segmentation model

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torchvision.models.detection import panoptic_segmentation
import timm
import matplotlib.pyplot as plt

# Load a pre-trained Panoptic FPN model
def load_panoptic_fpn_model():
    model = models.detection.panoptic_segmentation.panoptic_fpn_resnet50(pretrained=True)
    return model

# Load a pre-trained timm model with modified input channels
def load_pretrained_timm_model(model_name, input_channels=4):
    model = timm.create_model(model_name, pretrained=True)
    
    # Modify the first convolutional layer to accept 4 channels
    if hasattr(model, 'conv1'):
        model.conv1 = torch.nn.Conv2d(input_channels, model.conv1.out_channels, 
                                      kernel_size=model.conv1.kernel_size,
                                      stride=model.conv1.stride,
                                      padding=model.conv1.padding,
                                      bias=False)
    elif hasattr(model, 'features'):
        model.features[0] = torch.nn.Conv2d(input_channels, model.features[0].out_channels, 
                                           kernel_size=model.features[0].kernel_size,
                                           stride=model.features[0].stride,
                                           padding=model.features[0].padding,
                                           bias=False)
    else:
        raise ValueError("Unsupported model architecture. Modify the input channels manually.")

    return model

# Function to perform panoptic segmentation and concatenate the masks to the original image
def panoptic_segmentation_and_concat(image_path, panoptic_fpn_model, timm_model_name):
    panoptic_fpn_model.eval()

    # Load and preprocess the original image
    original_image = Image.open(image_path).convert('RGB')

    # Transformations for the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.ToTensor(),
    ])

    input_image = transform(original_image).unsqueeze(0)  # Add batch dimension

    # Forward pass through the Panoptic FPN model
    with torch.no_grad():
        panoptic_segmentation_result = panoptic_fpn_model(input_image)[0]

    # Extract instance segmentation masks
    instance_masks = panoptic_segmentation_result['instances'].numpy()

    # Aggregate the instance masks into a single channel mask (sum over all instances)
    aggregated_mask = instance_masks.sum(axis=0, keepdims=True)

    # Normalize the aggregated mask to the range [0, 1]
    aggregated_mask = aggregated_mask / aggregated_mask.max()

    # Convert the aggregated mask to a tensor
    aggregated_mask_tensor = torch.FloatTensor(aggregated_mask)

    # Concatenate the aggregated mask to the original image as a new channel
    input_with_mask = torch.cat([input_image, aggregated_mask_tensor], dim=1)

    # Load a pre-trained timm model with modified input channels
    timm_model = load_pretrained_timm_model(timm_model_name, input_channels=4)

    # Forward pass through the timm model using the image with the concatenated mask
    with torch.no_grad():
        output_timm_model = timm_model(input_with_mask)

    # Use the output as needed
    print("Timm Model Output:", output_timm_model)

    # Optionally, visualize the results
    visualize_result(original_image, aggregated_mask, input_with_mask)

def visualize_result(original_image, aggregated_mask, input_with_mask):
    # Plot original image, aggregated mask, and input image with the concatenated mask
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")

    plt.subplot(1, 3, 2)
    plt.imshow(aggregated_mask[0], cmap='gray')
    plt.title("Aggregated Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(input_with_mask.squeeze().permute(1, 2, 0))
    plt.title("Input with Concatenated Mask")

    plt.show()

# Example usage
image_path = "path/to/image.jpg"
timm_model_name = "resnet18"  # Replace with your preferred timm model
panoptic_fpn_model = load_panoptic_fpn_model()
panoptic_segmentation_and_concat(image_path, panoptic_fpn_model, timm_model_name)


##### 5 BREAK - code to checkout the masks for inference

def visualize_result(original_image, semantic_mask, instance_mask, input_with_mask):
    # Plot original image, semantic mask, instance mask, and input image with the concatenated mask
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(original_image)
    plt.title("Original Image")

    plt.subplot(1, 4, 2)
    plt.imshow(semantic_mask)
    plt.title("Semantic Mask")

    plt.subplot(1, 4, 3)
    plt.imshow(instance_mask)
    plt.title("Instance Mask")

    plt.subplot(1, 4, 4)
    plt.imshow(input_with_mask.squeeze().permute(1, 2, 0))
    plt.title("Input with Concatenated Mask")

    plt.show()

# Example usage
image_path = "path/to/image.jpg"
timm_model_name = "resnet18"  # Replace with your preferred timm model
panoptic_fpn_model = load_panoptic_fpn_model()
panoptic_segmentation_and_concat(image_path, panoptic_fpn_model, timm_model_name)

##### 6 BREAK - dataloader example of break 4.

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import boto3
from io import BytesIO

class PanopticSegmentationDataset(Dataset):
    def __init__(self, s3_bucket, s3_prefix, panoptic_fpn_model, timm_model_name):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.panoptic_fpn_model = panoptic_fpn_model
        self.timm_model_name = timm_model_name
        self.s3_client = boto3.client('s3')

        # Define transformations for the input images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust size as needed
            transforms.ToTensor(),
        ])

    def __len__(self):
        # Implement the total number of images in your dataset
        # You may need to list objects in the S3 bucket and calculate the length
        return len(list_objects_in_bucket())

    def __getitem__(self, idx):
        # Load and preprocess the original image from S3
        image_path = self.get_image_path_from_s3(idx)
        original_image = self.load_image_from_s3(image_path)
        input_image = self.transform(original_image).unsqueeze(0)  # Add batch dimension

        # Forward pass through the Panoptic FPN model
        with torch.no_grad():
            panoptic_segmentation_result = self.panoptic_fpn_model(input_image)[0]

        # Extract instance segmentation masks
        instance_masks = panoptic_segmentation_result['instances'].numpy()

        # Aggregate the instance masks into a single channel mask (sum over all instances)
        aggregated_mask = instance_masks.sum(axis=0, keepdims=True)

        # Normalize the aggregated mask to the range [0, 1]
        aggregated_mask = aggregated_mask / aggregated_mask.max()

        # Convert the aggregated mask to a tensor
        aggregated_mask_tensor = torch.FloatTensor(aggregated_mask)

        # Concatenate the aggregated mask to the original image as a new channel
        input_with_mask = torch.cat([input_image, aggregated_mask_tensor], dim=1)

        return input_with_mask

    def get_image_path_from_s3(self, idx):
        # Implement a method to retrieve the image path from your dataset structure
        # Here, you can use the s3_bucket and s3_prefix to construct the image path
        # Return the image path for the given index
        return f"{self.s3_prefix}/image_{idx}.jpg"

    def load_image_from_s3(self, image_path):
        # Download image from S3
        response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=image_path)
        img_bytes = BytesIO(response['Body'].read())
        img = Image.open(img_bytes).convert('RGB')
        return img

# Example usage
s3_bucket = 'your-s3-bucket-name'
s3_prefix = 'path/to/images'
panoptic_fpn_model = load_panoptic_fpn_model()
timm_model_name = 'resnet18'  # Replace with your preferred timm model
dataset = PanopticSegmentationDataset(s3_bucket, s3_prefix, panoptic_fpn_model, timm_model_name)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# Iterate over the dataloader
for batch in dataloader:
    # Now, 'batch' contains 4-channel images ready to be used as input to the timm model
    print("Batch shape:", batch.shape)

    # Perform further processing or use the batch as input to the timm model
    with torch.no_grad():
        output_timm_model = timm_model(batch)

    # Use the output as needed
    print("Timm Model Output:", output_timm_model)


##### BREAK 7 - detectron2

import torch
import torchvision.transforms as transforms
import timm
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo

# Function to load a pre-trained timm model with modified input channels
def load_pretrained_timm_model(model_name, input_channels=4):
    model = timm.create_model(model_name, pretrained=True)
    
    # Modify the first convolutional layer to accept 4 channels
    if hasattr(model, 'conv1'):
        model.conv1 = torch.nn.Conv2d(input_channels, model.conv1.out_channels, 
                                      kernel_size=model.conv1.kernel_size,
                                      stride=model.conv1.stride,
                                      padding=model.conv1.padding,
                                      bias=False)
    elif hasattr(model, 'features'):
        model.features[0] = torch.nn.Conv2d(input_channels, model.features[0].out_channels, 
                                           kernel_size=model.features[0].kernel_size,
                                           stride=model.features[0].stride,
                                           padding=model.features[0].padding,
                                           bias=False)
    else:
        raise ValueError("Unsupported model architecture. Modify the input channels manually.")

    return model

# Function to perform panoptic segmentation and concatenate the masks to the original image
def panoptic_segmentation_and_concat(image_path, timm_model_name):
    # Load and preprocess the original image
    original_image = Image.open(image_path).convert('RGB')

    # Transformations for the input image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust size as needed
        transforms.ToTensor(),
    ])

    input_image = transform(original_image).unsqueeze(0)  # Add batch dimension

    # Configure and load Detectron2 Panoptic FPN model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml")
    panoptic_predictor = DefaultPredictor(cfg)

    # Perform inference on the original image using Detectron2
    with torch.no_grad():
        panoptic_result = panoptic_predictor(original_image)

    # Extract panoptic segmentation masks
    panoptic_mask = panoptic_result["panoptic_seg"][0].numpy()

    # Normalize the panoptic mask to the range [0, 1]
    panoptic_mask = panoptic_mask / panoptic_mask.max()

    # Convert the panoptic mask to a tensor
    panoptic_mask_tensor = torch.FloatTensor(panoptic_mask)

    # Concatenate the panoptic mask to the original image as a new channel
    input_with_mask = torch.cat([input_image, panoptic_mask_tensor.unsqueeze(0)], dim=1)

    # Load a pre-trained timm model with modified input channels
    timm_model = load_pretrained_timm_model(timm_model_name, input_channels=4)

    # Forward pass through the timm model using the image with the concatenated mask
    with torch.no_grad():
        output_timm_model = timm_model(input_with_mask)

    # Use the output as needed
    print("Timm Model Output:", output_timm_model)

# Example usage
panoptic_segmentation_and_concat("example_image.jpg", "resnet50")


