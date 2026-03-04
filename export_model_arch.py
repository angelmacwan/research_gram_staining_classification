# this will be used to visualise the model architecture and use it for the documentation

import torch
import timm
import torch.nn as nn

# Load VOLO model from timm
print("Loading VOLO model...")
model = timm.create_model('volo_d5_512', pretrained=False, num_classes=4)
print(f"Model loaded: {type(model)}")

# Move model to eval mode
model.eval()

# Create a dummy input for the model
# VOLO expects images of size 512 (based on the model name)
dummy_input = torch.randn(1, 3, 512, 512)

print("\nExporting model to ONNX format...")

# Export the model to ONNX format (compatible with Netron)
output_path = 'volo_architecture.onnx'
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=['input_image'],
    output_names=['predictions'],
    opset_version=14,
    dynamic_axes={'input_image': {0: 'batch_size'}, 'predictions': {0: 'batch_size'}}
)
print(f"Model exported to ONNX format: {output_path}")
print(f"\nTo visualize the model, open {output_path} with Netron:")
print("  - Online: https://netron.app")
print("  - Or install locally: pip install netron")

# Also print model summary
print("\nModel Summary:")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

