"""Check model details"""
from ultralytics import YOLO

model = YOLO('models/best.pt')

print("=" * 50)
print("MODEL INFORMATION")
print("=" * 50)
print(f"Task: {model.task}")
print(f"Model Type: {type(model.model).__name__}")
print(f"Number of Classes: {len(model.names)}")
print("\nClass Names:")
for idx, name in model.names.items():
    print(f"  {idx}: {name}")
print("=" * 50)
