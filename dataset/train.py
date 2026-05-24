import wandb
from ultralytics import YOLO

def main():
    # 1. Initialize MLOps tracking explicitly
    wandb.init(
        project="defect-detection-api",
        job_type="model-training",
        name="yolov8n-baseline"
    )

    # 2. Load the lightweight YOLO architecture
    # We use the 'nano' (.pt) version for maximum FPS in the backend API
    model = YOLO("yolov8n.pt") 

    # 3. Execute training with production constraints
    results = model.train(
        data="data.yaml",
        epochs=100,           
        patience=15,          
        batch=16,             
        imgsz=640,            
        save=True,            
        save_period=10,       
        project="defect-detection-api", 
        exist_ok=True,        
        device="0"            # Use 'cpu' if you lack an Nvidia GPU, or 'mps' for Mac silicon
    )

    # 4. Safely close the MLOps stream
    wandb.finish()

if __name__ == "__main__":
    main()