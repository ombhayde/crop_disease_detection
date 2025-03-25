# src/main.py
import argparse
import os
from utils import download_dataset
from data_preparation import process_dataset, create_data_generators
from train import train_model
from evaluate import evaluate_model, predict_disease

def main():
    parser = argparse.ArgumentParser(description='Crop Disease Detection')
    parser.add_argument('--mode', type=str, default='train', choices=['download', 'process', 'train', 'evaluate', 'predict'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--image_path', type=str, help='Path to image for prediction')
    args = parser.parse_args()
    
    if args.mode == 'download':
        download_dataset()
    
    elif args.mode == 'process':
        process_dataset()
    
    elif args.mode == 'train':
        train_model(epochs=args.epochs, batch_size=args.batch_size, fine_tune=args.fine_tune)
    
    elif args.mode == 'evaluate':
        evaluate_model(
            model_path='models/saved_models/crop_disease_model.keras',
            test_data_dir='data/processed/test',
            class_names_path='models/saved_models/class_names.txt',
            batch_size=args.batch_size
        )
    
    elif args.mode == 'predict':
        if args.image_path and os.path.exists(args.image_path):
            predicted_class, confidence, top_3 = predict_disease(
                model_path='models/saved_models/crop_disease_model.h5',
                img_path=args.image_path,
                class_names_path='models/saved_models/class_names.txt'
            )
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")
            print("Top 3 predictions:")
            for cls, conf in top_3:
                print(f"  {cls}: {conf:.4f}")
        else:
            print("Please provide a valid image path using --image_path")

if __name__ == "__main__":
    main()