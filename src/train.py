# src/train.py
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
import time
from datetime import datetime

from data_preparation import process_dataset, create_data_generators
from model import create_model

def train_model(epochs=30, batch_size=32, fine_tune=True):
    """
    Train the crop disease detection model
    """
    # Process dataset if not already done
    if not os.path.exists('data/processed/train'):
        print("Processing dataset...")
        process_dataset()
    
    # Create data generators
    train_generator, valid_generator, test_generator = create_data_generators(batch_size)
    
    # Get number of classes
    num_classes = len(train_generator.class_indices)
    class_names = list(train_generator.class_indices.keys())
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Create model
    model, base_model = create_model(num_classes)
    
    # Create checkpoint directory
    checkpoint_dir = 'models/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup callbacks
    checkpoint = ModelCheckpoint(
        f'{checkpoint_dir}/model_best.keras',  # Changed extension to .keras
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Train the model (transfer learning)
    print("Starting transfer learning phase...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=valid_generator,
        validation_steps=valid_generator.samples // batch_size,
        epochs=10,
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard]
    )
    
    # Fine-tuning (optional)
    if fine_tune:
        print("Starting fine-tuning phase...")
        # Unfreeze the top layers of the base model
        for layer in base_model.layers[-30:]:
            layer.trainable = True
            
        # Recompile the model with a lower learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Continue training
        fine_tune_history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=valid_generator,
            validation_steps=valid_generator.samples // batch_size,
            epochs=epochs,
            initial_epoch=history.epoch[-1],
            callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard]
        )
        
        # Combine histories
        for k in fine_tune_history.history:
            history.history[k].extend(fine_tune_history.history[k])
    
    # Save the final model
    model.save('models/saved_models/crop_disease_model.keras')
    
    # Save class names
    with open('models/saved_models/class_names.txt', 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    
    # Evaluate on test set
    test_results = model.evaluate(test_generator)
    print(f"Test loss: {test_results[0]:.4f}")
    print(f"Test accuracy: {test_results[1]:.4f}")
    
    return model, history

if __name__ == "__main__":
    train_model()