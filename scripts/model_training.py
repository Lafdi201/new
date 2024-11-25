import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import cv2
from scripts.data_preprocessing import load_reference_image
import os
import json
from typing import List, Tuple, Dict

def build_siamese_model(input_shape):
    with tf.device('/CPU:0'):
        # Create the base network to be shared (identical for both towers)
        def create_base_network():
            base_model = ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape,
                pooling='avg'
            )
            
            for layer in base_model.layers[:100]:
                layer.trainable = False
            
            x = base_model.output
            x = layers.BatchNormalization()(x)
            x = layers.Dense(512, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dense(128, activation='relu')(x)
            return Model(inputs=base_model.input, outputs=x)
        
        # Create the base network
        base_network = create_base_network()
        
        # Create the Siamese Network
        input_a = layers.Input(shape=input_shape)
        input_b = layers.Input(shape=input_shape)
        
        # Process both inputs through the same network
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        
        # Calculate L1 distance between the two encoded outputs
        L1_distance = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([processed_a, processed_b])
        
        # Add a dense layer for final prediction
        prediction = layers.Dense(1, activation='sigmoid')(L1_distance)
        
        # Create the final model
        siamese_model = Model(inputs=[input_a, input_b], outputs=prediction)
        
        return siamese_model

def train_model(image):
    tf.keras.backend.clear_session()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    with tf.device('/CPU:0'):
        model = build_siamese_model((256, 256, 3))
        
        optimizer = optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=True
        )

        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        # Generate training pairs
        pairs, pair_labels = create_pairs(image, n_pairs=4000)
        pairs = pairs.astype('float32')
        pair_labels = pair_labels.astype('float32')

        # Manual train-validation split
        indices = np.random.permutation(len(pair_labels))
        validation_split = 0.2
        num_validation = int(len(pair_labels) * validation_split)
        
        train_idx = indices[num_validation:]
        val_idx = indices[:num_validation]

        # Split data into train and validation sets
        train_pairs = pairs[train_idx]
        train_labels = pair_labels[train_idx]

        val_pairs = pairs[val_idx]
        val_labels = pair_labels[val_idx]

        callbacks = [
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/checkpoint.h5',
                save_best_only=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.TerminateOnNaN()
        ]

        def data_generator(pairs, labels, batch_size):
            num_samples = len(labels)
            while True:
                indices = np.random.permutation(num_samples)
                for i in range(0, num_samples, batch_size):
                    batch_indices = indices[i:i + batch_size]
                    batch_pairs = pairs[batch_indices]
                    batch_labels = labels[batch_indices]
                    # Split pairs into inputs
                    yield [batch_pairs[:, 0], batch_pairs[:, 1]], batch_labels

        batch_size = 32
        train_steps = len(train_labels) // batch_size
        val_steps = len(val_labels) // batch_size

        # Create train and validation generators
        train_gen = data_generator(train_pairs, train_labels, batch_size)
        val_gen = data_generator(val_pairs, val_labels, batch_size)

        # Train using both generators
        history = model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=50,
            callbacks=callbacks,
            verbose=1
        )

        model.save_weights('models/siamese_model.weights.h5')
        return history

def create_pairs(image, n_pairs=4000):
    if len(image.shape) == 4:
        image = image[0]
        
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.85, 1.15]
    )

    batch_size = 200
    pairs = []
    labels = []
    
    for i in range(0, n_pairs, batch_size):
        current_batch_size = min(batch_size, n_pairs - i)
        
        # Generate similar pairs
        aug_params_similar = [datagen.get_random_transform(image.shape) 
                            for _ in range(current_batch_size)]
        aug_images_similar = []
        for params in aug_params_similar:
            aug_images_similar.append(datagen.apply_transform(image.copy(), params))
        
        # Generate dissimilar pairs
        aug_params_dissimilar = [datagen.get_random_transform(image.shape) 
                                for _ in range(current_batch_size)]
        aug_images_dissimilar = []
        for params in aug_params_dissimilar:
            aug_images_dissimilar.append(datagen.apply_transform(image.copy(), params))
        
        pairs.extend([[image, aug_image] for aug_image in aug_images_similar])
        labels.extend([1] * current_batch_size)
        
        pairs.extend([[image, aug_image] for aug_image in aug_images_dissimilar])
        labels.extend([0] * current_batch_size)

        del aug_images_similar
        del aug_images_dissimilar

    return np.array(pairs), np.array(labels)