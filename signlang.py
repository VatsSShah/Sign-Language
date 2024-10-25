import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import os
import json

class SignLanguageRecognition:
    def __init__(self, num_classes=5, sequence_length=30, num_sensors=5):
        """
        Initialize the Sign Language Recognition model with default values 
        for classes, sequence length, and number of sensors.
        """
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.num_sensors = num_sensors
        self.model = None
        self.scaler = StandardScaler()
        
        # Configure logging with more detailed format
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('sign_language_model.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Check TensorFlow version and GPU availability
        self.logger.info(f"TensorFlow version: {tf.__version__}")
        self.logger.info(f"GPU Available: {tf.test.is_built_with_cuda()}")
        if tf.test.is_built_with_cuda():
            self.logger.info(f"GPU Devices: {tf.config.list_physical_devices('GPU')}")

    def load_and_preprocess_data(self, X_data, y_data, test_size=0.2, validation_size=0.2):
        """
        Preprocess the data by scaling it and splitting it into training, validation, and test sets.
        """
        try:
            # Input validation
            if not isinstance(X_data, np.ndarray) or not isinstance(y_data, np.ndarray):
                raise ValueError("Input data must be numpy arrays")
            
            if len(X_data) != len(y_data):
                raise ValueError(f"X_data and y_data must have same length, got {len(X_data)} and {len(y_data)}")
            
            # Ensure data has the expected shape
            if X_data.shape[1:] != (self.sequence_length, self.num_sensors):
                raise ValueError(f"Expected shape (samples, {self.sequence_length}, {self.num_sensors}), got {X_data.shape}")
            
            # Save mean and std for later use
            self.data_stats = {
                'mean': np.mean(X_data),
                'std': np.std(X_data)
            }
            
            # Reshape and scale the data
            original_shape = X_data.shape
            X_data_reshaped = X_data.reshape(-1, self.num_sensors)
            X_data_scaled = self.scaler.fit_transform(X_data_reshaped)
            X_data = X_data_scaled.reshape(original_shape)
            
            # Convert labels to one-hot encoding
            y_data = tf.keras.utils.to_categorical(y_data, num_classes=self.num_classes)
            
            # Split the data
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X_data, y_data, test_size=test_size, random_state=42, 
                stratify=y_data.argmax(axis=1)
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=validation_size, 
                random_state=42, stratify=y_train_val.argmax(axis=1)
            )
            
            # Log dataset information
            self.logger.info(f"Training set shape: {X_train.shape}")
            self.logger.info(f"Validation set shape: {X_val.shape}")
            self.logger.info(f"Test set shape: {X_test.shape}")
            self.logger.info(f"Class distribution in training set: {np.sum(y_train, axis=0)}")
            
            return (X_train, y_train), (X_val, y_val), (X_test, y_test)
            
        except Exception as e:
            self.logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def build_model(self, lstm_units=64, dense_units=32, dropout_rate=0.3):
        """
        Construct the LSTM-based model with improved architecture.
        """
        try:
            # Enable mixed precision for better performance
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            model = Sequential([
                GaussianNoise(0.1, input_shape=(self.sequence_length, self.num_sensors)),
                
                # First LSTM layer
                LSTM(lstm_units, return_sequences=True, 
                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                BatchNormalization(),
                Dropout(dropout_rate),
                
                # Second LSTM layer
                LSTM(lstm_units, return_sequences=False,
                     kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                BatchNormalization(),
                Dropout(dropout_rate),
                
                # Dense layers
                Dense(dense_units, activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                BatchNormalization(),
                Dropout(dropout_rate),
                
                Dense(self.num_classes, activation='softmax')
            ])
            
            # Use gradient clipping in optimizer
            optimizer = Adam(learning_rate=0.001, clipvalue=1.0)
            
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['categorical_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            self.model = model
            self.logger.info(model.summary())
            
        except Exception as e:
            self.logger.error(f"Error in model building: {str(e)}")
            raise

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the model with improved monitoring and callbacks.
        """
        try:
            class_weights = self.compute_class_weights(y_train)
            
            # Create callbacks directory if it doesn't exist
            os.makedirs('model_checkpoints', exist_ok=True)
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    'model_checkpoints/best_model_{epoch:02d}_{val_categorical_accuracy:.4f}.h5',
                    monitor='val_categorical_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.0001,
                    verbose=1
                ),
                tf.keras.callbacks.TensorBoard(log_dir='./logs')
            ]
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            # Save training history
            with open('training_history.json', 'w') as f:
                json.dump(history.history, f)
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise

    def predict(self, X_test):
        """
        Predict classes with confidence scores.
        """
        try:
            # Scale the test data
            original_shape = X_test.shape
            X_test_reshaped = X_test.reshape(-1, self.num_sensors)
            X_test_scaled = self.scaler.transform(X_test_reshaped)
            X_test = X_test_scaled.reshape(original_shape)
            
            # Generate predictions with confidence scores
            predictions = self.model.predict(X_test)
            predicted_classes = np.argmax(predictions, axis=1)
            confidence_scores = np.max(predictions, axis=1)
            
            return predictions, predicted_classes, confidence_scores
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise

    def save_model(self, filepath):
        """Save the model and all configurations."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            # Save model
            self.model.save(filepath)
            
            # Save configurations
            config = {
                'num_classes': self.num_classes,
                'sequence_length': self.sequence_length,
                'num_sensors': self.num_sensors,
                'scaler_params': self.scaler.get_params(),
                'data_stats': self.data_stats
            }
            
            with open(f"{filepath}_config.json", 'w') as f:
                json.dump(config, f)
            
            self.logger.info(f"Model and configurations saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    @classmethod
    def load_model(cls, filepath):
        """Load a previously saved model and configurations."""
        try:
            # Load configurations
            with open(f"{filepath}_config.json", 'r') as f:
                config = json.load(f)
            
            # Create instance with saved configurations
            instance = cls(
                num_classes=config['num_classes'],
                sequence_length=config['sequence_length'],
                num_sensors=config['num_sensors']
            )
            
            # Load model and scaler
            instance.model = load_model(filepath)
            instance.scaler.set_params(**config['scaler_params'])
            instance.data_stats = config['data_stats']
            
            return instance
            
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Generate sample data
        num_samples = 1000
        sequence_length = 30
        num_sensors = 5
        num_classes = 5
        
        X_data = np.random.rand(num_samples, sequence_length, num_sensors)
        y_data = np.random.randint(0, num_classes, num_samples)
        
        # Initialize and train model
        model = SignLanguageRecognition(num_classes=num_classes, 
                                      sequence_length=sequence_length, 
                                      num_sensors=num_sensors)
        
        # Train model
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = \
            model.load_and_preprocess_data(X_data, y_data)
        
        model.build_model()
        history = model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        predictions, predicted_classes, confidence_scores = model.predict(X_test)
        accuracy = np.mean(predicted_classes == np.argmax(y_test, axis=1))
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Save model
        model.save_model('models/sign_language_model.h5')
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise
