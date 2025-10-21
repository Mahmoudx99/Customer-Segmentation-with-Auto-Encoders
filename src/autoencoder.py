"""
Auto Encoder model for customer segmentation.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt


class CustomerAutoEncoder:
    """
    Auto Encoder for learning compressed customer representations.
    """

    def __init__(self, input_dim, encoding_dim=8, hidden_layers=[16, 12]):
        """
        Initialize the Auto Encoder.

        Args:
            input_dim (int): Number of input features
            encoding_dim (int): Dimension of the encoded representation
            hidden_layers (list): List of hidden layer sizes
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.history = None

        self._build_model()

    def _build_model(self):
        """
        Build the Auto Encoder architecture.
        """
        # Input layer
        input_layer = layers.Input(shape=(self.input_dim,), name='input')

        # Encoder
        encoded = input_layer
        for i, units in enumerate(self.hidden_layers):
            encoded = layers.Dense(units, activation='relu',
                                 name=f'encoder_dense_{i+1}')(encoded)
            encoded = layers.Dropout(0.2, name=f'encoder_dropout_{i+1}')(encoded)

        # Bottleneck layer (encoded representation)
        encoded = layers.Dense(self.encoding_dim, activation='relu',
                             name='bottleneck')(encoded)

        # Decoder (mirror of encoder)
        decoded = encoded
        for i, units in enumerate(reversed(self.hidden_layers)):
            decoded = layers.Dense(units, activation='relu',
                                 name=f'decoder_dense_{i+1}')(decoded)
            decoded = layers.Dropout(0.2, name=f'decoder_dropout_{i+1}')(decoded)

        # Output layer (reconstruction)
        decoded = layers.Dense(self.input_dim, activation='linear',
                             name='output')(decoded)

        # Create models
        self.autoencoder = Model(inputs=input_layer, outputs=decoded, name='autoencoder')
        self.encoder = Model(inputs=input_layer, outputs=encoded, name='encoder')

        # Compile the autoencoder
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

    def train(self, X_train, X_val=None, epochs=100, batch_size=32, verbose=1):
        """
        Train the Auto Encoder.

        Args:
            X_train (np.ndarray): Training data
            X_val (np.ndarray): Validation data (optional)
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            verbose (int): Verbosity mode

        Returns:
            History: Training history
        """
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Train the autoencoder
        validation_data = (X_val, X_val) if X_val is not None else None

        self.history = self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )

        return self.history

    def encode(self, X):
        """
        Encode data using the trained encoder.

        Args:
            X (np.ndarray): Data to encode

        Returns:
            np.ndarray: Encoded representations
        """
        return self.encoder.predict(X, verbose=0)

    def decode(self, encoded_data):
        """
        Decode encoded representations.

        Args:
            encoded_data (np.ndarray): Encoded data

        Returns:
            np.ndarray: Decoded representations
        """
        # Create a decoder model from bottleneck to output
        bottleneck_input = layers.Input(shape=(self.encoding_dim,))

        # Get decoder layers from autoencoder
        decoder_layers = [layer for layer in self.autoencoder.layers
                         if 'decoder' in layer.name or layer.name == 'output']

        # Build decoder
        x = bottleneck_input
        bottleneck_output = self.autoencoder.get_layer('bottleneck').output

        # Find where decoder starts
        found_bottleneck = False
        for layer in self.autoencoder.layers:
            if found_bottleneck and 'decoder' in layer.name or layer.name == 'output':
                x = layer(x)
            if layer.name == 'bottleneck':
                found_bottleneck = True

        decoder_model = Model(inputs=bottleneck_input, outputs=x)
        return decoder_model.predict(encoded_data, verbose=0)

    def reconstruct(self, X):
        """
        Reconstruct input data.

        Args:
            X (np.ndarray): Input data

        Returns:
            np.ndarray: Reconstructed data
        """
        return self.autoencoder.predict(X, verbose=0)

    def get_reconstruction_error(self, X):
        """
        Calculate reconstruction error.

        Args:
            X (np.ndarray): Input data

        Returns:
            np.ndarray: Reconstruction errors
        """
        reconstructed = self.reconstruct(X)
        mse = np.mean((X - reconstructed) ** 2, axis=1)
        return mse

    def plot_training_history(self, save_path=None):
        """
        Plot training history.

        Args:
            save_path (str): Path to save the plot (optional)
        """
        if self.history is None:
            print("Model has not been trained yet.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAE plot
        axes[1].plot(self.history.history['mae'], label='Training MAE', linewidth=2)
        if 'val_mae' in self.history.history:
            axes[1].plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2)
        axes[1].set_title('Model MAE', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")

        plt.show()

    def summary(self):
        """
        Print model summary.
        """
        print("="*60)
        print("AUTO ENCODER ARCHITECTURE")
        print("="*60)
        self.autoencoder.summary()
        print("\n" + "="*60)
        print("ENCODER ARCHITECTURE")
        print("="*60)
        self.encoder.summary()

    def save(self, filepath):
        """
        Save the autoencoder model.

        Args:
            filepath (str): Path to save the model
        """
        self.autoencoder.save(filepath)
        print(f"Autoencoder model saved to {filepath}")

    @staticmethod
    def load(filepath):
        """
        Load a saved autoencoder model.

        Args:
            filepath (str): Path to the saved model

        Returns:
            CustomerAutoEncoder: Loaded model
        """
        loaded_autoencoder = keras.models.load_model(filepath)

        # Recreate the CustomerAutoEncoder object
        input_dim = loaded_autoencoder.input_shape[1]
        encoding_dim = loaded_autoencoder.get_layer('bottleneck').output_shape[1]

        # Infer hidden layers from model
        hidden_layers = []
        for layer in loaded_autoencoder.layers:
            if 'encoder_dense' in layer.name:
                hidden_layers.append(layer.units)

        model = CustomerAutoEncoder(input_dim, encoding_dim, hidden_layers)
        model.autoencoder = loaded_autoencoder

        # Recreate encoder
        input_layer = loaded_autoencoder.input
        bottleneck_layer = loaded_autoencoder.get_layer('bottleneck').output
        model.encoder = Model(inputs=input_layer, outputs=bottleneck_layer, name='encoder')

        return model
