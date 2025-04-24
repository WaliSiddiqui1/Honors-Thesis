import os
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, constraints
from tensorflow.keras.optimizers import Adam

# Directory settings - pointing to output from previous classification step
DATA_DIR = "cloud_data1"
OUTPUT_DIR = os.path.join(DATA_DIR, "generated1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load paired data created by the previous optimization script
print("Loading paired data for GAN training...")
with open(os.path.join(DATA_DIR, "paired_data_optimized.json"), 'r') as f:
    paired_data = json.load(f)

print(f"Found {len(paired_data)} paired images")

# Function to load and normalize paired images
def load_paired_images(pairs, max_pairs=None):
    """Load geographically matched pairs of cloudy and clear images
    
    This function handles loading the paired image data created in previous steps,
    ensuring proper normalization and consistent shapes for training.
    """
    if max_pairs and max_pairs < len(pairs):
        # Use only a subset if requested - helps manage memory constraints
        pairs = pairs[:max_pairs]
        
    cloudy_images = []
    clear_images = []
    
    for i, pair in enumerate(pairs):
        # Load cloudy image metadata and file path
        with open(os.path.join(DATA_DIR, "metadata", f"{pair['cloudy_id']}.json"), 'r') as f:
            cloudy_meta = json.load(f)
        cloudy_path = cloudy_meta['img_path']
        
        # Load clear image metadata and file path
        with open(os.path.join(DATA_DIR, "metadata", f"{pair['clear_id']}.json"), 'r') as f:
            clear_meta = json.load(f)
        clear_path = clear_meta['img_path']
        
        # Load numpy arrays containing the actual image data
        cloudy_img = np.load(cloudy_path)
        clear_img = np.load(clear_path)
        
        # Quality control: ensure both images have the same shape
        if cloudy_img.shape != clear_img.shape:
            print(f"Warning: Shape mismatch in pair {i}. Skipping.")
            continue
            
        # Robust normalization to [0,1] range using percentile clipping
        # This helps handle outliers and ensures consistent value ranges
        for img in [cloudy_img, clear_img]:
            for c in range(img.shape[-1]):
                channel = img[:,:,c]
                if np.max(channel) > 1.0 or np.min(channel) < 0.0:
                    p1, p99 = np.percentile(channel, (1, 99))
                    channel = np.clip(channel, p1, p99)
                    channel = (channel - p1) / (p99 - p1 + 1e-8)
                    img[:,:,c] = channel
                    
        cloudy_images.append(cloudy_img)
        clear_images.append(clear_img)
        
        # Progress update to monitor loading process
        if (i+1) % 100 == 0:
            print(f"Loaded {i+1}/{len(pairs)} pairs")
            
    return np.array(cloudy_images), np.array(clear_images)

# Load paired data with memory-safe limit
print("Loading and preparing paired images...")
cloudy, clear = load_paired_images(paired_data, max_pairs=2000)  # Adjust this limit based on available RAM

print(f"Loaded data: cloudy {cloudy.shape}, clear {clear.shape}")

# Visualize sample pairs to verify proper loading and normalization
plt.figure(figsize=(12, 6))
for i in range(min(5, len(cloudy))):
    plt.subplot(2, 5, i+1)
    plt.imshow(np.clip(cloudy[i], 0, 1))
    plt.title(f"Cloudy {i}")
    plt.axis('off')
    
    plt.subplot(2, 5, i+6)
    plt.imshow(np.clip(clear[i], 0, 1))
    plt.title(f"Clear {i}")
    plt.axis('off')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/sample_pairs.png")
plt.close()
print(f"✓ Saved sample paired data visualization")

# Generator (Improved U-Net architecture)
def build_generator(input_shape):
    """
    Build a U-Net style generator for the cloud removal task
    
    This architecture uses skip connections between encoder and decoder
    to preserve spatial information important for reconstruction.
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder - progressively downsamples the image while increasing feature depth
    x1 = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.LeakyReLU(0.2)(x1)
    
    x2 = layers.Conv2D(128, 4, strides=2, padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.LeakyReLU(0.2)(x2)
    
    x3 = layers.Conv2D(256, 4, strides=2, padding='same')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.LeakyReLU(0.2)(x3)

    # Bottleneck - deepest layer with highest feature abstraction
    b = layers.Conv2D(512, 4, strides=2, padding='same')(x3)
    b = layers.BatchNormalization()(b)
    b = layers.LeakyReLU(0.2)(b)

    # Decoder - progressively upsamples while incorporating encoder features via skip connections
    d3 = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(b)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.Dropout(0.5)(d3)  # Dropout for regularization in early upsampling layers
    d3 = layers.ReLU()(d3)
    d3 = layers.Concatenate()([d3, x3])  # Skip connection from encoder
    
    d2 = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(d3)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.ReLU()(d2)
    d2 = layers.Concatenate()([d2, x2])  # Skip connection from encoder
    
    d1 = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(d2)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.ReLU()(d1)
    d1 = layers.Concatenate()([d1, x1])  # Skip connection from encoder
    
    d0 = layers.Conv2DTranspose(32, 4, strides=2, padding='same')(d1)
    d0 = layers.BatchNormalization()(d0)
    d0 = layers.ReLU()(d0)

    # Output layer with sigmoid activation to ensure proper [0,1] range
    outputs = layers.Conv2D(input_shape[-1], 3, padding='same', activation='sigmoid')(d0)
    
    return Model(inputs, outputs, name="Generator")

# Discriminator with stability enhancements
def build_discriminator(input_shape):
    """
    Build a discriminator network with enhanced stability features
    
    Uses kernel constraints to prevent exploding gradients during adversarial training.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Use kernel constraint for training stability
    kernel_constraint = constraints.MaxNorm(1.0)
    
    x = layers.Conv2D(64, 4, strides=2, padding='same', 
                     kernel_constraint=kernel_constraint)(inputs)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(128, 4, strides=2, padding='same', 
                     kernel_constraint=kernel_constraint)(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(256, 4, strides=2, padding='same', 
                     kernel_constraint=kernel_constraint)(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv2D(512, 4, strides=2, padding='same', 
                     kernel_constraint=kernel_constraint)(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1, kernel_constraint=kernel_constraint)(x)
    
    return Model(inputs, x, name="Discriminator")

# Build models with input shape derived from loaded data
input_shape = cloudy.shape[1:]
generator = build_generator(input_shape)
discriminator = build_discriminator(input_shape)

# Print model summaries for architecture verification
generator.summary()
discriminator.summary()

# Custom GAN training class with advanced metrics and visualization
class CloudRemovalGAN:
    """
    Cloud removal GAN implementation with comprehensive training, 
    evaluation, and visualization capabilities
    """
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        
        # Optimizers with parameters tuned for GAN stability
        self.g_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        self.d_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        
        # Loss and metrics tracking
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")
        self.l1_loss_tracker = tf.keras.metrics.Mean(name="l1_loss")
        self.psnr_metric = tf.keras.metrics.Mean(name="psnr")
        self.ssim_metric = tf.keras.metrics.Mean(name="ssim")
        
        # Binary cross entropy loss for adversarial component
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        # Weight for L1 loss - balances pixel reconstruction vs. adversarial objectives
        self.lambda_l1 = 100.0
    
    @tf.function
    def train_discriminator_step(self, cloudy_images, clear_images):
        """Single training step for discriminator, optimized with TF graph execution"""
        # Generate fake images
        fake_images = self.generator(cloudy_images, training=True)
        
        # Get discriminator loss
        with tf.GradientTape() as disc_tape:
            # Real images
            real_output = self.discriminator(clear_images, training=True)
            # Fake images
            fake_output = self.discriminator(fake_images, training=True)
            
            # Discriminator losses - should correctly classify real vs fake
            real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss
        
        # Apply gradients
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return disc_loss
    
    @tf.function
    def train_generator_step(self, cloudy_images, clear_images):
        """Single training step for generator, optimized with TF graph execution"""
        with tf.GradientTape() as gen_tape:
            # Generate fake images
            fake_images = self.generator(cloudy_images, training=True)
            
            # Discriminator output on fake images
            fake_output = self.discriminator(fake_images, training=False)
            
            # Generator adversarial loss - wants discriminator to think its images are real
            gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            
            # L1 loss - pixel-wise absolute difference for content preservation
            l1_loss = tf.reduce_mean(tf.abs(clear_images - fake_images))
            
            # Combined loss with weighted L1 component
            total_gen_loss = gen_loss + self.lambda_l1 * l1_loss
        
        # Apply gradients
        gen_gradients = gen_tape.gradient(total_gen_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        
        # Calculate image quality metrics for monitoring
        psnr = tf.reduce_mean(tf.image.psnr(clear_images, fake_images, max_val=1.0))
        ssim = tf.reduce_mean(tf.image.ssim(clear_images, fake_images, max_val=1.0))
        
        return gen_loss, l1_loss, psnr, ssim
    
    def train(self, cloudy_images, clear_images, epochs=100, batch_size=8):
        """Train the GAN with comprehensive progress tracking and visualization"""
        dataset_size = len(cloudy_images)
        steps_per_epoch = dataset_size // batch_size
        
        # Create TensorFlow dataset for efficient batching and shuffling
        dataset = tf.data.Dataset.from_tensor_slices((cloudy_images, clear_images))
        dataset = dataset.shuffle(buffer_size=dataset_size).batch(batch_size)
        
        # Sample images for periodic visualization of progress
        test_cloudy = cloudy_images[:5]
        test_clear = clear_images[:5]
        
        # Training history for plotting learning curves
        history = {
            'gen_loss': [],
            'disc_loss': [],
            'l1_loss': [],
            'psnr': [],
            'ssim': []
        }
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Reset metrics for new epoch
            self.gen_loss_tracker.reset_states()
            self.disc_loss_tracker.reset_states()
            self.l1_loss_tracker.reset_states()
            self.psnr_metric.reset_states()
            self.ssim_metric.reset_states()
            
            for batch_idx, (cloudy_batch, clear_batch) in enumerate(dataset):
                # Train discriminator
                disc_loss = self.train_discriminator_step(cloudy_batch, clear_batch)
                
                # Train generator (twice for better balance, common practice in GANs)
                gen_loss, l1_loss, psnr, ssim = self.train_generator_step(cloudy_batch, clear_batch)
                gen_loss2, l1_loss2, psnr2, ssim2 = self.train_generator_step(cloudy_batch, clear_batch)
                
                # Update metrics with batch results
                self.disc_loss_tracker.update_state(disc_loss)
                self.gen_loss_tracker.update_state((gen_loss + gen_loss2) / 2)
                self.l1_loss_tracker.update_state((l1_loss + l1_loss2) / 2)
                self.psnr_metric.update_state((psnr + psnr2) / 2)
                self.ssim_metric.update_state((ssim + ssim2) / 2)
                
                # Print progress periodically
                if (batch_idx + 1) % 20 == 0:
                    print(f"  Batch {batch_idx+1}/{steps_per_epoch} - "
                          f"D loss: {self.disc_loss_tracker.result():.4f}, "
                          f"G loss: {self.gen_loss_tracker.result():.4f}, "
                          f"L1: {self.l1_loss_tracker.result():.4f}, "
                          f"PSNR: {self.psnr_metric.result():.2f}, "
                          f"SSIM: {self.ssim_metric.result():.4f}")
            
            # Record history at end of epoch
            history['gen_loss'].append(self.gen_loss_tracker.result().numpy())
            history['disc_loss'].append(self.disc_loss_tracker.result().numpy())
            history['l1_loss'].append(self.l1_loss_tracker.result().numpy())
            history['psnr'].append(self.psnr_metric.result().numpy())
            history['ssim'].append(self.ssim_metric.result().numpy())
            
            # Visualize generator progress every few epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.visualize_progress(test_cloudy, test_clear, epoch + 1)
            
            # Save model checkpoints periodically
            if (epoch + 1) % 10 == 0:
                self.generator.save(f"{OUTPUT_DIR}/generator_epoch_{epoch+1}.h5")
                print(f"✓ Saved generator checkpoint at epoch {epoch+1}")
        
        # Save final models for future use
        self.generator.save(f"{OUTPUT_DIR}/cloud_removal_generator.h5")
        self.discriminator.save(f"{OUTPUT_DIR}/cloud_removal_discriminator.h5")
        
        # Also save in the root directory for the restoration script
        self.generator.save("cloud_removal_generator.h5")
        
        # Plot training history
        self.plot_history(history)
        
        return history
    
    def visualize_progress(self, test_images, ground_truth_images, epoch):
        """Generate and save example output from the generator with ground truth comparison
        
        Creates visual comparisons showing the model's current cloud removal capabilities
        """
        generated_images = self.generator.predict(test_images)
        
        plt.figure(figsize=(15, 5 * len(test_images)))
        for i in range(len(test_images)):
            # Original cloudy image
            plt.subplot(len(test_images), 3, i*3 + 1)
            plt.imshow(np.clip(test_images[i], 0, 1))
            plt.title(f"Cloudy Input {i+1}")
            plt.axis('off')
            
            # Generated clear image
            plt.subplot(len(test_images), 3, i*3 + 2)
            plt.imshow(np.clip(generated_images[i], 0, 1))
            plt.title(f"Generated Image {i+1}")
            plt.axis('off')
            
            # Ground truth clear image
            plt.subplot(len(test_images), 3, i*3 + 3)
            plt.imshow(np.clip(ground_truth_images[i], 0, 1))
            plt.title(f"Ground Truth {i+1}")
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/progress_epoch_{epoch}.png")
        plt.close()
        
        # Also save a visualization of the differences
        self.visualize_differences(test_images, generated_images, ground_truth_images, epoch)
    
    def visualize_differences(self, cloudy_images, generated_images, ground_truth_images, epoch):
        """Visualize pixel-wise differences between generated and ground truth images
        
        Creates heatmaps showing where the model's predictions differ from ground truth
        """
        plt.figure(figsize=(15, 5 * len(cloudy_images)))
        
        for i in range(len(cloudy_images)):
            # Difference map (absolute difference between generated and ground truth)
            diff = np.abs(generated_images[i] - ground_truth_images[i])
            
            # Calculate error metrics for this specific image
            mae = np.mean(diff)
            mse = np.mean(diff**2)
            psnr = tf.image.psnr(
                tf.convert_to_tensor(ground_truth_images[i:i+1]), 
                tf.convert_to_tensor(generated_images[i:i+1]), 
                max_val=1.0
            ).numpy()[0]
            ssim = tf.image.ssim(
                tf.convert_to_tensor(ground_truth_images[i:i+1]), 
                tf.convert_to_tensor(generated_images[i:i+1]), 
                max_val=1.0
            ).numpy()[0]
            
            # Original cloudy image
            plt.subplot(len(cloudy_images), 3, i*3 + 1)
            plt.imshow(np.clip(cloudy_images[i], 0, 1))
            plt.title(f"Cloudy Input {i+1}")
            plt.axis('off')
            
            # Difference map with heatmap colorscale
            plt.subplot(len(cloudy_images), 3, i*3 + 2)
            plt.imshow(np.mean(diff, axis=2), cmap='hot', vmin=0, vmax=0.5)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title(f"Difference Map {i+1}")
            plt.axis('off')
            
            # Side-by-side comparison
            plt.subplot(len(cloudy_images), 3, i*3 + 3)
            # Create a side-by-side comparison for easy visual assessment
            comparison = np.hstack((generated_images[i], ground_truth_images[i]))
            plt.imshow(np.clip(comparison, 0, 1))
            plt.title(f"Gen vs Truth | MAE: {mae:.4f}, PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/difference_epoch_{epoch}.png")
        plt.close()
    
    def plot_history(self, history):
        """Plot and save training history metrics over time
        
        Creates a comprehensive visualization of how model performance evolved during training
        """
        epochs = range(1, len(history['gen_loss']) + 1)
        
        plt.figure(figsize=(15, 10))
        
        # Plot adversarial losses
        plt.subplot(2, 2, 1)
        plt.plot(epochs, history['gen_loss'], 'b-', label='Generator Loss')
        plt.plot(epochs, history['disc_loss'], 'r-', label='Discriminator Loss')
        plt.title('GAN Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot L1 reconstruction loss
        plt.subplot(2, 2, 2)
        plt.plot(epochs, history['l1_loss'], 'g-')
        plt.title('L1 Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # Plot PSNR (Peak Signal-to-Noise Ratio)
        plt.subplot(2, 2, 3)
        plt.plot(epochs, history['psnr'], 'm-')
        plt.title('PSNR (Higher is better)')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        
        # Plot SSIM (Structural Similarity Index)
        plt.subplot(2, 2, 4)
        plt.plot(epochs, history['ssim'], 'c-')
        plt.title('SSIM (Higher is better)')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/training_history.png")
        plt.close()
        
    def evaluate_model(self, test_cloudy, test_clear):
        """Evaluate the model on test data and generate comprehensive metrics
        
        Produces detailed metrics on model performance across multiple quality dimensions
        """
        generated_images = self.generator.predict(test_cloudy)
        
        # Calculate various image quality metrics
        metrics = {
            'psnr': [],
            'ssim': [],
            'mae': [],
            'mse': []
        }
        
        for i in range(len(test_cloudy)):
            # Ensure proper normalization for consistent metrics
            gen_img = np.clip(generated_images[i], 0, 1)
            clear_img = np.clip(test_clear[i], 0, 1)
            
            # Calculate standard image quality metrics
            psnr = tf.image.psnr(
                tf.convert_to_tensor(clear_img[np.newaxis,...]), 
                tf.convert_to_tensor(gen_img[np.newaxis,...]), 
                max_val=1.0
            ).numpy()[0]
            
            ssim = tf.image.ssim(
                tf.convert_to_tensor(clear_img[np.newaxis,...]), 
                tf.convert_to_tensor(gen_img[np.newaxis,...]), 
                max_val=1.0
            ).numpy()[0]
            
            mae = np.mean(np.abs(clear_img - gen_img))
            mse = np.mean((clear_img - gen_img) ** 2)
            
            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim)
            metrics['mae'].append(mae)
            metrics['mse'].append(mse)
        
        # Print average metrics for quick assessment
        print("\nFinal Evaluation Metrics:")
        print(f"Average PSNR: {np.mean(metrics['psnr']):.2f} dB")
        print(f"Average SSIM: {np.mean(metrics['ssim']):.4f}")
        print(f"Average MAE: {np.mean(metrics['mae']):.4f}")
        print(f"Average MSE: {np.mean(metrics['mse']):.4f}")
        
        # Create a detailed visualization of results
        self.create_evaluation_plot(test_cloudy, generated_images, test_clear, metrics)
        
        return metrics
    
    def create_evaluation_plot(self, cloudy_images, generated_images, clear_images, metrics):
        """Create a comprehensive evaluation plot with metrics
        
        Shows side-by-side comparisons of input, output, and ground truth with error measurements
        """
        # Select a subset of images for visualization if there are many
        num_samples = min(10, len(cloudy_images))
        
        plt.figure(figsize=(20, 6 * num_samples))
        
        for i in range(num_samples):
            # Original cloudy image
            plt.subplot(num_samples, 4, i*4 + 1)
            plt.imshow(np.clip(cloudy_images[i], 0, 1))
            plt.title(f"Cloudy Input")
            plt.axis('off')
            
            # Generated clear image
            plt.subplot(num_samples, 4, i*4 + 2)
            plt.imshow(np.clip(generated_images[i], 0, 1))
            plt.title(f"Generated")
            plt.axis('off')
            
            # Ground truth clear image
            plt.subplot(num_samples, 4, i*4 + 3)
            plt.imshow(np.clip(clear_images[i], 0, 1))
            plt.title(f"Ground Truth")
            plt.axis('off')
            
            # Difference map to visualize errors
            plt.subplot(num_samples, 4, i*4 + 4)
            diff = np.mean(np.abs(generated_images[i] - clear_images[i]), axis=2)
            plt.imshow(diff, cmap='hot', vmin=0, vmax=0.5)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title(f"Error Map | PSNR: {metrics['psnr'][i]:.2f}, SSIM: {metrics['ssim'][i]:.4f}")
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/final_evaluation.png")
        plt.close()
        
        # Also create a summary plot for distribution of metrics
        self.create_summary_plot(metrics)
    
    def create_summary_plot(self, metrics):
        """Create a summary plot showing distributions of evaluation metrics
        
        Provides statistical overview of model performance across the test set
        """
        plt.figure(figsize=(15, 10))
        
        # PSNR distribution
        plt.subplot(2, 2, 1)
        plt.hist(metrics['psnr'], bins=10, color='royalblue', alpha=0.7)
        plt.axvline(np.mean(metrics['psnr']), color='red', linestyle='dashed', linewidth=2)
        plt.title(f'PSNR Distribution (Mean: {np.mean(metrics["psnr"]):.2f} dB)')
        plt.xlabel('PSNR (dB)')
        plt.ylabel('Count')
        
        # SSIM distribution
        plt.subplot(2, 2, 2)
        plt.hist(metrics['ssim'], bins=10, color='green', alpha=0.7)
        plt.axvline(np.mean(metrics['ssim']), color='red', linestyle='dashed', linewidth=2)
        plt.title(f'SSIM Distribution (Mean: {np.mean(metrics["ssim"]):.4f})')
        plt.xlabel('SSIM')
        plt.ylabel('Count')
        
        # MAE distribution
        plt.subplot(2, 2, 3)
        plt.hist(metrics['mae'], bins=10, color='orange', alpha=0.7)
        plt.axvline(np.mean(metrics['mae']), color='red', linestyle='dashed', linewidth=2)
        plt.title(f'MAE Distribution (Mean: {np.mean(metrics["mae"]):.4f})')
        plt.xlabel('MAE')
        plt.ylabel('Count')
        
        # MSE distribution
        plt.subplot(2, 2, 4)
        plt.hist(metrics['mse'], bins=10, color='purple', alpha=0.7)
        plt.axvline(np.mean(metrics['mse']), color='red', linestyle='dashed', linewidth=2)
        plt.title(f'MSE Distribution (Mean: {np.mean(metrics["mse"]):.4f})')
        plt.xlabel('MSE')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/metrics_summary.png")
        plt.close()

# Memory optimization for TensorFlow (if using GPU)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s), memory growth enabled")
    except RuntimeError as e:
        print(f"Memory growth setting error: {e}")
else:
    print("No GPUs found, using CPU")

# Train the model
print("Starting GAN training...")
cloud_gan = CloudRemovalGAN(generator, discriminator)
history = cloud_gan.train(cloudy, clear, epochs=2500, batch_size=4)

print("✓ Training complete!")
print(f"Final PSNR: {history['psnr'][-1]:.2f}")
print(f"Final SSIM: {history['ssim'][-1]:.4f}")

# Compute final evaluation with comprehensive metrics
print("Performing final evaluation...")
test_metrics = cloud_gan.evaluate_model(cloudy[:100], clear[:100])  # Use a subset for final evaluation

print("\n Evaluation complete!")
print("Final average metrics:")
print(f"  PSNR: {np.mean(test_metrics['psnr']):.2f} dB")
print(f"  SSIM: {np.mean(test_metrics['ssim']):.4f}")
print(f"  MAE: {np.mean(test_metrics['mae']):.4f}")
print(f"  MSE: {np.mean(test_metrics['mse']):.4f}")
