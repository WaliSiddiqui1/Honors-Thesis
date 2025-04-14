import os
import numpy as np
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, constraints
from tensorflow.keras.optimizers import Adam

# Directories
DATA_DIR = "cloud_data1"
OUTPUT_DIR = os.path.join(DATA_DIR, "generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load paired data
print("Loading paired data for GAN training...")
with open(os.path.join(DATA_DIR, "paired_data_optimized.json"), 'r') as f:
    paired_data = json.load(f)

print(f"Found {len(paired_data)} paired images")

# Function to load and normalize paired images
def load_paired_images(pairs, max_pairs=None):
    """Load geographically matched pairs of cloudy and clear images"""
    if max_pairs and max_pairs < len(pairs):
        # Use only a subset if requested
        pairs = pairs[:max_pairs]
        
    cloudy_images = []
    clear_images = []
    
    for i, pair in enumerate(pairs):
        # Load cloudy image
        with open(os.path.join(DATA_DIR, "metadata", f"{pair['cloudy_id']}.json"), 'r') as f:
            cloudy_meta = json.load(f)
        cloudy_path = cloudy_meta['img_path']
        
        # Load clear image
        with open(os.path.join(DATA_DIR, "metadata", f"{pair['clear_id']}.json"), 'r') as f:
            clear_meta = json.load(f)
        clear_path = clear_meta['img_path']
        
        # Load numpy arrays
        cloudy_img = np.load(cloudy_path)
        clear_img = np.load(clear_path)
        
        # Ensure both images have the same shape
        if cloudy_img.shape != clear_img.shape:
            print(f"Warning: Shape mismatch in pair {i}. Skipping.")
            continue
            
        # Ensure normalization to [0,1]
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
        
        # Progress update
        if (i+1) % 100 == 0:
            print(f"Loaded {i+1}/{len(pairs)} pairs")
            
    return np.array(cloudy_images), np.array(clear_images)

# Load paired data
print("Loading and preparing paired images...")
cloudy, clear = load_paired_images(paired_data, max_pairs=2000)  # Limit pairs for memory management

print(f"Loaded data: cloudy {cloudy.shape}, clear {clear.shape}")

# Visualize sample pairs to verify proper loading
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

# Generator (Improved U-Net)
def build_generator(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder with batch normalization
    x1 = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.LeakyReLU(0.2)(x1)
    
    x2 = layers.Conv2D(128, 4, strides=2, padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.LeakyReLU(0.2)(x2)
    
    x3 = layers.Conv2D(256, 4, strides=2, padding='same')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.LeakyReLU(0.2)(x3)

    # Bottleneck
    b = layers.Conv2D(512, 4, strides=2, padding='same')(x3)
    b = layers.BatchNormalization()(b)
    b = layers.LeakyReLU(0.2)(b)

    # Decoder with dropout for early layers
    d3 = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(b)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.Dropout(0.5)(d3)  # Dropout for regularization
    d3 = layers.ReLU()(d3)
    d3 = layers.Concatenate()([d3, x3])
    
    d2 = layers.Conv2DTranspose(128, 4, strides=2, padding='same')(d3)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.ReLU()(d2)
    d2 = layers.Concatenate()([d2, x2])
    
    d1 = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(d2)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.ReLU()(d1)
    d1 = layers.Concatenate()([d1, x1])
    
    d0 = layers.Conv2DTranspose(32, 4, strides=2, padding='same')(d1)
    d0 = layers.BatchNormalization()(d0)
    d0 = layers.ReLU()(d0)

    # Output layer with sigmoid to ensure proper [0,1] range
    outputs = layers.Conv2D(input_shape[-1], 3, padding='same', activation='sigmoid')(d0)
    
    return Model(inputs, outputs, name="Generator")

# Discriminator with spectral normalization
def build_discriminator(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Use kernel constraint for stability
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

# Build models
input_shape = cloudy.shape[1:]
generator = build_generator(input_shape)
discriminator = build_discriminator(input_shape)

# Print model summaries
generator.summary()
discriminator.summary()

# Custom GAN training class
class CloudRemovalGAN:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        
        # Optimizers
        self.g_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        self.d_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
        
        # Loss tracking
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")
        self.l1_loss_tracker = tf.keras.metrics.Mean(name="l1_loss")
        self.psnr_metric = tf.keras.metrics.Mean(name="psnr")
        self.ssim_metric = tf.keras.metrics.Mean(name="ssim")
        
        # Binary cross entropy loss for adversarial component
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        # Weight for L1 loss
        self.lambda_l1 = 100.0
    
    @tf.function
    def train_discriminator_step(self, cloudy_images, clear_images):
        # Generate fake images
        fake_images = self.generator(cloudy_images, training=True)
        
        # Get discriminator loss
        with tf.GradientTape() as disc_tape:
            # Real images
            real_output = self.discriminator(clear_images, training=True)
            # Fake images
            fake_output = self.discriminator(fake_images, training=True)
            
            # Discriminator losses
            real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss
        
        # Apply gradients
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return disc_loss
    
    @tf.function
    def train_generator_step(self, cloudy_images, clear_images):
        with tf.GradientTape() as gen_tape:
            # Generate fake images
            fake_images = self.generator(cloudy_images, training=True)
            
            # Discriminator output on fake images
            fake_output = self.discriminator(fake_images, training=False)
            
            # Generator adversarial loss - wants discriminator to think its images are real
            gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            
            # L1 loss - pixel-wise absolute difference
            l1_loss = tf.reduce_mean(tf.abs(clear_images - fake_images))
            
            # Combined loss
            total_gen_loss = gen_loss + self.lambda_l1 * l1_loss
        
        # Apply gradients
        gen_gradients = gen_tape.gradient(total_gen_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        
        # Calculate image quality metrics
        psnr = tf.reduce_mean(tf.image.psnr(clear_images, fake_images, max_val=1.0))
        ssim = tf.reduce_mean(tf.image.ssim(clear_images, fake_images, max_val=1.0))
        
        return gen_loss, l1_loss, psnr, ssim
    
    def train(self, cloudy_images, clear_images, epochs=100, batch_size=8):
        """Train the GAN"""
        dataset_size = len(cloudy_images)
        steps_per_epoch = dataset_size // batch_size
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((cloudy_images, clear_images))
        dataset = dataset.shuffle(buffer_size=dataset_size).batch(batch_size)
        
        # For visualization
        test_cloudy = cloudy_images[:5]  # Sample images for visualization
        
        # Training history
        history = {
            'gen_loss': [],
            'disc_loss': [],
            'l1_loss': [],
            'psnr': [],
            'ssim': []
        }
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Reset metrics
            self.gen_loss_tracker.reset_states()
            self.disc_loss_tracker.reset_states()
            self.l1_loss_tracker.reset_states()
            self.psnr_metric.reset_states()
            self.ssim_metric.reset_states()
            
            for batch_idx, (cloudy_batch, clear_batch) in enumerate(dataset):
                # Train discriminator
                disc_loss = self.train_discriminator_step(cloudy_batch, clear_batch)
                
                # Train generator (twice for stability)
                gen_loss, l1_loss, psnr, ssim = self.train_generator_step(cloudy_batch, clear_batch)
                gen_loss2, l1_loss2, psnr2, ssim2 = self.train_generator_step(cloudy_batch, clear_batch)
                
                # Update metrics
                self.disc_loss_tracker.update_state(disc_loss)
                self.gen_loss_tracker.update_state((gen_loss + gen_loss2) / 2)
                self.l1_loss_tracker.update_state((l1_loss + l1_loss2) / 2)
                self.psnr_metric.update_state((psnr + psnr2) / 2)
                self.ssim_metric.update_state((ssim + ssim2) / 2)
                
                # Print progress
                if (batch_idx + 1) % 20 == 0:
                    print(f"  Batch {batch_idx+1}/{steps_per_epoch} - "
                          f"D loss: {self.disc_loss_tracker.result():.4f}, "
                          f"G loss: {self.gen_loss_tracker.result():.4f}, "
                          f"L1: {self.l1_loss_tracker.result():.4f}, "
                          f"PSNR: {self.psnr_metric.result():.2f}, "
                          f"SSIM: {self.ssim_metric.result():.4f}")
            
            # Record history
            history['gen_loss'].append(self.gen_loss_tracker.result().numpy())
            history['disc_loss'].append(self.disc_loss_tracker.result().numpy())
            history['l1_loss'].append(self.l1_loss_tracker.result().numpy())
            history['psnr'].append(self.psnr_metric.result().numpy())
            history['ssim'].append(self.ssim_metric.result().numpy())
            
            # Visualize generator progress every few epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self.visualize_progress(test_cloudy, epoch + 1)
            
            # Save model checkpoints
            if (epoch + 1) % 10 == 0:
                self.generator.save(f"{OUTPUT_DIR}/generator_epoch_{epoch+1}.h5")
                print(f"✓ Saved generator checkpoint at epoch {epoch+1}")
        
        # Save final models
        self.generator.save(f"{OUTPUT_DIR}/cloud_removal_generator.h5")
        self.discriminator.save(f"{OUTPUT_DIR}/cloud_removal_discriminator.h5")
        
        # Also save in the root directory for the restoration script
        self.generator.save("cloud_removal_generator.h5")
        
        # Plot training history
        self.plot_history(history)
        
        return history
    
    def visualize_progress(self, test_images, epoch):
        """Generate and save example output from the generator"""
        generated_images = self.generator.predict(test_images)
        
        plt.figure(figsize=(15, 5 * len(test_images)))
        for i in range(len(test_images)):
            # Original cloudy image
            plt.subplot(len(test_images), 2, i*2 + 1)
            plt.imshow(np.clip(test_images[i], 0, 1))
            plt.title(f"Cloudy Input {i+1}")
            plt.axis('off')
            
            # Generated clear image
            plt.subplot(len(test_images), 2, i*2 + 2)
            plt.imshow(np.clip(generated_images[i], 0, 1))
            plt.title(f"Generated Image {i+1}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/progress_epoch_{epoch}.png")
        plt.close()
    
    def plot_history(self, history):
        """Plot and save training history"""
        epochs = range(1, len(history['gen_loss']) + 1)
        
        plt.figure(figsize=(15, 10))
        
        # Plot losses
        plt.subplot(2, 2, 1)
        plt.plot(epochs, history['gen_loss'], 'b-', label='Generator Loss')
        plt.plot(epochs, history['disc_loss'], 'r-', label='Discriminator Loss')
        plt.title('GAN Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot L1 loss
        plt.subplot(2, 2, 2)
        plt.plot(epochs, history['l1_loss'], 'g-')
        plt.title('L1 Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # Plot PSNR
        plt.subplot(2, 2, 3)
        plt.plot(epochs, history['psnr'], 'm-')
        plt.title('PSNR (Higher is better)')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        
        # Plot SSIM
        plt.subplot(2, 2, 4)
        plt.plot(epochs, history['ssim'], 'c-')
        plt.title('SSIM (Higher is better)')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/training_history.png")
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
print(f"Model saved to {OUTPUT_DIR}/cloud_removal_generator.h5")
