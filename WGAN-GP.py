"""
Title: Wasserstein GAN with Gradient Penalty (WGAN-GP) — CIFAR-100

Description:
This script implements a WGAN with Gradient Penalty (WGAN-GP) trained on the
CIFAR-100 dataset using TensorFlow / Keras. The implementation includes a
convolutional generator and critic (discriminator), the gradient penalty
calculation to enforce the Lipschitz constraint, and a training loop where
the critic is updated multiple times per generator update.

Key Steps:
1. Load and preprocess CIFAR-100 to [-1, 1] range expected by generator output.
2. Define generator and critic network architectures using Conv / ConvTranspose.
3. Compute gradient penalty by interpolating real and fake samples and
   enforcing gradient norm ~ 1.
4. Perform multiple critic updates per generator update (CRITIC_STEPS).
5. Log losses each epoch. Optionally generate and visualize samples.

Purpose:
- Provide a minimal, readable WGAN-GP example suitable for research
  experimentation or fine-tuning on small image datasets.
- Demonstrate stable training practices (gradient penalty, critic updates).

Frameworks:
- TensorFlow 2.x / Keras
- NumPy
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Hyperparameters and constants
# ------------------------------
IMG_SIZE = 32         # Target image size (CIFAR images are 32x32)
CHANNELS = 3          # RGB images
LATENT_DIM = 128      # Size of the random noise vector fed to generator
BATCH_SIZE = 64       # Batch size used for training
CRITIC_STEPS = 5      # Number of critic updates per generator update
LAMBDA_GP = 10        # Gradient penalty coefficient (as in WGAN-GP paper)
EPOCHS = 50           # Number of training epochs

# ------------------------------
# Dataset: CIFAR-100 loading and preprocessing
# ------------------------------
# Load CIFAR-100; we ignore labels (_)
(x_train, _), (_, _) = tf.keras.datasets.cifar100.load_data()
# Convert to float32 for numerical ops
x_train = x_train.astype("float32")
# Scale pixel values from [0,255] to [-1,1] because generator uses tanh
x_train = (x_train / 127.5) - 1.0  # outputs and inputs in [-1, 1]

# Create a tf.data.Dataset: shuffle and batch with drop_remainder for fixed shapes
dataset = (
    tf.data.Dataset.from_tensor_slices(x_train)
    .shuffle(50000)
    .batch(BATCH_SIZE, drop_remainder=True)
)

# ------------------------------
# Generator: builds images from latent vectors
# ------------------------------
def build_generator():
    # Sequential generator: Dense -> reshape -> Conv2DTranspose upsamples
    model = tf.keras.Sequential([
        # Project latent vector into a small spatial feature map
        layers.Dense(4 * 4 * 256, use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        # Reshape to 4x4 feature map with 256 channels
        layers.Reshape((4, 4, 256)),

        # Upsample to 8x8
        layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        # Upsample to 16x16
        layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        # Upsample to 32x32 and map to output channels with tanh activation
        # tanh produces outputs in [-1,1], matching our preprocessing
        layers.Conv2DTranspose(CHANNELS, 4, strides=2, padding="same",
                                activation="tanh")
    ])
    return model

# ------------------------------
# Critic (Discriminator): scores images with a linear output
# ------------------------------
def build_critic():
    # Critic outputs a scalar score (no activation) for Wasserstein loss
    model = tf.keras.Sequential([
        # Downsample 32x32 -> 16x16
        layers.Conv2D(64, 4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),

        # Downsample 16x16 -> 8x8
        layers.Conv2D(128, 4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),

        # Flatten spatial maps and produce a single linear score
        layers.Flatten(),
        layers.Dense(1)  # linear output for Wasserstein distance estimation
    ])
    return model

# ------------------------------
# Gradient penalty: enforces Lipschitz continuity by penalizing gradient norm
# ------------------------------
def gradient_penalty(critic, real, fake):
    # Interpolate between real and fake images with random alpha
    alpha = tf.random.uniform([BATCH_SIZE, 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * real + (1 - alpha) * fake

    # Watch the interpolated tensor so we can compute gradients w.r.t. it
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        # Critic score for interpolated images
        pred = critic(interpolated, training=True)

    # Compute gradients of critic's output wrt interpolated images
    grads = tape.gradient(pred, interpolated)
    # Compute L2 norm per-sample over height, width, and channels
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    # Gradient penalty term: (||grad||_2 - 1)^2 averaged over batch
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

# Instantiate models
generator = build_generator()
critic = build_critic()

# Optimizers: Adam with betas recommended for WGAN-GP (beta1 small or 0)
g_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9)
d_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9)

# ------------------------------
# Training step: performs CRITIC_STEPS discriminator updates, then generator update
# ------------------------------
@tf.function
def train_step(real_images):
    # Update the critic multiple times per generator update for better Wasserstein estimate
    for _ in range(CRITIC_STEPS):
        # Sample a batch of random noise vectors z ~ N(0, I)
        noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

        with tf.GradientTape() as d_tape:
            # Generate a batch of fake images
            fake_images = generator(noise, training=True)

            # Critic scores on real and fake images (higher = more "real" under WGAN)
            real_score = critic(real_images, training=True)
            fake_score = critic(fake_images, training=True)

            # Compute gradient penalty to stabilize critic (enforce 1-Lipschitz)
            gp = gradient_penalty(critic, real_images, fake_images)
            # WGAN critic loss: E[fake] - E[real] + lambda * GP
            d_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score) + LAMBDA_GP * gp

        # Compute and apply gradients to critic parameters
        d_grads = d_tape.gradient(d_loss, critic.trainable_variables)
        d_optimizer.apply_gradients(zip(d_grads, critic.trainable_variables))

    # Generator update: try to maximize critic(fake) i.e., minimize -E[critic(fake)]
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    with tf.GradientTape() as g_tape:
        fake_images = generator(noise, training=True)
        fake_score = critic(fake_images, training=True)
        # Generator loss for WGAN is negative mean critic score on fake images
        g_loss = -tf.reduce_mean(fake_score)

    # Compute and apply gradients to generator parameters
    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

    # Return scalar losses for logging/monitoring
    return d_loss, g_loss

# ------------------------------
# Training loop: iterate epochs and dataset batches, printing progress
# ------------------------------
for epoch in range(EPOCHS):
    # Loop over dataset batches; train_step handles critic sub-iterations
    for step, real_images in enumerate(dataset):
        d_loss, g_loss = train_step(real_images)

    # Print a concise summary after each epoch
    print(f"Epoch {epoch+1} | Critic Loss: {d_loss:.4f} | Generator Loss: {g_loss:.4f}")

# ------------------------------
# generate samples after training

def generate_and_plot(num_samples=16):
    noise = tf.random.normal([num_samples, LATENT_DIM])
    images = generator(noise, training=False)
    images = (images + 1) / 2.0  # Convert to [0,1] for display
    plt.figure(figsize=(6,6))
    for i in range(num_samples):
        plt.subplot(int(np.sqrt(num_samples)), int(np.sqrt(num_samples)), i+1)
        plt.imshow(images[i])
        plt.axis('off')
    plt.show()

# Consider saving checkpoints:
ckpt = tf.train.Checkpoint(generator=generator, critic=critic,
                            g_optimizer=g_optimizer, d_optimizer=d_optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, './checkpoints', max_to_keep=3)
ckpt_manager.save()