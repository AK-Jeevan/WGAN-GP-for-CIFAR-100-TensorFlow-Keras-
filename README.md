# 🎨 Wasserstein GAN with Gradient Penalty (WGAN-GP)  
**TensorFlow / Keras**

This repository demonstrates how to implement a **Wasserstein GAN with Gradient Penalty (WGAN-GP)** trained on the CIFAR-100 dataset using TensorFlow and Keras.

WGAN-GP addresses training instability in vanilla GANs by using the Wasserstein distance as a loss metric and enforcing 1-Lipschitz continuity through gradient penalty. This approach provides **more stable training dynamics**, **meaningful loss curves**, and **better image quality** compared to standard GANs.

---

## 🚀 What This Project Covers

- Loading and preprocessing **CIFAR-100 dataset** to `[-1, 1]` range
- Building a **convolutional generator** for image synthesis
- Building a **critic (discriminator)** that estimates Wasserstein distance
- Implementing **gradient penalty** for 1-Lipschitz constraint enforcement
- Multiple critic updates per generator update (**critic steps**)
- Wasserstein loss computation (not binary cross-entropy)
- Custom adversarial training loop with `tf.GradientTape`
- Stable optimizer configuration for WGAN-GP
- Checkpoint saving and image generation utilities

---

## 🧠 Why Use WGAN-GP?

This project helps you:

- Understand **Wasserstein GANs and gradient penalty**
- Learn about **alternative GAN loss functions** beyond BCE
- Achieve **stable GAN training** with meaningful loss metrics
- Generate **high-quality images** on realistic datasets
- Implement **production-ready GAN architectures**
- Apply **advanced adversarial training techniques**

WGAN-GP is especially beneficial when:
- Traditional GAN training suffers from mode collapse
- You need meaningful loss curves for monitoring progress
- Generating diverse, high-fidelity images is critical
- Training stability is more important than training speed

---

## 🏗️ Training Architecture

### 🔹 Generator
- **Input**: Random noise vector (`latent_dim = 128`)
- **Architecture**:
  - Dense layer (4 × 4 × 256 feature map)
  - Batch Normalization + ReLU
  - Conv2DTranspose: 4 × 4 → 8 × 8 (128 filters)
  - Conv2DTranspose: 8 × 8 → 16 × 16 (64 filters)
  - Conv2DTranspose: 16 × 16 → 32 × 32 (3 channels)
- **Output**: 32 × 32 × 3 RGB images with **tanh activation** ([-1, 1] range)
- **Purpose**: Maps random noise to realistic CIFAR-100 images

### 🔹 Critic (Discriminator)
- **Input**: Image (32 × 32 × 3) and optional conditioning
- **Architecture**:
  - Conv2D: 32 × 32 → 16 × 16 (64 filters) + LeakyReLU(0.2)
  - Conv2D: 16 × 16 → 8 × 8 (128 filters) + LeakyReLU(0.2)
  - Flatten → Dense(1) with **linear activation** (scalar score)
- **Output**: Single scalar value (Wasserstein distance estimate)
- **Purpose**: Estimates how "real" an image is; higher = more real

### 🔹 Gradient Penalty
- Interpolates between real and fake images: `x̂ = α·x_real + (1-α)·x_fake`
- Computes critic's gradient w.r.t. interpolated images
- Enforces constraint: `||∇_x̂ D(x̂)||_2 ≈ 1`
- Penalty loss: `λ·E[(||∇||_2 - 1)²]` where `λ = 10`

---

## 🧪 Training Strategy

- **Loss Function**: Wasserstein distance (not binary cross-entropy)
- **Critic Updates**: `CRITIC_STEPS = 5` per generator update
  - More critic updates → better gradient signal for generator
  - Stabilizes training compared to 1:1 update ratios
- **Gradient Penalty**: Enforces 1-Lipschitz constraint on critic
- **Optimizers**: Adam with `beta_1 = 0.0` (recommended for WGAN)
  - Learning rate: `1e-4` for both networks
  - Prevents exponential moving average from interfering with gradients
- **Data Normalization**: Images in `[-1, 1]` matching generator's tanh output

---

## 📉 Loss Functions

### Critic (Discriminator) Loss
```
L_critic = E[critic(fake)] - E[critic(real)] + λ * GP
```
- Maximizes score difference between fake and real images
- Gradient penalty term stabilizes training

### Generator Loss
```
L_generator = -E[critic(fake)]
```
- Minimizes negative critic score (i.e., maximize critic score on fakes)
- Encourages generator to produce images critic considers "real"

### Wasserstein Distance
- Non-saturating loss that provides meaningful gradients throughout training
- Loss curves directly correlate with image quality

---

## 🔍 Key Concepts Demonstrated

- **Wasserstein GAN (WGAN)**: Using Wasserstein distance instead of BCE
- **Gradient Penalty (GP)**: Enforcing 1-Lipschitz constraint via interpolation
- **Critic vs Discriminator**: Wasserstein distance estimator (not binary classifier)
- **Multiple Critic Updates**: Training critic more often for stability
- **Custom Training Loops**: Using `tf.GradientTape` for fine-grained control
- **Stable Optimizer Configuration**: Adam settings optimized for WGAN-GP
- **Image Normalization**: Preprocessing to [-1, 1] for generator compatibility

---

## 💾 Output Artifacts

After training, the script saves:

- `checkpoints/` directory containing:
  - Generator weights
  - Critic weights
  - Optimizer states
  - Training metadata
- Up to 3 checkpoint versions retained (rollback capability)

Generated images can be visualized using the `generate_and_plot()` function.

---

## ⚙️ Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `IMG_SIZE` | 32 | CIFAR-100 image resolution |
| `CHANNELS` | 3 | RGB color channels |
| `LATENT_DIM` | 128 | Noise vector dimensionality |
| `BATCH_SIZE` | 64 | Training batch size |
| `CRITIC_STEPS` | 5 | Critic updates per generator update |
| `LAMBDA_GP` | 10 | Gradient penalty weight |
| `EPOCHS` | 50 | Total training epochs |
| Learning Rate | 1e-4 | Adam optimizer learning rate |
| `beta_1` | 0.0 | Adam momentum (WGAN-recommended) |

---

## ⚠️ Important Notes

- **Critic Output**: Must be **linear** (no activation) for Wasserstein loss to work
- **Generator Output**: Must use **tanh** activation to match [-1, 1] preprocessing
- **Gradient Penalty**: Critical for training stability; `LAMBDA_GP = 10` is standard
- **Multiple Critic Updates**: Improves gradient signal; essential for convergence
- **No BatchNorm in Critic**: Traditional WGAN-GP avoids BatchNorm in discriminator
- **Loss Curves**: Unlike GANs, Wasserstein loss should **decrease monotonically**

---

## 📊 Expected Training Behavior

- **Epoch 1-10**: Critic and generator losses stabilize
- **Epoch 10-30**: Generator loss decreases; critic loss plateaus
- **Epoch 30+**: Loss curves are smooth; no mode collapse
- **Final Output**: Diverse, recognizable CIFAR-100 class images

---

## 📜 License

MIT License

---

## ⭐ Support

If this repository helped you:

⭐ Star the repo  
🧠 Share it with other GAN and deep learning learners  
🚀 Use it as a foundation for advanced generative model projects  
📖 Reference it in your research or technical articles
