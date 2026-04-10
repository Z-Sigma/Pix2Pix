# Pix2Pix: Image-to-Image Translation (Label-to-Photo)

An implementation of the Pix2Pix Conditional Generative Adversarial Network (cGAN) in PyTorch, optimized for high-performance training on Google Colab and local environments. This project specifically focuses on the **CMP Facades** dataset, translating architectural semantic maps into realistic building photos.

## 🚀 Key Features

*   **Label-to-Photo Translation**: Trained to generate realistic textures and architectural details from color-coded semantic maps.
*   **Colab Optimized**: Includes a consolidated Jupyter Notebook with automated dataset downloading and setup for T4 GPUs.
*   **Mixed Precision Training**: Uses the modern `torch.amp` API for faster training and reduced memory consumption.
*   **Modular Architecture**: Clean separation of models (`Generator`, `Discriminator`), `Dataset` loading, and `Training` logic.
*   **Dynamic Data Handling**: Automatically handles various image input sizes by dynamically splitting side-by-side dataset images.

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `generator.py` | U-Net based architecture with skip connections. |
| `discriminator.py` | PatchGAN discriminator for local texture validation. |
| `dataset.py` | Dynamic dataset loader for side-by-side images. |
| `train.py` | Main training loop with Mixed Precision support. |
| `config.py` | Centralized hyperparameters and transformation pipelines. |
| `utils.py` | Helper functions for checkpointing and visualization. |
| `pix2pix_colab.ipynb` | All-in-one notebook for Google Colab users. |

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd Pix2Pix
    ```

2.  **Install dependencies**:
    ```bash
    pip install torch torchvision numpy albumentations==1.3.1 matplotlib tqdm
    ```

## 📈 Usage

### Option 1: Google Colab (Recommended)
1.  Upload `pix2pix_colab.ipynb` to Google Colab.
2.  Set Runtime to **T4 GPU**.
3.  Run the \"Setup and Data Download\" cell.
4.  Execute the training cell.

### Option 2: Local Machine
1.  Download the CMP Facades dataset and place it in a `data/` folder.
2.  Adjust paths in `config.py`.
3.  Run the training script:
    ```bash
    python train.py
    ```

## 🖼️ Visualization

During training, the model saves evaluation samples in the `evaluation/` folder.
*   **Input Image**: The color-coded architectural map.
*   **Generated Image**: The AI-hallucinated realistic building.
*   **Ground Truth**: The original real building photo.

## 📜 Acknowledgments

*   Based on the paper: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) by Isola et al.
*   Dataset: [CMP Facades Dataset](https://cmp.felk.cvut.cz/~typet/facade/)
