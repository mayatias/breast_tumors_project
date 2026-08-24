# Image Segmentation Project

This project trains and evaluates breast ultrasound image segmentation models in PyTorch. It includes two model options:
- `CResUNet` from `main.py`
- `UNet` from `main_UNet.py`

## Requirements

Install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Notes:
- The requirements file is configured for CUDA-enabled PyTorch wheels.
- If you want CPU-only or a different CUDA version, update the PyTorch lines in `requirements.txt` before installing.

## Dataset Layout

The code expects a dataset folder named `BUSI_Dataset` by default.

Expected structure:

```text
BUSI_Dataset/
    benign/
        patient1.png
        patient1_mask.png
        patient2.png
        patient2_mask.png
        ...
    malignant/
    normal/
```

The loader reads grayscale images and matches each image with one or more mask files that end with the image stem plus `_mask`.

## How To Run

### Train and evaluate `CResUNet`

Run:

```bash
python main.py
```

This script prints the model parameter count, splits the dataset into train and test sets, trains `CResUNet`, and evaluates the result.

### Train and evaluate `UNet`

Run:

```bash
python main_UNet.py
```

This script runs the same training pipeline with the standard `UNet` model.

### Evaluate a saved model

`evaluate_saved_model.py` loads a saved checkpoint, copies the test split into `BUSI_data_test`, evaluates the model, and shows prediction overlays for the best results.

Before running it, make sure the model checkpoint path inside the file points to an existing `.pth` file.

## Outputs

Training can produce:
- model checkpoints such as `best_fold_1.pth` and `final_model.pth`
- TensorBoard logs under `runs/`
- metric plots such as loss, Dice, and IoU curves

To inspect TensorBoard logs:

```bash
tensorboard --logdir runs
```

## Python Files

- `main.py` defines the training entry point for `CResUNet`, prints parameter counts, and launches the experiment pipeline.
- `main_UNet.py` defines the training entry point for the baseline `UNet` model and launches the same pipeline.
- `Blocks.py` defines the building blocks used by the custom CResUNet architecture, including convolutional and residual-style modules.
- `config.py` stores channel-size configuration values shared by the encoder and decoder.
- `Decoder.py` builds the decoder path for `CResUNet` and combines upsampling with skip connections.
- `Encoder.py` builds the encoder path for `CResUNet` and returns bottleneck features plus skip tensors.
- `evaluate_saved_model.py` prepares a test split, evaluates a saved checkpoint, and visualizes predictions per class.
- `evaluation.py` evaluates a model on a validation loader and computes loss, Dice, IoU, accuracy, and AUC.
- `fold_utils.py` converts record objects into path lists and creates stratified patient-level folds.
- `model.py` assembles the custom `CResUNet` model from the encoder, decoder, and final segmentation layer.
- `preprocessing.py` handles image and mask transforms, dataset parsing, patient-level splitting, augmentation, and the PyTorch dataset class.
- `test_evaluation.py` evaluates a model on a test loader and returns per-image results for visualization.
- `train.py` runs one training epoch, applies gradient clipping, and logs training metrics.
- `training_pipeline.py` orchestrates training, cross-validation, checkpoint saving, TensorBoard logging, plotting, and final test evaluation.
- `UNet.py` defines a standard U-Net segmentation model used as a baseline.
- `visualization.py` displays predicted masks over images for qualitative inspection.

## Practical Notes

- Edit the dataset path and training hyperparameters inside `main.py` or `main_UNet.py` if your data is stored elsewhere.
- The project assumes grayscale ultrasound images.
- The test evaluation script writes copied test data into `BUSI_data_test` and saves training artifacts in the project root.
