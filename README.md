# diffusion-classifier-pytorch

## Project Overview

This project explores the use of latent diffusion models for image classification, comparing their performance to a standard ResNet-18 classifier. The main goal is to evaluate whether generative diffusion models can be adapted for classification tasks and how they compare to traditional deep learning approaches.

## Dataset

The experiments use the [UCMerced Land Use Dataset](https://weegee.vision.ucmerced.edu/datasets/landuse.html), a collection of aerial images categorized into various land use classes. The dataset is split into training and test sets (80/20 split) and loaded using PyTorch's `ImageFolder` utility.

## Methods

### 1. Latent Diffusion Classifier
- **Model**: Uses the Stable Diffusion 2.1 base model (from HuggingFace Diffusers) to extract latent representations of images.
- **Classification**: Prompts are embedded using the model's tokenizer and text encoder. For each test image, the classifier predicts the class by evaluating the error between the image's latent representation and the prompt embeddings through the diffusion process.
- **Selection**: An adaptive probability method is used to select the most likely class for each image.

### 2. ResNet-18 Baseline
- **Model**: Standard ResNet-18 from torchvision, with the final layer adapted to the number of classes in the dataset.
- **Training**: Trained using cross-entropy loss and Adam optimizer for 10 epochs.

## Results

- **Latent Diffusion Classifier**: ~50% accuracy on the test set.
- **ResNet-18 Baseline**: ~88% accuracy on the test set.

## Setup & Usage

1. **Install dependencies** (in a Jupyter environment):
   ```bash
   pip install -r requirements.txt
   pip install xformers
   ```
2. **Download the UCMerced Land Use Dataset** and place it in the expected directory (see `args['dataset_path']` in the notebook).
3. **Run the notebook** `PyTorch_Project_Muhammad_Yousif.ipynb` to reproduce experiments, visualize results, and compare classifiers.

## Evaluation Metrics

- **Accuracy**: Proportion of correct predictions.
- **Precision, Recall, F1 Score**: Computed using scikit-learn for a detailed performance breakdown.
- **Confusion Matrix**: For error analysis.

## Files
- `PyTorch_Project_Muhammad_Yousif.ipynb`: Main notebook with all code, experiments, and results.
- `Report.pdf`, `Presentation1.pdf`: Project report and presentation slides.

## Notes
- The diffusion classifier is experimental and demonstrates the potential (and current limitations) of generative models for classification.
- For best results, use a GPU-enabled environment.