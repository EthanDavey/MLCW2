# CIFAR-10 Active Learning with TypiClust

This repository contains an implementation of the TypiClust (TPC) active learning algorithm as described in "Active Learning by Acquiring Contrastive Examples" by Hacohen et al. (2022). The implementation focuses on image classification using the CIFAR-10 dataset.


## Code Structure

- `src/`: Contains the source code for the implementation
  - `clustering.py`: Functions for k-means clustering and feature loading
  - `typicality.py`: Implementation of typicality score calculation and example selection
  - `fully_supervised.py`: Implementation of the fully supervised framework
  - `fully_supervised_with_SSE.py`: Implementation of the fully supervised with self-supervised embeddings framework
  - `fully_supervised_with_LGFA.py`: Implementation of proposed extension with Label Guided Feature Adaptation
  - `semi_supervised.py`: Implementation of the semi-supervised framework (Self-Training with Consistency Regularization)
  - `visualize_tsne.py`: Script to visualize the feature space using t-SNE
  - `visualize_typiclust_selection.py`: Script to visualize the TypiClust selection process

- `features/`: Contains extracted features from a pre-trained self-supervised model
  - `normalized_train_features.pkl`: Normalized features for the training set
  - `normalized_test_features.pkl`: Normalized features for the test set

- `experiments/`: Contains experiment results

- `visualizations/`: Contains visualizations

## Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for faster training)

### Setting Up the Environment

1. Clone the repository

2. Create and activate a virtual environment

3. Install the required dependencies

4. Make sure to run src/representation_learning.py to extract features first, before running any of the frameworks.

## Implemented Frameworks

### 1. Fully Supervised

Trains a deep network solely on the labeled set obtained by active queries. This framework does not use any self-supervised embeddings or unlabeled data.

### 2. Fully Supervised with Self-Supervised Embeddings (SSE)

Trains a linear classifier on the embeddings obtained from a pre-trained self-supervised model. This approach leverages the representation learning capability of self-supervised models.

### 3. Semi-Supervised

Implements a Self-Training with Consistency Regularization approach. This framework uses both labeled and unlabeled data, applying consistency constraints between different augmentations of the same unlabeled example.

### 4. Label Guided Feature Adaptation (LGFA)

Proposed extension to the TypiClust algorithm that adapts the feature space based on label information. It uses a multi-objective loss function that balances classification accuracy, feature consistency, and class separation.

## Visualization Tools

The repository includes tools to visualize the feature space using t-SNE. These visualizations help understand the distribution of classes in the feature space and how TypiClust selects representative examples.
