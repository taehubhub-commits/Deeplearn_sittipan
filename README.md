# Deeplearn_sittipan
# Name Language Classification Using Character-Level LSTM

A Deep Learning project that predicts the language category of a name using a character-level Long Short-Term Memory (LSTM) neural network.

## Overview

This project uses a character-level LSTM model to classify names into 18 language categories. The model learns character patterns from names and predicts the most likely language class.

The project explores neural network training, sequence modeling, regularization, optimization, and model evaluation using PyTorch.

## Objectives

- Build a character-level LSTM for name language classification.
- Train a neural network to classify names into 18 language categories.
- Compare different training configurations using a validation set.
- Evaluate performance using accuracy, precision, recall, and F1-score.
- Analyze model limitations and identify possible improvements.

## Model Architecture

The model consists of three main components:

1. **Embedding Layer** — Converts character indices into dense vectors.
2. **LSTM Layer** — Learns sequential patterns in character sequences.
3. **Linear Classification Layer** — Produces scores for 18 language classes.

### Model Configuration

| Parameter | Value |
|---|---|
| Embedding dimension | 32 |
| LSTM hidden size | 64 |
| Number of classes | 18 |
| Approximate trainable parameters | 28,114 |
| Loss function | Cross-Entropy Loss |
| Optimizer | Adam |
| Baseline learning rate | 0.001 |
| Weight decay | 0.0001 |
| Gradient clipping | 1.0 |
| Training epochs | 8 |

## Dataset

The dataset contains 18 language categories and is divided into three subsets.

| Split | Number of Samples |
|---|---:|
| Training | 12,046 |
| Validation | 4,014 |
| Test | 4,014 |
| **Total** | **20,074** |

Each name is represented as a sequence of characters, which are converted into numerical indices before being passed to the model.

**Dataset source:** Add the original dataset name and URL here.

## Experiments

Three configurations were compared using the validation set.

| Experiment | Learning Rate | Dropout | Train Accuracy | Validation Accuracy | Validation Loss |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.001 | None | 75.04% | 71.77% | 0.9842 |
| Dropout | 0.001 | 0.3 | 75.42% | **72.40%** | 0.9871 |
| Lower Learning Rate | 0.0005 | None | 70.82% | 68.44% | 1.1155 |

The dropout configuration achieved the highest validation accuracy of **72.40%**, an improvement of 0.63 percentage points over the baseline.

The baseline achieved a slightly lower validation loss. The lower-learning-rate configuration achieved lower validation accuracy after eight epochs.

## Evaluation

The selected dropout model achieved the following validation results:

| Metric | Score |
|---|---:|
| Validation Accuracy | 72.40% |
| Macro-average Precision | 0.196 |
| Macro-average Recall | 0.242 |
| Macro-average F1-score | 0.216 |
| Weighted-average F1-score | 0.669 |

The model performed well on some classes, including Russian (F1-score: 0.883) and Arabic (F1-score: 0.827). However, several classes had zero precision, recall, and F1-score.

These results indicate uneven classification performance across the 18 classes. The imbalance in the number of examples per class may be one contributing factor.

### Baseline Comparison

A majority-class classifier was used as a trivial baseline and achieved a previously recorded test accuracy of **45.94%**.

An earlier LSTM evaluation recorded a test accuracy of **79.05%** and a test loss of **0.722**. This result came from an earlier training run and has not been verified as the test performance of the dropout model selected in the later experiments. It should not be treated as the final test score of that selected model.

## Limitations

- Performance varies considerably across language categories.
- Underrepresented classes may be difficult to classify reliably.
- Similar character patterns across languages can cause misclassification.
- The dropout improvement over the baseline is modest.
- The final test performance of the selected dropout model still needs to be verified.

## Future Improvements

- Use class-weighted cross-entropy loss to address class imbalance.
- Collect more training examples for underrepresented classes.
- Compare additional model architectures and training configurations.
- Analyze a confusion matrix to identify commonly confused classes.
- Evaluate the final selected model on the held-out test set once.

## Technologies

- Python
- PyTorch
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## How to Run

1. Clone or download this repository.
2. Install the required Python packages.
3. Open the project notebook in Jupyter Notebook or Google Colab.
4. Run the notebook cells in order to load the dataset, preprocess names, build the model, and train the experiments.
5. Review the validation results and evaluation metrics.

Install the main dependencies with:

```bash
pip install torch numpy pandas matplotlib scikit-learn jupyter
```

The dataset must be available at the location expected by the notebook.

## Project Status

The model training and validation experiments have been completed. Final verification of the selected model on the held-out test set is still required.

## Authors

Add the project members' names here.

## License

Add the appropriate license for this project and its dataset.
