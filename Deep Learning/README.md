# Deep Learning Assignments

This folder contains my deep learning homework scripts. I kept the datasets synthetic or built-in because the main purpose here is to practice the training loop and model structure.

## Programs

- `mlp_tabular_classification.py`: multilayer perceptron for tabular binary classification with validation tracking and early stopping.
- `cnn_image_classification.py`: compact convolutional neural network for image classification. It uses `torchvision.datasets.FakeData` by default so the program runs without downloading external data.
- `transformer_sequence_model.py`: transformer encoder for synthetic sequence classification with embeddings, positional encoding, and pooled sequence representations.

## Learning Objectives

- Understand the full PyTorch workflow: dataset, dataloader, module, optimizer, loss, train/evaluate loop.
- Track training and validation performance over epochs.
- Apply regularization through dropout, weight decay, and early stopping.
- Keep code modular enough for replacing synthetic data with real course datasets.
- Understand when attention-based architectures are useful for sequence structure.

## Questions I Am Tracking

1. How does a neural network training loop differ from a classical model-selection pipeline?
2. What diagnostics reveal overfitting before final test evaluation?
3. How do embeddings and positional encodings allow a transformer to represent ordered data?

## Things To Add Later

- Add experiment logging with multiple random seeds.
- Compare MLP, CNN, and transformer models under the same validation protocol.
- Add ablation studies for dropout, weight decay, number of heads, and depth.
- Write a short architecture note explaining inductive bias and failure modes.

## Run

```bash
python mlp_tabular_classification.py
python cnn_image_classification.py
python transformer_sequence_model.py
```
