# Pictionary-AI (Quick, Draw! Inspired Sketch Classifier)

This project is a PyTorch-based machine learning pipeline for training an AI model to recognize hand-drawn sketches, inspired by Google’s Quick, Draw! game. The trained model can be used to power an AI “guesser” in multiplayer Pictionary-style games.

## Features

- **Data Preparation**: Converts Quick, Draw! `.ndjson` files into NumPy `.npy` image arrays.
- **Training & Validation**: Modular training pipeline with PyTorch, automatic checkpointing, and validation accuracy reporting.
- **Evaluation**: Easily test and analyze trained models on validation/test sets.
- **Extensible**: Designed for easy extension with user-drawn sketches or custom word lists.

## Project Structure
