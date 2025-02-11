# README

## Introduction
This repository contains code for a name generator using an LSTM neural network implemented in PyTorch. The model is trained on a dataset of names and can generate new names based on the learned patterns.

## Prerequisites
- Python 3.7+
- PyTorch 1.7.1+
- numpy

Install the required packages using:
```bash
pip install torch numpy
```

## Code Overview

### LSTM Model Definition
- `NameGeneratorLSTM`: A class defining the LSTM model with an embedding layer and a fully connected layer.

### Model Creation and Loading
- `create_model(vocab_size)`: Creates an instance of the LSTM model.
- `load_trained_model(model_path, vocab_size)`: Loads a trained model from a specified path.

### Data Preparation
- `prepare_dataset(file_path)`: Loads names from a file and prepares the dataset, including vocabulary and maximum length calculations.
- `one_hot_encode(name, char_to_index, max_length, vocab_size)`: Encodes a name as a one-hot tensor and a target tensor.
- `NamesDataset`: A custom dataset class for handling name data.

### Training and Name Generation
- `train_model(model, dataloader, epochs=10)`: Trains the LSTM model using the provided DataLoader.
- `generate_name(model, char_to_index, index_to_char, max_length, vocab_size)`: Generates a name using the trained LSTM model.

## Usage

### 1. Data Preparation
Ensure you have a text file (`yob2018.txt`) with names. The file should contain names, one per line. Update the file path in the code as needed.
```python
file_path = 'yob2018.txt'
names, vocab_size, char_to_index, index_to_char, max_length = prepare_dataset(file_path)
dataset = NamesDataset(names, char_to_index, max_length, vocab_size)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
```

### 2. Create and Train Model
Create the LSTM model and train it using the prepared dataset.
```python
model = create_model(vocab_size)
n_epochs = 10
train_model(model, dataloader, n_epochs)
torch.save(model.state_dict(), 'trained_name_generator.pth')
```

### 3. Load Trained Model
Load the trained model from the saved state.
```python
model = load_trained_model('trained_name_generator.pth', vocab_size)
```

### 4. Generate Names
Generate new names using the trained model.
```python
example_name = generate_name(model, char_to_index, index_to_char, max_length, vocab_size)
print(f'Generated Name: {example_name}')
```

## Conclusion
This README provides an overview of the code structure and usage for training an LSTM model to generate names. Follow the steps outlined to preprocess data, train, evaluate, and generate names using the provided code.
