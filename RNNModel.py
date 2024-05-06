import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import Counter

# Define the LSTM model
class NameGeneratorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(NameGeneratorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)  # Output size is input_size because of one-hot encoding
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out
    
def create_model(vocab_size):
    return NameGeneratorLSTM(vocab_size)

def load_trained_model(model_path, vocab_size):
    model = create_model(vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def prepare_dataset(file_path):
    # Load and preprocess data
    with open(file_path, 'r') as f:
        names = [line.split(',')[0].lower() for line in f if line.split(',')[0].isalpha()]
    
    vocab = set('abcdefghijklmnopqrstuvwxyz ') | {'<EOS>', '<SOS>', '<PAD>'}
    vocab_size = len(vocab)
    char_to_index = {char: i for i, char in enumerate(sorted(vocab))}
    index_to_char = {i: char for char, i in char_to_index.items()}
    
    max_length = max(len(name) for name in names) + 2  
    
    return names, vocab_size, char_to_index, index_to_char, max_length

def one_hot_encode(name, char_to_index, max_length, vocab_size):
    input_tensor = torch.zeros(max_length, vocab_size)
    target_tensor = torch.zeros(max_length, dtype=torch.long)

    # Encode <SOS> in input_tensor
    input_tensor[0][char_to_index['<SOS>']] = 1
    
    # Loop to encode the name and set <EOS> correctly
    for li, letter in enumerate(name):
        input_tensor[li + 1][char_to_index[letter]] = 1
        target_tensor[li] = char_to_index[letter]
    
   
    target_tensor[len(name)] = char_to_index['<EOS>']

    return input_tensor, target_tensor


class NamesDataset(Dataset):
    def __init__(self, names, char_to_index, max_length, vocab_size):
        self.names = names
        self.char_to_index = char_to_index
        self.max_length = max_length
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        input_tensor, target_tensor = one_hot_encode(name, self.char_to_index, self.max_length, self.vocab_size)
        return input_tensor, target_tensor

def train_model(model, dataloader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        total_loss = 0
        for input_tensor, target_tensor in dataloader:
            optimizer.zero_grad()
            output = model(input_tensor)
            loss = criterion(output.view(-1, model.fc.out_features), target_tensor.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')

def generate_name(model, char_to_index, index_to_char, max_length, vocab_size):
    with torch.no_grad():
        input_tensor = torch.zeros(1, 1, vocab_size)
        input_tensor[0][0][char_to_index['<SOS>']] = 1  # Start sequence

        name = ''
        for _ in range(max_length - 1):
            output = model(input_tensor)
            probabilities = torch.softmax(output[0, -1], dim=0).detach().numpy()
            char_index = np.random.choice(len(probabilities), p=probabilities)

            if char_index == char_to_index['<EOS>']:
                break
            elif char_index in index_to_char:
                char = index_to_char[char_index]
                name += char
                input_tensor = torch.zeros(1, 1, vocab_size)
                input_tensor[0][0][char_index] = 1
            else:
                continue  
        return name



# Example usage
if __name__ == '__main__':
    file_path = 'yob2018.txt'  
    names, vocab_size, char_to_index, index_to_char, max_length = prepare_dataset(file_path)
    
    dataset = NamesDataset(names, char_to_index, max_length, vocab_size)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = create_model(vocab_size)
    n_epochs = 10  
    train_model(model, dataloader, n_epochs)
    torch.save(model.state_dict(), 'trained_name_generator.pth')  
    
    
    model = load_trained_model('trained_name_generator.pth', vocab_size) 
    example_name = generate_name(model, char_to_index, index_to_char, max_length, vocab_size)
    print(f'Generated Name: {example_name}')
    
