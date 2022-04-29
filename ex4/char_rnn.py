import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import random
import os 
import unidecode
import string
import math
import matplotlib.pyplot as plt

BATCH_SIZE = 256
CHUNK_LEN = 128
EPOCHS = 2000

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('NO GPU AVAILABLE ERROR')
    device = torch.device("cpu")

all_characters = string.printable
n_characters = len(all_characters)

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

file, file_len = read_file('shakespeare.txt')  

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model="gru", n_layers=1):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def forward2(self, input, hidden):
        encoded = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))


def generate(decoder, prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    hidden = hidden.to(device)
    prime_input = prime_input.to(device)

    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        inp = inp.to(device)
    return predicted

def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    inp = inp.to(device)
    target = target.to(device)
    return inp, target

decoder = CharRNN(
    n_characters,
    50, # Hidden size of GRU
    n_characters,
    n_layers=1,
)

decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

decoder.to(device)

def train(inp, target):
    hidden = decoder.init_hidden(BATCH_SIZE)
    hidden = hidden.to(device)
    decoder.zero_grad()
    loss = 0

    for c in range(CHUNK_LEN):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(BATCH_SIZE, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / CHUNK_LEN

all_losses = []
loss_avg = 0

perplexities = []

output_file = open('output_file.txt', 'w')

try:
    print("Training for %d epochs..." % EPOCHS)
    for epoch in tqdm(range(1, EPOCHS + 1)):
        loss = train(*random_training_set(CHUNK_LEN, BATCH_SIZE))
        loss_avg += loss
        perplexities.append(math.exp(loss))

        if epoch % 100 == 0:
            print(epoch, epoch / EPOCHS * 100, loss)
            print(generate(decoder, 'Wh', 100), '\n')


    # Task 1
    plt.plot(perplexities)
    plt.ylabel('Perplexity')
    plt.xlabel('Epoch')
    plt.savefig('perplexities.png')

    print('Task 2\n')
    
    test1 = generate(decoder, '2 b3n', 100)
    test2 = generate(decoder, 'bg09Z', 100)
    test3 = generate(decoder, 'az@1q', 100)

    output_file.write("Test1:\n" + test1 + '\n')
    output_file.write("Test2:\n" + test2 + '\n')
    output_file.write("Test3:\n" + test3 + '\n')

    print('Task 3\n')
    test4 = generate(decoder, 'The', 100)
    test5 = generate(decoder, 'What is', 100)
    test6 = generate(decoder, 'Shall I give', 100)
    test7 = generate(decoder, 'X087hNYB BHN BYFVuhsdbs', 100)
    
    output_file.write("Test4:\n" + test4 + '\n')
    output_file.write("Test5:\n" + test5 + '\n')
    output_file.write("Test6:\n" + test6 + '\n')
    output_file.write("Test7:\n" + test7 + '\n')

    output_file.close()
    
    torch.save(decoder, 'full_run.pkl')

except Exception as e: 
    print(e)
