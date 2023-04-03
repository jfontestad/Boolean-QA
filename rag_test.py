import torch
import argparse
import numpy as np
from torch import nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import RagRetriever, RagTokenizer, RagModel


class BoolQADataset(Dataset):
    def __init__(self, model, rag, include_gold: bool, split='train', ):
        super().__init__()

        self.include_gold = include_gold
        self.model = model
        dataset = load_dataset("boolq")

        if split == 'train':
            dataset = dataset['train'][:500]
        elif split == 'validation':
            dataset = dataset['validation']
        else:
            dataset = dataset['train'][8000:]

        passages = dataset['passage']
        questions = dataset['question']
        answers = dataset['answer']

        self.tokenizer = RagTokenizer.from_pretrained(self.model)
        self.retriever = RagRetriever.from_pretrained(self.model, index_name="exact", use_dummy_dataset=True)
        # initialize with RagRetriever to do everything in one forward call
        self.rag = RagModel.from_pretrained(self.model, retriever=self.retriever)

        self.generator_enc_features = []
        self.question_enc_features = []
        self.answers = []

        for i in range(len(passages)):
            passage = str(passages[i])
            question = questions[i]

            inputs = question + " " + passage if self.include_gold else question
            inputs = self.tokenizer(inputs, return_tensors="pt")
            if inputs["input_ids"].shape[1] < 512:
                outputs = self.rag(input_ids=inputs["input_ids"], n_docs=1, output_hidden_states=True)
                self.generator_enc_features.append(
                    outputs.generator_enc_last_hidden_state[0].detach().numpy())  # [300, 1024]
                self.question_enc_features.append(
                    outputs.question_encoder_last_hidden_state[0].detach().numpy())  # [784]
                self.answers.append(answers[i])

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, index):
        return self.generator_enc_features[index], self.question_enc_features[index], self.answers[index]

    def process_new_data(self, inputs):
        inputs = self.tokenizer(inputs, return_tensors="pt")
        outputs = self.rag(input_ids=inputs["input_ids"], n_docs=1, output_hidden_states=True)

        return outputs.generator_enc_last_hidden_state[0], outputs.question_encoder_last_hidden_state[0]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_to_embed = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(),
            nn.Linear(512, 1024)
        )

        self.transformer = nn.Sequential(
            PositionalEncoding(1024, ),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=1024, nhead=4), num_layers=2
            )
        )

        self.classification = nn.Sequential(
            nn.Linear(1024, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, generator_enc_features, question_enc_features):
        question_enc_features = self.q_to_embed(question_enc_features)[:, None, :]
        features = torch.cat((generator_enc_features, question_enc_features), dim=1)
        features = self.transformer(features)[:, 0]
        return self.classification(features)


def evaluate_model(model, val_dl):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (generator_enc_features, question_enc_features, answers) in enumerate(val_dl):
            generator_enc_features = generator_enc_features.to('cuda')
            question_enc_features = question_enc_features.to('cuda')
            answers = answers.to('cuda')
            outputs = model(generator_enc_features, question_enc_features)
            outputs = torch.round(torch.sigmoid(outputs))
            correct += (outputs[:, 0] == answers).sum().item()
            total += len(answers)

    return correct / total


def train(train_ds, val_ds, lr, epochs):
    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=True)

    model = ClassificationHead().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_obj = nn.BCEWithLogitsLoss()

    for e in range(epochs):
        model.train()
        for i, (generator_enc_features, question_enc_features, answers) in enumerate(train_dl):
            generator_enc_features = generator_enc_features.to('cuda')
            question_enc_features = question_enc_features.to('cuda')
            answers = answers.to('cuda')
            outputs = model(generator_enc_features, question_enc_features)
            loss = loss_obj(outputs, answers[:, None].float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch: {e}, Batch: {i}, Loss: {loss.item()}")

        print(e, evaluate_model(model, val_dl), evaluate_model(model, train_dl))
    print("learning rate: %.5f, epochs: %d" % (lr, epochs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--small_subset", type=str, default=False)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default='facebook/rag-token-base')

    args = parser.parse_args()
    print(f"Specified arguments: {args}")
    model = args.model
    # initialize with RagRetriever to do everything in one forward call
    retriever = RagRetriever.from_pretrained(model, index_name="exact", use_dummy_dataset=True)
    rag = RagModel.from_pretrained(model, retriever=retriever)

    train_ds = BoolQADataset(model=model, rag=rag, include_gold=True, split='train')
    val_ds = BoolQADataset(model=model, rag=rag, include_gold=True, split='validation')

    train(train_ds, val_ds, lr=args.lr, epochs=args.epochs)
