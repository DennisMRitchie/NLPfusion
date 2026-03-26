class NERModel:
    """
    A Named Entity Recognition model using BiLSTM architecture.
    """

    def __init__(self, vocab_size, embedding_dim, hidden_units, output_size, dropout_rate):
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_units, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_units * 2, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
