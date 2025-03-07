import paddle
import paddle.nn as nn

class DualTrackWordEmbedding(nn.Layer):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.2):
        super(DualTrackWordEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Word embedding layer
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)

        # First BiLSTM layer
        self.bilstm_1 = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, direction='bidirectional',
                                dropout=dropout)
        # Second BiLSTM layer
        self.bilstm_2 = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, direction='bidirectional',
                                dropout=dropout)

        # Linear layer to adjust the output dimension of BiLSTM to the embedding dimension
        self.adjust_linear = nn.Linear(2 * hidden_dim, embedding_dim)

    def forward(self, text):
        batch_size, seq_len = text.shape
        # Perform word embedding
        word_embeds = self.word_embedding(text)

        # Initialize hidden state and cell state for the first BiLSTM
        h_0_1 = paddle.zeros([2 * self.num_layers, batch_size, self.hidden_dim])
        c_0_1 = paddle.zeros([2 * self.num_layers, batch_size, self.hidden_dim])
        # Initialize hidden state and cell state for the second BiLSTM
        h_0_2 = paddle.zeros([2 * self.num_layers, batch_size, self.hidden_dim])
        c_0_2 = paddle.zeros([2 * self.num_layers, batch_size, self.hidden_dim])

        # Store encoding results for each step
        encoded_outputs = []
        prev_encoding = None

        for t in range(seq_len):
            current_word_embed = word_embeds[:, t:t + 1, :]

            if prev_encoding is not None:
                # Concatenate the previous encoding with the current word embedding and input to the first BiLSTM
                input_1 = paddle.concat([prev_encoding, current_word_embed], axis=-1)
            else:
                input_1 = current_word_embed

            # Process through the first BiLSTM
            output_1, (h_1, c_1) = self.bilstm_1(input_1, (h_0_1, c_0_1))

            # Process through the second BiLSTM
            output_2, (h_2, c_2) = self.bilstm_2(current_word_embed, (h_0_2, c_0_2))

            # Adjust the dimension of the output from the second BiLSTM
            prev_encoding = self.adjust_linear(output_2)

            # Store the encoding result of the current step
            encoded_outputs.append(prev_encoding)

            # Update hidden state and cell state
            h_0_1 = h_1
            c_0_1 = c_1
            h_0_2 = h_2
            c_0_2 = c_2

        # Concatenate the encoding results of all steps
        encoded_outputs = paddle.concat(encoded_outputs, axis=1)
        return encoded_outputs
