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

        # 词嵌入层
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)

        # 第一个 BiLSTM 层
        self.bilstm_1 = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, direction='bidirectional',
                                dropout=dropout)
        # 第二个 BiLSTM 层
        self.bilstm_2 = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, direction='bidirectional',
                                dropout=dropout)

        # 用于将 BiLSTM 输出维度调整为嵌入维度
        self.adjust_linear = nn.Linear(2 * hidden_dim, embedding_dim)

    def forward(self, text):
        batch_size, seq_len = text.shape
        # 进行词嵌入
        word_embeds = self.word_embedding(text)

        # 初始化第一个 BiLSTM 的隐藏状态和细胞状态
        h_0_1 = paddle.zeros([2 * self.num_layers, batch_size, self.hidden_dim])
        c_0_1 = paddle.zeros([2 * self.num_layers, batch_size, self.hidden_dim])
        # 初始化第二个 BiLSTM 的隐藏状态和细胞状态
        h_0_2 = paddle.zeros([2 * self.num_layers, batch_size, self.hidden_dim])
        c_0_2 = paddle.zeros([2 * self.num_layers, batch_size, self.hidden_dim])

        # 存储每一步的编码结果
        encoded_outputs = []
        prev_encoding = None

        for t in range(seq_len):
            current_word_embed = word_embeds[:, t:t + 1, :]

            if prev_encoding is not None:
                # 将前一个编码与当前词嵌入拼接后输入第一个 BiLSTM
                input_1 = paddle.concat([prev_encoding, current_word_embed], axis=-1)
            else:
                input_1 = current_word_embed

            # 通过第一个 BiLSTM 进行处理
            output_1, (h_1, c_1) = self.bilstm_1(input_1, (h_0_1, c_0_1))

            # 通过第二个 BiLSTM 进行处理
            output_2, (h_2, c_2) = self.bilstm_2(current_word_embed, (h_0_2, c_0_2))

            # 对第二个 BiLSTM 的输出进行维度调整
            prev_encoding = self.adjust_linear(output_2)

            # 存储当前步骤的编码结果
            encoded_outputs.append(prev_encoding)

            # 更新隐藏状态和细胞状态
            h_0_1 = h_1
            c_0_1 = c_1
            h_0_2 = h_2
            c_0_2 = c_2

        # 将所有步骤的编码结果拼接起来
        encoded_outputs = paddle.concat(encoded_outputs, axis=1)
        return encoded_outputs
