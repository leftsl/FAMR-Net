import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.vision.datasets import DatasetFolder
from paddle.metric import BLEU, ROUGE
import numpy as np


# 图像编码器
class ImageEncoder(nn.Layer):
    def __init__(self, backbone=None, preprocess=None):
        super(ImageEncoder, self).__init__()
        if backbone is None:
            self.backbone = paddle.vision.models.resnet50(pretrained=False)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            self.backbone = backbone

        if preprocess is None:
            self.preprocess = T.Compose([
                T.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()) * 255.0),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.preprocess = preprocess

        self.mlp = nn.Sequential(
            nn.Linear(self._get_backbone_output_dim(), 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)
        )

    def _get_backbone_output_dim(self):
        temp_input = paddle.randn([1, 3, 224, 224])
        output = self.backbone(temp_input)
        return output.shape[1]

    def forward(self, x):
        x = self.preprocess(x)
        x = self.backbone(x)
        x = self.mlp(x)
        return x


# 精炼模块
class RefiningModule(nn.Layer):
    def __init__(self, input_dim=1024):
        super(RefiningModule, self).__init__()
        self.linear1_1 = nn.Linear(input_dim, 512)
        self.linear1_2 = nn.Linear(512, 256)
        self.linear2_1 = nn.Linear(input_dim, 256)
        self.linear2_2 = nn.Linear(256, 256)
        self.silu = nn.Silu()
        self.maxpool = nn.MaxPool1D(kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm1D(512)

    def forward(self, x, guidance_info):
        x1 = self.linear1_1(x)
        x1 = self.silu(x1)
        x1 = self.linear1_2(x1)

        x2 = self.linear2_1(x)
        x2 = self.silu(x2)
        x2 = self.linear2_2(x2)

        x = paddle.concat([x1, x2], axis=-1)
        x = self.batch_norm(x)
        x = self.maxpool(x.unsqueeze(1)).squeeze(1)
        # 根据架构图，这里用guidance_info来增强特征
        x = x * guidance_info
        return x


# 双轨词嵌入
class DualTrackWordEmbedding(nn.Layer):
    def __init__(self, vocab_size, word_embedding_dim):
        super(DualTrackWordEmbedding, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim)
        self.bilstm = nn.LSTM(word_embedding_dim, word_embedding_dim, num_layers=1, direction='bidirectional')
        self.linear = nn.Linear(2 * word_embedding_dim, word_embedding_dim)

    def forward(self, text):
        x = self.word_embedding(text)
        x, _ = self.bilstm(x)
        x = self.linear(x)
        x = x + self.word_embedding(text)
        return x


# 反馈隐式注意力机制
class FeedbackImplicitAttention(nn.Layer):
    def __init__(self, in_dim, num_heads):
        super(FeedbackImplicitAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.q_linear = nn.Linear(in_dim, in_dim)
        self.k_linear = nn.Linear(in_dim, in_dim)
        self.v_linear = nn.Linear(in_dim, in_dim)
        self.out_linear = nn.Linear(in_dim, in_dim)
        self.silu = nn.Silu()

    def forward(self, x):
        batch_size, seq_len, in_dim = x.shape
        q = self.q_linear(x).reshape([batch_size, seq_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        k = self.k_linear(x).reshape([batch_size, seq_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        v = self.v_linear(x).reshape([batch_size, seq_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])

        attn = paddle.matmul(q, k.transpose([0, 1, 3, 2])) / np.sqrt(self.head_dim)
        attn = paddle.softmax(attn, axis=-1)
        x = paddle.matmul(attn, v).transpose([0, 2, 1, 3]).reshape([batch_size, seq_len, in_dim])
        x = self.out_linear(x)
        x = self.silu(x)
        return x


# 嵌套解码器
class NestingDecoder(nn.Layer):
    def __init__(self, word_embedding_dim, hidden_dim, vocab_size):
        super(NestingDecoder, self).__init__()
        self.natural_language_generation = nn.LSTM(word_embedding_dim + 1024, hidden_dim, num_layers=1, direction='bidirectional')
        self.natural_language_reference = nn.LSTM(word_embedding_dim + 1024, hidden_dim, num_layers=1, direction='bidirectional')
        self.natural_language_output = nn.Sequential(
            nn.Linear(2 * hidden_dim, vocab_size),
            nn.Softmax(axis=-1)
        )

    def forward(self, attention, refined_feature, text_embedding):
        gen_input = paddle.concat([attention, refined_feature, text_embedding], axis=-1)
        gen_output, _ = self.natural_language_generation(gen_input)

        ref_input = paddle.concat([refined_feature, text_embedding], axis=-1)
        ref_output, _ = self.natural_language_reference(ref_input)

        output = self.natural_language_output(gen_output - ref_output)
        return output


# ENIMNet模型整合
class ENIMNet(nn.Layer):
    def __init__(self, vocab_size, word_embedding_dim, hidden_dim, num_heads,
                 image_encoder_backbone=None, image_encoder_preprocess=None):
        super(ENIMNet, self).__init__()
        self.image_encoder = ImageEncoder(backbone=image_encoder_backbone, preprocess=image_encoder_preprocess)
        self.refining_module = RefiningModule()
        self.dual_track_word_embedding = DualTrackWordEmbedding(vocab_size, word_embedding_dim)
        self.feedback_attention = FeedbackImplicitAttention(1024, num_heads)
        self.decoder = NestingDecoder(word_embedding_dim, hidden_dim, vocab_size)

    def forward(self, image, text):
        image_feature = self.image_encoder(image)
        attention = self.feedback_attention(image_feature)
        # 假设guidance_info是通过其他方式获取，这里简单用attention代替
        guidance_info = attention
        refined_feature = self.refining_module(image_feature, guidance_info)

        word_embedding = self.dual_track_word_embedding(text)
        output = self.decoder(attention, refined_feature, word_embedding)
        return output


# 数据预处理
def preprocess_image(image):
    transform = T.Compose([
        T.Resize(512, interpolation='bicubic'),
        T.CenterCrop(512),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)


# 加载数据集
class MedicalDataset(DatasetFolder):
    def __init__(self, root, transform=None):
        super(MedicalDataset, self).__init__(root, loader=lambda x: x, extensions=('.jpg', '.jpeg', '.png'), transform=transform)


# 模型训练
def train(model, train_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (images, texts) in enumerate(train_loader):
            optimizer.clear_grad()
            outputs = model(images, texts)
            loss = criterion(outputs, texts)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}')


# 模型评估
def evaluate(model, test_loader):
    model.eval()
    bleu_metric = BLEU()
    rouge_metric = ROUGE()
    with paddle.no_grad():
        for images, texts in test_loader:
            outputs = model(images, texts)
            # 这里需要将outputs转换为预测文本，根据实际情况调整
            predicted_texts = paddle.argmax(outputs, axis=-1)
            bleu_score = bleu_metric.compute(predicted_texts, texts)
            rouge_score = rouge_metric.compute(predicted_texts, texts)
            print(f'BLEU Score: {bleu_score}, ROUGE Score: {rouge_score}')


# 示例使用
vocab_size = 1000  # 根据实际词汇表大小调整
word_embedding_dim = 80  # 根据论文实验结果调整
hidden_dim = 512
num_heads = 8

# 你可以传入自定义的骨干网络和预处理方法
custom_backbone = paddle.vision.models.resnet101(pretrained=False)
custom_preprocess = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

model = ENIMNet(vocab_size, word_embedding_dim, hidden_dim, num_heads,
                image_encoder_backbone=custom_backbone, image_encoder_preprocess=custom_preprocess)
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.001)
criterion = nn.CrossEntropyLoss()

# 假设数据集存储在data_root目录下
data_root = 'your_data_root'
train_dataset = MedicalDataset(data_root + '/train', transform=preprocess_image)
test_dataset = MedicalDataset(data_root + '/test', transform=preprocess_image)

train_loader = paddle.io.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = paddle.io.DataLoader(test_dataset, batch_size=16, shuffle=False)

train(model, train_loader, optimizer, criterion, epochs=10)
evaluate(model, test_loader)
