import paddle
import paddle.nn as nn


class RefiningModule(nn.Layer):
    def __init__(self, input_dim=1024):
        super(RefiningModule, self).__init__()
        # 第一个分支
        self.branch1_linear1 = nn.Linear(input_dim, 512)
        self.branch1_linear2 = nn.Linear(512, 256)
        self.branch1_linear3 = nn.Linear(256, 128)
        self.branch1_silu = nn.Silu()

        # 第二个分支
        self.branch2_linear1 = nn.Linear(input_dim, 256)
        self.branch2_linear2 = nn.Linear(256, 128)
        self.branch2_linear3 = nn.Linear(128, 64)
        self.branch2_silu = nn.Silu()

        # 用于融合注意力的线性层
        self.attention_fusion_linear = nn.Linear(1024, 1024)
        # 用于处理分支输出融合后再变换的线性层
        self.final_linear = nn.Linear(192, 128)
        self.batch_norm = nn.BatchNorm1D(128)
        self.maxpool = nn.MaxPool1D(kernel_size=2, stride=2)
        self.feedback_attention = FeedbackImplicitAttention(1024, 8)

    def forward(self, image_feature, attention):
        # 第一个分支计算
        branch1_linear1_out = self.branch1_linear1(image_feature)
        branch1_linear2_in = branch1_linear1_out
        branch1_linear2_out = self.branch1_linear2(branch1_linear2_in)
        branch1_linear2_out = self.branch1_silu(branch1_linear2_out)

        branch1_linear3_in = branch1_linear1_out
        branch1_linear3_out = self.branch1_linear3(branch1_linear3_in)
        branch1_linear3_out = self.branch1_silu(branch1_linear3_out)

        branch1_add_out = branch1_linear2_out + branch1_linear3_out

        # 第二个分支计算
        branch2_linear1_out = self.branch2_linear1(image_feature)
        branch2_linear2_out = self.branch2_linear2(branch2_linear1_out)
        branch2_linear2_out = self.branch2_silu(branch2_linear2_out)
        branch2_linear3_out = self.branch2_linear3(branch2_linear1_out)
        branch2_linear3_out = self.branch2_silu(branch2_linear3_out)

        branch2_add_out = branch2_linear2_out + branch2_linear3_out

        # 两个分支相加结果拼接
        concat_out = paddle.concat([branch1_add_out, branch2_add_out], axis=-1)

        # 注意力融合
        attention_transformed = self.attention_fusion_linear(attention)
        add_attention_out = concat_out + attention_transformed

        # 反馈隐式注意力机制
        feedback_attention_input = paddle.concat([branch1_linear1_out, branch2_linear1_out], axis=-1)
        feedback_attention_out = self.feedback_attention(feedback_attention_input)
        # 这里假设反馈注意力输出的维度和分支1_linear1_out、branch2_linear1_out的维度适配，
        # 实际可能需要调整，这里简单相加
        branch1_linear1_out = branch1_linear1_out + feedback_attention_out[:, :512]
        branch2_linear1_out = branch2_linear1_out + feedback_attention_out[:, 512:]

        # 再次计算分支1的后续层
        branch1_linear2_in = branch1_linear1_out
        branch1_linear2_out = self.branch1_linear2(branch1_linear2_in)
        branch1_linear2_out = self.branch1_silu(branch1_linear2_out)
        branch1_linear3_in = branch1_linear1_out
        branch1_linear3_out = self.branch1_linear3(branch1_linear3_in)
        branch1_linear3_out = self.branch1_silu(branch1_linear3_out)
        branch1_final_out = branch1_linear2_out + branch1_linear3_out

        # 再次计算分支2的后续层
        branch2_linear2_out = self.branch2_linear2(branch2_linear1_out)
        branch2_linear2_out = self.branch2_silu(branch2_linear2_out)
        branch2_linear3_out = self.branch2_linear3(branch2_linear1_out)
        branch2_linear3_out = self.branch2_silu(branch2_linear3_out)
        branch2_final_out = branch2_linear2_out + branch2_linear3_out

        # 两个分支最终结果相加
        final_add_out = branch1_final_out + branch2_final_out

        # 经过线性变换、池化和归一化
        final_linear_out = self.final_linear(final_add_out)
        norm_out = self.batch_norm(final_linear_out)
        maxpool_out = self.maxpool(norm_out.unsqueeze(1)).squeeze(1)

        # 与两个分支各自的2，3相加结果再次相加
        final_output = maxpool_out + branch1_add_out + branch2_add_out

        return final_output


