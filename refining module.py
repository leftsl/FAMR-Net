import paddle
import paddle.nn as nn

class RefiningModule(nn.Layer):
    def __init__(self, input_dim=1024):
        super(RefiningModule, self).__init__()
        # First branch layers
        self.branch1_linear1 = nn.Linear(input_dim, 512)
        self.branch1_linear2 = nn.Linear(512, 256)
        self.branch1_linear3 = nn.Linear(256, 128)
        self.branch1_silu = nn.Silu()

        # Second branch layers
        self.branch2_linear1 = nn.Linear(input_dim, 256)
        self.branch2_linear2 = nn.Linear(256, 128)
        self.branch2_linear3 = nn.Linear(128, 64)
        self.branch2_silu = nn.Silu()

        # Linear layer for attention fusion
        self.attention_fusion_linear = nn.Linear(1024, 1024)
        # Linear layer for processing the concatenated branch outputs
        self.final_linear = nn.Linear(192, 128)
        self.batch_norm = nn.BatchNorm1D(128)
        self.maxpool = nn.MaxPool1D(kernel_size=2, stride=2)
        # Assuming FeedbackImplicitAttention is a custom module, replace with actual module
        self.feedback_attention = FeedbackImplicitAttention(1024, 8)

    def forward(self, image_feature, attention):
        # Compute the first branch
        branch1_linear1_out = self.branch1_linear1(image_feature)
        branch1_linear2_in = branch1_linear1_out
        branch1_linear2_out = self.branch1_linear2(branch1_linear2_in)
        branch1_linear2_out = self.branch1_silu(branch1_linear2_out)

        branch1_linear3_in = branch1_linear1_out
        branch1_linear3_out = self.branch1_linear3(branch1_linear3_in)
        branch1_linear3_out = self.branch1_silu(branch1_linear3_out)

        branch1_add_out = branch1_linear2_out + branch1_linear3_out

        # Compute the second branch
        branch2_linear1_out = self.branch2_linear1(image_feature)
        branch2_linear2_out = self.branch2_linear2(branch2_linear1_out)
        branch2_linear2_out = self.branch2_silu(branch2_linear2_out)
        branch2_linear3_out = self.branch2_linear3(branch2_linear1_out)
        branch2_linear3_out = self.branch2_silu(branch2_linear3_out)

        branch2_add_out = branch2_linear2_out + branch2_linear3_out

        # Concatenate the outputs of both branches
        concat_out = paddle.concat([branch1_add_out, branch2_add_out], axis=-1)

        # Fuse attention
        attention_transformed = self.attention_fusion_linear(attention)
        add_attention_out = concat_out + attention_transformed

        # Apply feedback implicit attention mechanism
        feedback_attention_input = paddle.concat([branch1_linear1_out, branch2_linear1_out], axis=-1)
        feedback_attention_out = self.feedback_attention(feedback_attention_input)
        # Assuming the dimensions of feedback_attention_out are compatible for addition
        branch1_linear1_out += feedback_attention_out[:, :512]
        branch2_linear1_out += feedback_attention_out[:, 512:]

        # Recompute the subsequent layers of the first branch
        branch1_linear2_in = branch1_linear1_out
        branch1_linear2_out = self.branch1_linear2(branch1_linear2_in)
        branch1_linear2_out = self.branch1_silu(branch1_linear2_out)
        branch1_linear3_in = branch1_linear1_out
        branch1_linear3_out = self.branch1_linear3(branch1_linear3_in)
        branch1_linear3_out = self.branch1_silu(branch1_linear3_out)
        branch1_final_out = branch1_linear2_out + branch1_linear3_out

        # Recompute the subsequent layers of the second branch
        branch2_linear2_out = self.branch2_linear2(branch2_linear1_out)
        branch2_linear2_out = self.branch2_silu(branch2_linear2_out)
        branch2_linear3_out = self.branch2_linear3(branch2_linear1_out)
        branch2_linear3_out = self.branch2_silu(branch2_linear3_out)
        branch2_final_out = branch2_linear2_out + branch2_linear3_out

        # Sum the final outputs of both branches
        final_add_out = branch1_final_out + branch2_final_out

        # Apply linear transformation, pooling, and normalization
        final_linear_out = self.final_linear(final_add_out)
        norm_out = self.batch_norm(final_linear_out)
        maxpool_out = self.maxpool(norm_out.unsqueeze(1)).squeeze(1)

        # Sum the max pooled output with the outputs of branches 2 and 3
        final_output = maxpool_out + branch1_add_out + branch2_add_out

        return final_output
