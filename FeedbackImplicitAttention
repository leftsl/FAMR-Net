import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class FeedbackImplicitAttention(nn.Layer):
    def __init__(self, d_model, num_heads, num_batches, num_enhance_modules):
        super(FeedbackImplicitAttention, self).__init__()
        self.multihead_attention = nn.MultiHeadAttention(d_model, num_heads)
        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        
        # Large linear layer after flatten
        self.large_linear = nn.Linear(d_model * num_heads, d_model * num_heads)
        
        # Independent linear layers for each batch
        self.batch_linears = nn.LayerList([
            nn.Linear(d_model * num_heads, d_model * num_heads)
            for _ in range(num_batches)
        ])
        
        # Enhance modules
        self.enhance_modules = nn.LayerList([
            nn.Sequential(
                nn.Linear(d_model * num_heads, d_model * num_heads),
                nn.ReLU(),
                nn.Linear(d_model * num_heads, d_model * num_heads)
            )
            for _ in range(num_enhance_modules)
        ])

    def forward(self, query, key, value):
        # Multi-head self attention
        attn_output, _ = self.multihead_attention(query, key, value)
        
        # Flatten operation
        flattened_output = self.flatten(attn_output)
        
        # Large linear layer
        large_linear_output = self.large_linear(flattened_output)
        
        # Split into batches and apply independent linear layers
        batch_outputs = []
        for i in range(num_batches):
            start_idx = i * (d_model * num_heads)
            end_idx = (i + 1) * (d_model * num_heads)
            batch_output = self.batch_linears[i](large_linear_output[:, start_idx:end_idx])
            batch_outputs.append(batch_output)
        
        # Combine batch outputs using addition and multiplication
        combined_output = None
        for i in range(len(batch_outputs)):
            if i == 0:
                combined_output = batch_outputs[i]
            else:
                combined_output += batch_outputs[i]
        
        # Apply enhance modules
        enhance_output = None
        for module in self.enhance_modules:
            if enhance_output is None:
                enhance_output = module(combined_output)
            else:
                enhance_output += module(enhance_output)
        
        return enhance_output

# Example usage
d_model = d_model
num_heads = num_heads
num_batches = num_batches  
num_enhance_modules = num_enhance_modules
  
feedback_implicit_attention = FeedbackImplicitAttention(d_model, n      um_heads, num_batches, num_enhance_modules)
  

query = paddle.randn([10, 60, d_model])
key = paddle.randn([10, 60, d_model])  
value = paddle.randn([10, 60, d_model])


output = feedback_implicit_attention(query, key, value)
print(output.shape)  
