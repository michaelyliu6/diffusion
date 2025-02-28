#!/usr/bin/env python
# coding: utf-8

# # The DDPM Architecutre
# 
# The DDPM paper uses a custom archituecture based off the PixleCNN++ paper, which is based on papers like U-Net and Wide ResNet which we will be breaking down in this notebook. 
# 
# ## Table of Contents
# 
# - [The U-Net](#the-u-net)
# - [The DDPM Architecture](#the-ddpm-architecutre)
# - [GroupNorm](#groupnorm)
# - [Sinusoidal Time Embeddings](#sinusoidal-time-embeddings)
# - [Sigmoid Linear Unit](#sigmoid-linear-unit)
# - [Self-Attention and the Attention Block](#self-attention-and-the-attention-block)
# - [Downsampling block](#downsampling-block)
# - [MidBlock (Bottleneck)](#midblock-bottneck)
# - [Upsampling block](#upsampling-block)
# - [Putting everything together](#putting-everything-together)

# # The U-Net
# 
# At the high level, the shape of the DDPM network resembles the U-Net architecture, pictured below. The U-Net, like ResNet, is mostly composed of convolutional layers, but ResNets were developed for classifying an image as a whole, whereas U-Net was developed for medical segmentation tasks that predict an output class such as "this is part of a tumour" for each input pixel in the image. By combining both U-Net and ResNet, the DDPM network has to both get an understanding of the global structure of the image (like ResNet), and also the ability to make fine-grained predictions at the pixel level (like U-Net).
# 
# <img src="references/U-Net Architecture.png" alt="U-Net Architecture">
# <figcaption><code>Fig. 1</code> from Ronneberger et al. (2015) in <a href="https://arxiv.org/pdf/1505.04597#page=2">U-Net: Convolutional Networks for Biomedical Image Segmentation</a></figcaption>
# 
# In the diagram, the grey rectanges represent tensors where the width of the rectangle is the number of channels (i.e. initally 3 channels for RGB images), and the height of the rectangle being the number of spatial dimensions (i.e. initally heigh x width dimensions).
# 
# The U-Net can be divided into three main parts:
# - **Downsampling** part (also called the "Encoder") - Starts at the very left of the diagram until reaching the bottom. Each downsampling block represents applying some convolutional layers followed by a max pooling downsampling operation, progressively reducing spaital dimensions while increasing the number of channels.
# - **Bottleneck** part - Represented by the bottom row of the diagram. Just a couple of convolutions that represent the most compressed represntation. Has the highest number of channels and the lowest number of spatial dimensions. 
# - **Upsampling** part (also called the "Decoder") - Starts at the last tensor on the bottom row until the very right of the diagram. Each upsampling block represents applying some convolution layers followed by a **transposed convolution** upsampling operation, progressively increasing the spatial dimensions while decreasing the number of channels. Also, uses skip connections from the **downsampling** part to bring back low level spatial details and combines them with high level features from the deeper layers. 
# 
# While regular convolutions perform a sliding window operation where the input values are combined into producing a single output value, transposed convolutions do the opposite - each input value contribute to multiple output values through the kernel weights ("spreading out" the in put values).
# 
# 
# <img src="references/transposed_conv.png" alt="Transposed Convolution" width="911" height="298">
# <figcaption><code>Fig. 14.10.1</code> in <a href="https://d2l.ai/chapter_computer-vision/transposed-conv.html">Dive into Deep Learning</a></figcaption>
# 
# 
# Finally, there is a final convolution that converts the latents back to the desired number of ouput channels. In the case of medical segmentation, there would be one channel for each class of tumor that you want to detect. In the case of image diffusion models, there will be 1 or 3 output channels (greyscale or RGB).

# # The DDPM Architecture
# 
# <img src="references/DDPM-architecture.png" 
# alt="U-Net Architecture\"
# data-mermaid="
# graph TD
#     subgraph DDPM Architecture
#         subgraph Overview
#             MTime[Num Noise Steps] --> MTimeLayer[SinusoidalEmbedding<br/>Linear: Steps -> 4C<br/>GELU<br/>Linear: 4C -> 4C]
#             MTimeLayer -->|emb|DownBlock0 & DownBlock1 & DownBlock2 & MidBlock & UpBlock0 & UpBlock1 & OutBlock
#             Image -->|3, H| InConv[7x7 Conv<br/>Padding 3] -->|C, H| DownBlock0 -->|C, H/2| DownBlock1 -->|2C,H/4| DownBlock2 -->|4C,H/4| MidBlock -->|4C,H/4| UpBlock0 -->|2C,H/2| UpBlock1 -->|C,H| OutBlock[Residual Block] -->|C,H| FinalConv[1x1 Conv] -->|3,H| Output
#             DownBlock2 -->|4C,H/4| UpBlock0
#             DownBlock1 -->|2C,H/2| UpBlock1
#         end
# end
# ">

# The model used in the paper, shown above, looks very similar to the U-Net. It has the same three part stucture with **downsampling** parts (DownBlocks), a **bottleneck** part (MidBlock), and **upsampling** parts (Upblocks). However, the DDPM architecture also has many differences. The most notable difference is, in addition to the an image input, there is also a single integer represeting the number of steps of noise added. Other differences in the DDPM model include: group normalization, sinusoidal time embeddings, new nonlineararities (SiLu), Self-attention, and residual connections instead of concatenation for skip connections. 
# 
# We will dive deeper and implemenet each part of this architecture separately in this notebooks and then finally put everything together at the end. 
# 
# 

# In[181]:


from typing import Optional, Union
import matplotlib.pyplot as plt
import torch as t
from einops import rearrange
from fancy_einsum import einsum
from torch import nn
import tests.part2_unet_architecture_tests


# # GroupNorm
# 
# <img src="references/groupnorm.png" alt="GroupNorm">
# <figcaption><code>Fig. 2</code> from Wu and He (2018) in <a href="https://arxiv.org/pdf/1803.08494#page=3">Group Normalization</a></figcaption>
# 
# In our previous "toy model" notebook, we had a **Layer Normalization** that preprocessed each training example by computing the mean and standard deviation across all channels.
# 
# In **Group Normalization** we divide our chanels into some number of groups, and we calculate the mean and standard deviation for each training example AND group. For exmaple, when the number of groups is 1, GroupNorm is can be expressed to LayerNorm. The main difference being that while LayerNorm expects the channel embedding to be last (the PyTorch convention for NLP), GroupNorm expects the channel dimension to be right after the batch dimensions (the PyTorch convention for images).
# 
# The core concept for Group norm is that it divides channels into groups and normalizes within each group for each sample independently. Features within channels often have meanintful relationships (i.e. edge detectors or color channels), by grouping channels together, we maintain these relatinoships while getting the benefits of normalization. In addition, we get more benefits like batch-size independence and more stable training. 
# 
# For more intuition, read the paper [Group Normalization](https://arxiv.org/pdf/1803.08494). Also, see [pytorch implementation](https://pytorch.org/docs/stable/_modules/torch/nn/modules/normalization.html#GroupNorm).

# In[182]:


class GroupNorm(nn.Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True, 
        device: Optional[Union[t.device, str]] = None,
        dtype: Optional[t.dtype] = None,
    ) -> None:
        super().__init__()
        assert num_channels % num_groups == 0
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine # if true, this module has learnable per-channel affine parameters initialized to ones (for weights) and zeros (for biases)
        if self.affine:
            self.weight = nn.Parameter(t.empty((self.num_channels,), device=device, dtype=dtype))  
            self.bias = nn.Parameter(t.empty((self.num_channels,), device=device, dtype=dtype)) 
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the weight and bias, if applicable."""
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Apply normalization to each group of channels.

        x: shape (batch, channels, height, width)
        out: shape (batch, channels, height, width)
        """
        # split into groups
        x = rearrange(x, "n (g c) h w -> n g c h w", g=self.num_groups) 

        # calculate mean and variance
        dim = (2, 3, 4)
        mean = x.mean(dim=dim, keepdim=True)
        var = x.var(dim=dim, keepdim=True, unbiased=False)

        # normalize
        x = x - mean
        x = x / ((var + self.eps) ** 0.5)

        # reassemble to original shape
        x = rearrange(x, "n g c h w -> n (g c) h w")

        # apply learned scale (weight) and bias parameters per-channel if affine=True
        if self.affine:
            x = x * self.weight.view(1, -1, 1, 1)
            x = x + self.bias.view(1, -1, 1, 1)
        return x


# In[183]:


tests.part2_unet_architecture_tests.test_groupnorm(GroupNorm, affine=False)
tests.part2_unet_architecture_tests.test_groupnorm(GroupNorm, affine=True)


# # Sinusoidal Time Embeddings
# 
# In BERT and GPT, the mapping from the sequence position to the embedding vector was learned by the model during training. However, it is also common to simply hardcode this mapping using a combination of sine and cosine functions at different frequences. By hardcoding the mapping, we slightly reduce the nubmer of paramters without sacrificing much performance. 
# 
# In our model, instead of sequence position, our model needs to know the number of noise steps were added to be able to anticipate how much noise to expect since the amount of noise increases with number of steps.
# 
# TLDR - Sinusoidal embeddings allows a simple linear transformation in of the "query" and "key" vectors can express relative positional relationships (i.e. this positions attends to information k positions ago). For more intuition, see [this blog post](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/). We will be reproducing, Figure 2 and Figure 3 from this blog.
# 
# 
# 
# <img src="references/positional_embedding.png" alt="Positional Embedding">
# <figcaption>From Calvo (2018) in <a href="https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3">Dissecting BERT Part 1: The Encoder</a> (different blog post from the one previously mentioned)</figcaption>

# In[184]:


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        wk = 1.0 / 10000.0 ** ((t.arange(0, embedding_size, 2)) / embedding_size)
        self.register_buffer("wk", wk)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, ) - for each batch element, the number of noise steps
        Out: shape (batch, embedding_size)
        """
        wkx = t.einsum("x, k -> xk", x, self.wk)
        stacked = t.stack((wkx.sin(), wkx.cos()), dim=-1)
        flat = rearrange(stacked, "batch k func -> batch (k func)", func=2)
        return flat


# In[185]:


emb = SinusoidalPositionEmbeddings(128)
out = emb(t.arange(50))

fig, ax = plt.subplots(figsize=(15, 5))
ax.set(xlabel="Embedding Dimension", ylabel="Num Steps", title="Position Embeddings")
im = ax.imshow(out, vmin=-1, vmax=1)
fig.colorbar(im)

fig, ax = plt.subplots(figsize=(9, 9))
im = ax.imshow(out @ out.T)
fig.colorbar(im)
ax.set(xlabel="Num Steps", ylabel="Num Steps", title="Dot product of position embeddings")


# # Sigmoid Linear Unit
# 
# The Sigmoid Linear Unit (SiLu) (also sometimes called "Swish") nonlinearity is simply elementwise `x * sigmoid(x)`. 
# 
# TLDR: It's just another activation function that claims to lead to better performance. For more details about this activation function, [Swish: A Self-Gated Activation Function](https://arxiv.org/pdf/1710.05941v1).

# In[186]:


def swish(x: t.Tensor) -> t.Tensor:
    return x * x.sigmoid()


class SiLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return swish(x)


# In[187]:


silu = SiLU()


fig, ax = plt.subplots()
x = t.linspace(-5, 5, 100)
ax.plot(x, silu(x))
ax.set(xlabel="x", ylabel="swish(x)")


# # Self-Attention and the Attention Block
# 
# <img src="references/self_attention.png" alt="Multi-Head Attention">
# <figcaption><code>Figure 2</code> from Vaswani et al. (2017) in <a href="https://arxiv.org/pdf/1706.03762#page=4">Attention Is All You Need</a></figcaption>
# 
# The self-attention mechanism in the DDPM model look very similar to the self-attention mechanism used in LLMs like GPT. However, in the DDPM model, there is no causal attention meaning that we don't need to apply as mask. Also, instead of having only one spatial dimension (token sequences), we have image data that has two spatial dimensions (height and width). 
# 

# In[188]:


class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        """Self-Attention with two spatial dimensions.

        channels: the number of channels. Should be divisible by the number of heads.
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0
        self.head_size = channels // num_heads
        self.in_proj = nn.Linear(channels, channels * 3) # 3 for Key (W_K), Query (W_Q), and Value (W_V) vectors
        self.out_proj = nn.Linear(channels, channels) # Ouput (W_O)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)
        out: shape (batch, channels, height, width)
        """
        b, c, h, w = x.shape
        assert self.channels == c

        # Reshape 2D image into sequence of feature vectors
        x = rearrange(x, "batch channels height width -> batch (height width) channels") 

        # Project input into query, key, and value vectors all at once
        qkv = self.in_proj(x).chunk(3, dim=-1)

        # Separate Q, K, V and reshape to split channels into multiple heads
        q, k, v = [
            rearrange(tensor, "batch seq (head head_size) -> batch head seq head_size", head=self.num_heads)
            for tensor in qkv
        ]

        # Compute and scale attention scores 
        out = einsum("batch head seq_q head_size, batch head seq_k head_size -> batch head seq_q seq_k", q, k)
        out = out / (self.head_size**0.5)
        attn = out.softmax(dim=-1)

        # Use attention probabilties to computed a weighted sum of values 
        out = einsum("batch head seq_k head_size, batch head seq_q seq_k -> batch head seq_q head_size", v, attn)

        # Combine all heads and restore original sequence dimension
        out = rearrange(out, "batch head seq head_size -> batch seq (head head_size)")

        # Final projection to mix information between heads
        out = self.out_proj(out)

        # Reshape back into a 2D image 
        out = rearrange(out, "batch (height width) channels -> batch channels height width", height=h, width=w)

        return out


# In[189]:


tests.part2_unet_architecture_tests.test_self_attention(SelfAttention)


# Below is a diagram of the AttentionBlock. It is pretty straight forward. There is a GroupNorm layer, followed the self-attention layer implemented above that gets added back to the residual stream at the very end.

# <img src="references/attention_block.png" alt="Attention Block
# graph TD
#     subgraph AttentionBlock
#         Image --> GroupNorm[Group Norm<br/>1 group] --> Self-Attention[Self-Attention<br/>4 heads] --> Output
#         Image --> Output
#     end"
#  width="322.33333333" height="640">

# In[190]:


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = GroupNorm(1, channels)
        self.attn = SelfAttention(channels)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x + self.attn(self.norm(x))


# In[191]:


tests.part2_unet_architecture_tests.test_attention_block(SelfAttention)


# In[192]:


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, step_dim: int, groups: int):
        """
        input_channels: number of channels in the input to foward
        output_channels: number of channels in the returned output
        step_dim: embedding dimension size for the number of steps
        groups: number of groups in the GroupNorms

        Note that the conv in the left branch is needed if c_in != c_out.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            SiLU(), nn.Linear(step_dim, output_channels)
        )  # start with silu because time_mlp in the base ended in Linear without nonlinearity

        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1), GroupNorm(groups, output_channels), SiLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, padding=1), GroupNorm(groups, output_channels), SiLU()
        )
        self.res_conv = (
            nn.Conv2d(input_channels, output_channels, 1) if input_channels != output_channels else nn.Identity() # extra convolution to get the input/output dimensions to match
        )

    def forward(self, x: t.Tensor, time_emb: t.Tensor) -> t.Tensor:
        h = self.block1(x)

        time_emb = self.mlp(time_emb)
        h = rearrange(time_emb, "b c -> b c 1 1") + h # matching up matrix dimensions
        h = self.block2(h)
        return h + self.res_conv(x)


# In[193]:


tests.part2_unet_architecture_tests.test_residual_block(ResidualBlock)


# # Downsampling Block ("Encoder")
# 
# The downblock takes in an input with height `h` and an embedding for the number of steps `NumSteps` and returns: a skip output of height `h` to the corresponding `UpBlock` **AND** a dampsampled output of height `h//2`
# 
# <img src="references/down_block.png" alt="Down Block"
# data-mermaid="
# graph TD
#     subgraph DownBlock
#         NumSteps -->|emb| DResnetBlock1 & DResnetBlock2
#         DImage[Input] -->|c_in, h| DResnetBlock1[Residual Block 1] -->|c_out, h| DResnetBlock2[Residual Block 2] -->|c_out, h| DAttention[Attention Block] -->|c_out, h| DConv2d[4x4 Conv<br/>Stride 2<br/>Padding 1] -->|c_out, h/2| Output
#         DAttention -->|c_out, h| SkipToUpBlock[Skip To<br/>UpBlock]
#     end
# " width="277" height="640">

# In[194]:


class DownBlock(nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        time_emb_dim: int,
        groups: int,
        downsample: bool,
    ):
        super().__init__()
        self.block0 = ResidualBlock(channels_in, channels_out, time_emb_dim, groups)
        self.block1 = ResidualBlock(channels_out, channels_out, time_emb_dim, groups)
        self.attn = AttentionBlock(channels_out)
        self.downsample = nn.Conv2d(channels_out, channels_out, 4, 2, 1) if downsample else nn.Identity()

    def forward(self, x: t.Tensor, step_emb: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        """
        x: shape (batch, channels, height, width)
        step_emb: shape (batch, emb)
        Return: (downsampled output, full size output to skip to matching UpBlock)
        """
        B, C, H, W = x.shape
        x = self.block0(x, step_emb)
        x = self.block1(x, step_emb)
        skip = self.attn(x)
        if isinstance(self.downsample, nn.Conv2d):
            assert H % 2 == 0, f"{H} not divisible by 2 - this will break the upsample later"
            assert W % 2 == 0, f"{W} not divisible by 2 - this will break the upsample later"
        x = self.downsample(skip)
        return x, skip


# In[195]:


tests.part2_unet_architecture_tests.test_downblock(DownBlock, downsample=True)
tests.part2_unet_architecture_tests.test_downblock(DownBlock, downsample=False)


# # MidBlock (Bottneck)
# 
# <img src="references/up_block.png" alt="Mid Block"
# data-mermaid="
# graph TD
#     subgraph MidBlock
#         UNumSteps[NumSteps] -->|emb| UResnetBlock1 & UResnetBlock2
#         UImage[Image] -->|c_mid, h| UResnetBlock1[Residual Block 1] -->|c_mid, h| UAttention[Attention Block] -->|c_mid, h| UResnetBlock2[Residual Block 2] -->|c_mid, h| UOutput[Output]
#     end
# " width="345.3" height="640">
# 
# After passing through all the `DownBlocks`, the image tensor is passed through `MidBlocks` which doesn't modify the tensor dimensions. 

# In[196]:


class MidBlock(nn.Module):
    def __init__(self, mid_dim: int, time_emb_dim: int, groups: int):
        super().__init__()
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_emb_dim, groups)
        self.mid_attn = AttentionBlock(mid_dim)
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_emb_dim, groups)

    def forward(self, x: t.Tensor, step_emb: t.Tensor):
        x = self.mid_block1(x, step_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, step_emb)
        return x


# In[197]:


tests.part2_unet_architecture_tests.test_midblock(MidBlock)


# # Upsampling Block ("Decoder")
# 
# <img src="references/up_block.png" alt="Up Block"
# data-mermaid="
# graph TD
#     subgraph UpBlock
#         UNumSteps[NumSteps] -->|emb| UResnetBlock1 & UResnetBlock2
#         Skip[Skip From<br/>DownBlock<br/>] -->|c_out, h| Concatenate
#         UImage[Image] -->|c_out, h| Concatenate -->|2*c_out, h| UResnetBlock1[Residual Block 1] -->|c_in, h| UResnetBlock2[Residual Block 2] -->|c_in, h| UAttention[Attention Block] -->|c_in, h| DConvTranspose2d[4x4 Transposed Conv<br/>Stride 2<br/>Padding 1] -->|c_in, 2h| UOutput[Output]
#     end
# " width="301" height="640">
# 
# In addition to `NumSteps` and the image data, the `UpBlock` also takes in a skip connection from the corresponding `DownBlock` (i.e. the first `UpBlock` corresponds to the last `DownBlock`, second `UpBlock` corresponds to the second to last `DownBlock`, etc).
# 
# Note: The shape dimensions `c_in` and `c_out` are with respect to the corresponding `DownBlock`. It's confusing either way you try to label it.

# In[198]:


class UpBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, time_emb_dim: int, groups: int, upsample: bool):
        super().__init__()
        self.block0 = ResidualBlock(dim_out * 2, dim_in, time_emb_dim, groups)
        self.block1 = ResidualBlock(dim_in, dim_in, time_emb_dim, groups)
        self.attn = AttentionBlock(dim_in)
        self.upsample = nn.ConvTranspose2d(dim_in, dim_in, 4, 2, 1) if upsample else nn.Identity()

    def forward(self, x: t.Tensor, step_emb: t.Tensor, skip: t.Tensor) -> t.Tensor:
        x = t.cat((x, skip), dim=1)  # cat along channel dim
        x = self.block0(x, step_emb)
        x = self.block1(x, step_emb)
        x = self.attn(x)
        x = self.upsample(x)
        return x


# In[199]:


tests.part2_unet_architecture_tests.test_upblock(UpBlock, upsample=True)
tests.part2_unet_architecture_tests.test_upblock(UpBlock, upsample=False)


# # Putting everything together
# 
# Now that we have implemented all of the pieces, we can put stitch together the entire model. Here is the high level diagram of the DDPM model again:
# 
# <img src="references/DDPM-architecture.png" 
# alt="U-Net Architecture\"
# data-mermaid="
# graph TD
#     subgraph DDPM Architecture
#         subgraph Overview
#             MTime[Num Noise Steps] --> MTimeLayer[SinusoidalEmbedding<br/>Linear: Steps -> 4C<br/>GELU<br/>Linear: 4C -> 4C]
#             MTimeLayer -->|emb|DownBlock0 & DownBlock1 & DownBlock2 & MidBlock & UpBlock0 & UpBlock1 & OutBlock
#             Image -->|3, H| InConv[7x7 Conv<br/>Padding 3] -->|C, H| DownBlock0 -->|C, H/2| DownBlock1 -->|2C,H/4| DownBlock2 -->|4C,H/4| MidBlock -->|4C,H/4| UpBlock0 -->|2C,H/2| UpBlock1 -->|C,H| OutBlock[Residual Block] -->|C,H| FinalConv[1x1 Conv] -->|3,H| Output
#             DownBlock2 -->|4C,H/4| UpBlock0
#             DownBlock1 -->|2C,H/2| UpBlock1
#         end
# end
# ">
# 

# In[200]:


class Unet(nn.Module):
    def __init__(
        self,
        image_shape: tuple[int, int, int],
        channels: int = 128,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        groups: int = 4,
        max_steps: int = 1000,
    ):
        """
        image_shape: the input and output image shape, a tuple of (C, H, W)
        channels: the number of channels after the first convolution.
        dim_mults: the number of output channels for downblock i is dim_mults[i] * channels. Note that the default arg of (1, 2, 4, 8) will contain one more DownBlock and UpBlock than the DDPM image above.
        groups: number of groups in the group normalization of each ResnetBlock (doesn't apply to attention block)
        max_steps: the max number of (de)noising steps. We also use this value as the sinusoidal positional embedding dimension (although in general these do not need to be related).
        """
        self.noise_schedule = None
        self.img_shape = image_shape
        super().__init__()
        time_emb_dim = 4 * channels
        self.init_conv = nn.Conv2d(image_shape[0], channels, 7, padding=3)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(max_steps),
            nn.Linear(max_steps, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        dims = [channels] + [channels * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        print("Channel sizes for in/out:", in_out) 

        self.downs = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= len(in_out) - 1
            self.downs.append(DownBlock(dim_in, dim_out, time_emb_dim, groups, not is_last))

        self.mid = MidBlock(dims[-1], time_emb_dim, groups)

        self.ups = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = i >= len(in_out) - 1
            self.ups.append(UpBlock(dim_in, dim_out, time_emb_dim, groups, not is_last))

        self.final_block = ResidualBlock(channels, channels, time_emb_dim, groups)
        self.final_conv = nn.Conv2d(channels, image_shape[0], 1)

    def forward(self, x: t.Tensor, num_steps: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, channels, height, width)
        num_steps: shape (batch, )

        out: shape (batch, channels, height, width)
        """
        x = self.init_conv(x)
        step_emb = self.time_mlp(num_steps)
        skips = [] # push/pop to match `UpBlocks` to `DownBlocks`
        for d in self.downs:
            assert isinstance(d, DownBlock)
            x, skip = d(x, step_emb)
            skips.append(skip)

        x = self.mid(x, step_emb)

        for u in self.ups:
            assert isinstance(u, UpBlock)
            skip = skips.pop()
            x = u(x, step_emb, skip)

        x = self.final_block(x, step_emb)
        x = self.final_conv(x)
        return x


# In[201]:


tests.part2_unet_architecture_tests.test_unet(Unet)


# # FashionMNIST

# In[202]:


import os
from pathlib import Path
from typing import Any
import numpy as np
import torch as t
import torchvision
from einops import rearrange, repeat
from IPython.display import display
from PIL import Image
from scipy import linalg
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.transforms import CenterCrop, Compose, Lambda, RandomHorizontalFlip, Resize, ToPILImage, ToTensor
from tqdm.auto import tqdm
from part1_toy_model import NoiseSchedule, sample, train

torch_device = 'cuda' if t.cuda.is_available() else "mps" if t.backends.mps.is_available() else "cpu"

def make_transform(image_size=128):
    """Pipeline from PIL Image to Tensor."""
    return Compose(
        [
            Resize(image_size), 
            CenterCrop(image_size), 
            ToTensor(), 
            Lambda(lambda t: t * 2 - 1)
        ]
    )


def make_reverse_transform():
    """Pipeline from Tensor to PIL Image."""
    return Compose(
        [
            Lambda(lambda t: (t + 1) / 2),
            Lambda(lambda t: t.permute(1, 2, 0)),
            Lambda(lambda t: t * 255.0),
            Lambda(lambda t: t.clamp(0, 255)),
            Lambda(lambda t: t.numpy().astype(np.uint8)),
            ToPILImage(),
        ]
    )

transform = make_transform()
reverse_transform = make_reverse_transform()

def display_image_grid(tensors, images_per_row=4, show_steps=False, step_interval=None, batch_index=0):
    """
    Display multiple tensor images in a grid layout with optional step labels
    
    Args:
        tensors: List of tensors (can be single images or batched)
        images_per_row: Number of images to display per row
        show_steps: If True, will select images at regular intervals and show step numbers
        step_interval: Optional custom interval for steps. If None, will show ~20 steps
        batch_index: Index to use when tensors are batched (default 0)
    """
    # Handle step selection if requested
    if show_steps:
        interval = step_interval or len(tensors) // 20
        steps, tensors = zip(*[(i, t) for i, t in enumerate(tensors) if i % interval == 0])
    
    # Convert tensors to PIL images
    pil_images = []
    for t in tensors:
        # Extract from batch if needed
        img_tensor = t[batch_index] if len(t.shape) == 4 else t
        # Convert to PIL image
        img = reverse_transform(img_tensor.cpu() if hasattr(img_tensor, 'cpu') else img_tensor)
        pil_images.append(img.resize((4 * 28, 4 * 28)))
    
    if not pil_images:
        return
        
    # Calculate dimensions
    image_width, image_height = pil_images[0].size
    rows = (len(pil_images) + images_per_row - 1) // images_per_row
    grid_width = images_per_row * image_width
    # Add extra height for labels if showing steps
    label_height = 20 if show_steps else 0
    grid_height = rows * (image_height + label_height)
    
    # Create background
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Paste images and add labels
    if show_steps:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(grid_image)
    
    for idx, img in enumerate(pil_images):
        row = idx // images_per_row
        col = idx % images_per_row
        y_position = row * (image_height + label_height)
        grid_image.paste(img, (col * image_width, y_position))
        
        # Add step label if showing steps
        if show_steps:
            label = f"Step {steps[idx]}"
            draw.text((col * image_width + 5, y_position + image_height), label, fill='black')
    
    display(grid_image)


# In[203]:


train_transform = Compose([
    ToTensor(), # Convert to tensor
    RandomHorizontalFlip(), # Data augmentation
    Lambda(lambda t: t * 2 - 1) # Normalize to [-1, 1]
])

def get_fashion_mnist(train_transform, test_transform) -> tuple[TensorDataset, TensorDataset]:
    """Return MNIST data using the provided Tensor class."""
    mnist_train = datasets.FashionMNIST("../data", train=True, download=True)
    mnist_test = datasets.FashionMNIST("../data", train=False)
    print("Preprocessing data...")
    train_tensors = TensorDataset(
        t.stack([train_transform(img) for (img, label) in tqdm(mnist_train, desc="Training data")])
    )
    test_tensors = TensorDataset(t.stack([test_transform(img) for (img, label) in tqdm(mnist_test, desc="Test data")]))
    return (train_tensors, test_tensors)


# In[204]:


data_folder = Path("data/w3d4")
data_folder.mkdir(exist_ok=True, parents=True) # Create folder if it doesn't exist
DATASET_FILENAME = data_folder / "generative_models_dataset_fashion.pt" # Save to a file to avoid preprocessing every time

if DATASET_FILENAME.exists():
    (train_dataset, test_dataset) = t.load(str(DATASET_FILENAME))
else:
    (train_dataset, test_dataset) = get_fashion_mnist(train_transform, train_transform)
    t.save((train_dataset, test_dataset), str(DATASET_FILENAME))


# In[205]:


config_dict: dict[str, Any] = dict(
    model_channels=28, # Number of channels in the model
    model_dim_mults=(1, 2, 4), # Dimension multipliers for each block
    image_shape=(1, 28, 28), # Shape of the image (28x28 black and white images)
    max_steps=1000, # Maximum number of denoising steps
    epochs=10, # Number of epochs to train
    lr=0.001, # Learning rate
    batch_size=128, 
    img_log_interval=400,
    n_images_to_log=3,
    device=torch_device,
    enable_wandb=False
)


# In[206]:


model = Unet(
    max_steps=config_dict["max_steps"],
    channels=config_dict["model_channels"],
    image_shape=config_dict["image_shape"],
    dim_mults=config_dict["model_dim_mults"],
).to(config_dict["device"])
assert isinstance(data_folder, Path)
MODEL_FILENAME = data_folder / "unet_model_thousand.pt"


if MODEL_FILENAME.exists():
    model = Unet(
        max_steps=config_dict["max_steps"],
        channels=config_dict["model_channels"],
        image_shape=config_dict["image_shape"],
        dim_mults=config_dict["model_dim_mults"],
    ).to(config_dict["device"])
    model.noise_schedule = NoiseSchedule(config_dict["max_steps"], config_dict["device"])
    model.load_state_dict(t.load(str(MODEL_FILENAME)))
else:
    print("Training model from scratch!")
    model = train(model, config_dict, train_dataset, test_dataset)
    t.save(model.state_dict(), str(MODEL_FILENAME))


# In[207]:


from huggingface_hub import HfApi, login
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize API
api = HfApi()

# Login using token from .env
login(token=os.getenv('HUGGINGFACE_API_TOKEN'))

# Rest of your upload code...
api.upload_file(
    path_or_fileobj="./data/w3d4/unet_model_thousand.pt",
    path_in_repo="ddpm-fashionmnist.pt",
    repo_id="michaelyliu6/ddpm-fashionmnist",
    repo_type="model",
    create_pr=False
)


# In[208]:


# with t.inference_mode():
#     samples = sample(model, n_samples=10)


# # Usage for samples
# print("Generated images: ")
# display_image_grid(samples, images_per_row=10)

# # Usage for training images
# print("Training images: ")
# (xs,) = train_dataset[:10]
# display_image_grid(xs, images_per_row=10)


# In[209]:


# with t.inference_mode():
#     samples = sample(model, n_samples=1, return_all_steps=True)

# print("Sequential denoising: ")
# display_image_grid(samples, images_per_row=10, show_steps=True)


# # Denoising Diffusion Implicit Models (DDIM)
# 
# 

# DDIM built upon the foundations of DDPM but offer more flexible and efficent sampling. In DDIM, the diffusion process is reinterpreted as a non-Markovian generative process which allows for faster and determinstic sampling. Without the need for a specific Markovian structure, DDIM introduced a more general framework where DDPM is just a simple case. Additionally, with deterministic sampling, DDIMs have a "consistency" propoerty meaning that samples conditioned on the same latent space should have similar high-level features. This "consistency" property gives DDIMs the ability to do semantically meaningful interpoliations on latent variables.
# 
# The main advantage of DDIM is that they can generate high-quality images with signficantly lower sampling steps (i.e. 10-50 steps) compared to DDPM (i.e. 1000 steps). This is achieved by taking "larger" steps during the denoising process than during training.
# 
# <img src="references/DDIM_sampling.png" alt="Equation 12">
# <figcaption><code>Equation 12</code> from Song et al. (2020) in <a href="https://arxiv.org/pdf/2010.02502#page=5">U-Net: Denoising Diffusion Implicit Models</a></figcaption>
# 
# 

# Conveniently for us, DDIM uses the same trained models as DDPM, and only changes the sampling procedure. We just need to implement the above equation.
# 
# Lets brake down each of these terms and what they mean:
# 
# - **"predicted $x_0$"** - This term can be interpreted as predicting $x_0$ (original image) from $x_t$ (noised image) using a trained neural network. DDPM does not explicitly estimate for $x_0$ in its sampling process, but focuses on estimating the noise that was added at each time step. 
# - **"direction pointing to $x_t$"** - This term represents a directional vector that guides the reconstruction process back towards the noise image state, $x_t$ based on the predicted noise. 
# - **random noise** - This term is a random noise term, where $\epsilon_t$ is sampled from a standard Gausssian and then scaled by $\sigma_t$. By adjusting the $\sigma_t$ parameter, DDIM can interpolate between a fully determinstic ($\sigma_t = 0$) and fully stochastic sampling process. 

# In[210]:


import torch as t
from tqdm import tqdm
from typing import Union

def sample_ddim(model, n_samples: int, steps: int, eta: float = 0.0, return_all_steps: bool = False) -> Union[t.Tensor, list[t.Tensor]]:
    """
    Sample using DDIM, which can be much faster than DDPM.

    model: The trained noise-predictor
    n_samples: The number of samples to generate
    steps: The number of sampling steps (less than the number of training steps)
    eta: The DDIM hyperparameter (0.0 yields deterministic sampling, 1.0 yields DDPM sampling)
    return_all_steps: if true, return a list of the reconstructed tensors generated at each step, rather than just the final reconstructed image tensor.

    out: shape (B, C, H, W), the denoised images
    """
    schedule = model.noise_schedule
    assert schedule is not None
    model.eval()


    # 1. Define the new (shorter) DDIM timestep sequence
    orig_T = len(schedule)
    ddim_timesteps = get_ddim_timesteps(orig_T, steps)

    shape = (n_samples, *model.img_shape)
    B, C, H, W = shape
    x = t.randn(shape, device=schedule.device)  # Start with random noise

    if return_all_steps:
        all_steps = [(x.cpu().clone())]

    for i, step in tqdm(enumerate(reversed(ddim_timesteps)), total=len(ddim_timesteps)): # Reverse to start from the last timestep 
        num_steps = t.full((n_samples,), fill_value=step, device=schedule.device) 

        # 2. Predict the noise and x_0
        pred = model(x, num_steps)  # Predict noise epsilon_theta(x_t, t)

        # 3. Get alpha_t and alpha_t-1 (for the current and previous DDIM timesteps)
        alpha = schedule.alpha_bar(step)
        alpha_prev = schedule.alpha_bar(ddim_timesteps[len(ddim_timesteps)-2-i]) if i < len(ddim_timesteps)-1 else schedule.alpha_bar(0)

        # 4. Compute sigma_t (stochasticity parameter)
        sigma = eta * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)).sqrt()

        # 5. DDIM update rule
        pred_x0 = (x - (1 - alpha).sqrt() * pred) / alpha.sqrt()  # "predicted x_0"
        pred_dir = (1 - alpha_prev - sigma**2).sqrt() * pred # "direction point to x_t"
        noise = sigma * t.randn_like(x) if i > 0 else 0  # random noise (except for the first step)

        # 6. Take the DDIM step
        x = alpha_prev.sqrt() * pred_x0 + pred_dir + noise

        if return_all_steps:
            all_steps.append(x.cpu().clone())

    if return_all_steps:
        return all_steps
    return x

def get_ddim_timesteps(original_steps, ddim_steps):
    """Helper function to get DDIM timesteps, equispaced."""
    ddim_ratio = original_steps / ddim_steps
    timesteps = [round(i * ddim_ratio) for i in range(ddim_steps)]
    return timesteps


# In[211]:


with t.inference_mode():
    samples = sample_ddim(model, n_samples=10, steps=50)

print("Generated images: ")
display_image_grid(samples, images_per_row=10)

# Usage for training images
print("Training images: ")
(xs,) = train_dataset[:10]
display_image_grid(xs, images_per_row=10)


# In[231]:


with t.inference_mode():
    samples = sample_ddim(model, n_samples=1, steps=20, return_all_steps=True)

print("Sequential denoising: ")
display_image_grid(samples, images_per_row=10, show_steps=True)

