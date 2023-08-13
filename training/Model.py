import torch
from einops import rearrange, repeat
from torch import nn


class CNN(nn.Module):
    def __init__(self, dim,patch_size, device):
        super().__init__()
        self.patch_size = patch_size
        self.device = device
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=0, stride=4),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # nn.MaxPool2d(3, 2),

            nn.Conv2d(64, 64, 3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # nn.MaxPool2d(3, 2),

            nn.Conv2d(64, 64, 3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 64, 3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # nn.MaxPool2d(3, 2),

            nn.Flatten(),
            # nn.Linear(256, dim),
            nn.Linear(3136, dim),
            # nn.Linear(576, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            # nn.GELU(),

        )

    def forward(self, x):
        x = rearrange(x,'b c (n1 p1) (n2 p2) ->(n1 n2) b c p1 p2',p1=self.patch_size,p2=self.patch_size)
        n, b, c, p1, p2 = x.shape
        img_embedding = torch.tensor([]).to(self.device)
        for i in range(n):
            img = x[i]
            embedding = self.conv(img)
            img_embedding = torch.cat((img_embedding, embedding), dim=0)
        img_embedding = rearrange(img_embedding, '(n b) embedding ->b n embedding', n=n)
        return img_embedding


class MSA(nn.Module):
    def __init__(self, in_dim, heads=8, dim_head=64, dropout=0.):
        """
        sample: for a head: input's dimension is (10 * 4096), W_Q and W_K and W_V is (4069 * 1024), so Q and K and V
        is (10*1024). Then h_i's dimension is (10 * 1024). After a linear, the output is (10 * 500)
         :param in_dim: each patch embedding's dimension 4096
         :param heads: the number of self-attention
         :param dim_head: the dimension of each head's output 1024
        """

        super().__init__()
        # 计算最终进行全连接操作时输入神经元的个数
        # The dimension of mutil heads' output
        inner_dim = dim_head * heads
        # print('inner_dim', inner_dim)
        # # 多头注意力并且输入和输出维度相同时为True
        # project_out = not (heads == 1 and dim_head == dim)

        # 多头注意力中“头”的个数
        self.heads = heads

        # 缩放操作，论文 Attention is all you need 中有介绍
        self.scale = dim_head ** -0.5

        # 初始化一个Softmax操作
        self.attend = nn.Softmax(dim=-1)

        # 对Q、K、V三组向量先进性线性操作
        self.to_qkv = nn.Linear(in_dim, inner_dim * 3, bias=False)

        # 线性全连接
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, in_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 获得输入x的维度和多头注意力的“头”数
        b, n, _, h = *x.shape, self.heads
        # 先对Q、K、V进行线性操作，然后chunk乘三三份
        # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # 整理维度，获得Q、K、V
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        # Q, K 向量先做点乘，来计算相关性，然后除以缩放因子
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=1024, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Norm(nn.Module):
    def __init__(self, in_dim, fn):
        """
        :param in_dim: input dimension
        :param fn: the function which needs the standardized data as input
        """
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, hidden_dim, dropout=0.):
        """
        :param dim: the data's dimension
        :param depth: the number of encoder
        :param heads: the number of self-attention
        :param dim_head: the dimension of each head's output
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(0, depth):
            self.layers.append(nn.ModuleList([
                Norm(dim, MSA(in_dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                Norm(dim, FeedForward(dim=dim, hidden_dim=hidden_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for msa, ff in self.layers:
            x = msa(x) + x
            x = ff(x) + x
        return x


class MLDoLC(nn.Module):
    def __init__(self, *, patch_size, embedding_dim, depth, heads, hidden_dim, num_classes=2, pool='cls',
                 batch_size, token_number, dim_head=64, device, dropout=0., emb_dropout=0.):
        """
        :param embedding_dim: the dimension of the patch embedding. 4096
        :param depth:
        :param heads: the number of multi self attention's head
        :param token_number: patches' number. 60
        :param dim_head: the dimension of each head's output 1024
        """
        super().__init__()
        # 池化方法必须为cls或者mean
        assert pool in {'cls',
                        'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_size = patch_size
        self.cnn = CNN(dim=embedding_dim,patch_size=patch_size, device=device)

        # 位置编码，获取一组正态分布的数据用于训练
        self.pos_embedding = nn.Parameter(torch.randn(batch_size, token_number + 1, embedding_dim))
        # 分类令牌，可训练
        self.cls_token = nn.Parameter(torch.randn(batch_size, 1, embedding_dim))  # nn.Parameter()定义可学习参数
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer block
        self.transformer = Transformer(dim=embedding_dim, depth=depth, heads=heads, dim_head=dim_head,
                                       hidden_dim=hidden_dim, dropout=dropout)

        self.pool = pool
        # 占位操作
        self.to_latent = nn.Identity()
        # MLP
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),  # 正则化
            nn.Linear(embedding_dim, num_classes),  # 线性输出
            nn.Softmax(dim=1),
        )

    def forward(self, img):
        # 对x进行卷积
        x = self.cnn(img)
        # 将cls_token拼接到patch token中去  x的shape (b, n+1, 1024)
        x = torch.cat((self.cls_token, x), dim=1)
        # 进行位置编码，shape (b, n+1, 1024)
        x += self.pos_embedding[::]

        x = self.dropout(x)
        x = self.transformer(x)  # ( 32, dim)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # ( dim)
        x = self.to_latent(x)  # Identity (b, dim)
        return self.mlp_head(x)  # (b, num_classes)

#
# if __name__ == '__main__':
#     image = torch.rand( 4, 3, 512 * 5, 512)
    # print(image.shape)
    # model = MLDoLC(patch_size=512, embedding_dim=25, depth=4, heads=3, hidden_dim=24, num_classes=2, token_number=5, dim_head=14,batch_size=4)
    # print(model(image).shape)


    # model = MSA(in_dim=d, heads=2, dim_head=64, out_dim=500)
    # print(model(embedding).shape)
