import torch
from torch import nn
import torch.nn.functional as F

from GlobalAttention import GlobalAttentionGeneral as ATT_NET
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
from self_attn import SelfAttention


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


# ############## Text2Image Encoder-Decoder #######
class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = 10
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = 'LSTM'
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))
        #
        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        self.nef = 256  # define a uniform ranker

        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(model)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)
        return features, cnn_code


class CA_NET(nn.Module):
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = 256
        self.c_dim = 100
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, ncf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = 100 + ncf
        self.define_module()

    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code):
        c_z_code = torch.cat((c_code, z_code), 1)
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size ngf/3 x 8 x 8
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)
        # state size ngf/8 x 32 x 32
        out_code32 = self.upsample3(out_code)
        # state size ngf/16 x 64 x 64
        out_code64 = self.upsample4(out_code32)

        return out_code64


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, nef, ncf):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = 3
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(3):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        self.att = ATT_NET(ngf, self.ef_dim)
        self.residual = self._make_layer(ResBlock, ngf * 2)
        self.upsample = upBlock(ngf * 2, ngf)

    def forward(self, h_code, c_code, word_embs, mask):
        self.att.applyMask(mask)
        c_code, att = self.att(h_code, word_embs)
        h_c_code = torch.cat((h_code, c_code), 1)
        out_code = self.residual(h_c_code)

        out_code = self.upsample(out_code)

        return out_code, att


def get_norm(name, out_channels):
    if name == 'batch':
        norm = nn.BatchNorm2d(out_channels)
    elif name == 'instance':
        norm = nn.InstanceNorm2d(out_channels)
    else:
        norm = None
    return norm


def get_act(name):
    if name == 'relu':
        activation = nn.ReLU(inplace=True)
    elif name == 'elu':
        activation = nn.ELU(inplace=True)
    elif name == 'leaky_relu':
        activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif name == 'tanh':
        activation = nn.Tanh()
    elif name == 'sigmoid':
        activation = nn.Sigmoid()
    else:
        activation = None
    return activation


##################### Generator ##########################
class CoarseEncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 normalization=None, activation=None):
        super().__init__()

        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1))
        if normalization:
            layers.append(get_norm(normalization, out_channels))
        if activation:
            layers.append(get_act(activation))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class RefineSelfModulationEncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, dilation=2, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
            nn.BatchNorm2d(out_channels, affine=False),

        )
        self.weight = nn.Linear(200, out_channels)
        self.bias = nn.Linear(200, out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, cond):
        x = self.encode(x)
        weight = self.weight(cond).view(x.shape[0], x.shape[1], 1, 1)
        bias = self.bias(cond).view(x.shape[0], x.shape[1], 1, 1)
        x = self.leaky_relu(x * weight + bias)
        return x


class RefineAttnModulationEncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, dilation=2, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
            nn.BatchNorm2d(out_channels, affine=False),
        )
        self.attn = ATT_NET(out_channels, 256)
        self.weight = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bias = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, word_embs, text_mask):
        self.attn.applyMask(text_mask)
        x = self.encode(x)
        cond, attn = self.attn(x, word_embs)
        weight = self.weight(cond)
        bias = self.bias(cond)
        x = self.leaky_relu(x * weight + bias)
        return x, attn


class RefineEncodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 normalization=None, activation=None):
        super().__init__()

        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, 4, 2, dilation=2, padding=3))
        if normalization:
            layers.append(get_norm(normalization, out_channels))
        if activation:
            layers.append(get_act(activation))
        layers.append(
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1))
        if normalization:
            layers.append(get_norm(normalization, out_channels))
        if activation:
            layers.append(get_act(activation))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class RefineDecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 normalization=None, activation=None):
        super().__init__()

        layers = []
        layers.append(
            nn.ConvTranspose2d(in_channels, out_channels, 3, 1, padding=1))
        if normalization:
            layers.append(get_norm(normalization, out_channels))
        if activation:
            layers.append(get_act(activation))

        layers.append(
            nn.ConvTranspose2d(out_channels, out_channels, 4, 2, padding=1))
        if normalization:
            layers.append(get_norm(normalization, out_channels))
        if activation:
            layers.append(get_act(activation))
        self.decode = nn.Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class RefineNet(nn.Module):
    def __init__(self, c_img=3,
                 norm='batch', act_en='leaky_relu', act_de='relu'):
        super().__init__()

        cnum = 64
        self.en_1 = nn.Conv2d(c_img, cnum, 3, 1, padding=1)
        self.en_2 = RefineSelfModulationEncodeBlock(cnum, cnum * 2)
        self.en_3 = RefineEncodeBlock(cnum * 2, cnum * 4, normalization=norm, activation=act_en)
        self.en_4 = RefineSelfModulationEncodeBlock(cnum * 4, cnum * 8)
        self.en_5 = RefineEncodeBlock(cnum * 8, cnum * 8, normalization=norm, activation=act_en)
        self.en_6 = RefineAttnModulationEncodeBlock(cnum * 8, cnum * 8)
        self.en_7 = RefineEncodeBlock(cnum * 8, cnum * 8, normalization=norm, activation=act_en)
        self.en_8 = RefineAttnModulationEncodeBlock(cnum * 8, cnum * 8)
        self.en_9 = nn.Sequential(
            nn.Conv2d(cnum * 8, cnum * 8, 4, 2, padding=1),
            get_act(act_en))

        self.de_9 = nn.Sequential(
            nn.ConvTranspose2d(cnum * 8, cnum * 8, 4, 2, padding=1),
            get_norm(norm, cnum * 8),
            get_act(act_de))
        self.de_8 = RefineDecodeBlock(cnum * 8 * 2, cnum * 8, normalization=norm, activation=act_de)
        self.de_7 = RefineDecodeBlock(cnum * 8 * 2, cnum * 8, normalization=norm, activation=act_de)
        self.de_6 = RefineDecodeBlock(cnum * 8 * 2, cnum * 8, normalization=norm, activation=act_de)
        self.de_5 = RefineDecodeBlock(cnum * 8 * 2, cnum * 8, normalization=norm, activation=act_de)
        self.de_4 = RefineDecodeBlock(cnum * 8 * 2, cnum * 4, normalization=norm, activation=act_de)
        self.de_3 = RefineDecodeBlock(cnum * 4 * 2, cnum * 2, normalization=norm, activation=act_de)
        self.de_2 = RefineDecodeBlock(cnum * 2 * 2, cnum, normalization=norm, activation=act_de)
        self.de_1 = nn.Sequential(
            nn.ConvTranspose2d(cnum * 2, c_img, 3, 1, padding=1),
            get_act('tanh'))

        self.ca_net = CA_NET()

    def forward(self, x, z_code, sent_emb, word_embs, text_mask):
        out_1 = self.en_1(x)
        c_code, mu, logvar = self.ca_net(sent_emb)
        cond1 = torch.cat([c_code, z_code], 1)
        out_2 = self.en_2(out_1, cond1)
        out_3 = self.en_3(out_2)
        out_4 = self.en_4(out_3, cond1)
        out_5 = self.en_5(out_4)
        out_6, attn1 = self.en_6(out_5, word_embs, text_mask)
        out_7 = self.en_7(out_6)
        out_8, attn2 = self.en_8(out_7, word_embs, text_mask)
        out_9 = self.en_9(out_8)

        dout_9 = self.de_9(out_9)
        dout_9_out_8 = torch.cat([dout_9, out_8], 1)
        dout_8 = self.de_8(dout_9_out_8)
        dout_8_out_7 = torch.cat([dout_8, out_7], 1)
        dout_7 = self.de_7(dout_8_out_7)
        dout_7_out_6 = torch.cat([dout_7, out_6], 1)
        dout_6 = self.de_6(dout_7_out_6)
        dout_6_out_5 = torch.cat([dout_6, out_5], 1)
        dout_5 = self.de_5(dout_6_out_5)
        dout_5_out_4 = torch.cat([dout_5, out_4], 1)
        dout_4 = self.de_4(dout_5_out_4)
        dout_4_out_3 = torch.cat([dout_4, out_3], 1)
        dout_3 = self.de_3(dout_4_out_3)
        dout_3_out_2 = torch.cat([dout_3, out_2], 1)
        dout_2 = self.de_2(dout_3_out_2)
        dout_2_out_1 = torch.cat([dout_2, out_1], 1)
        dout_1 = self.de_1(dout_2_out_1)

        return dout_1


class InpaintNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.refine_t = RefineNet()


    def forward(self, image, mask, z_code, sent_emb, word_embs, text_mask):
        out_r_t = self.refine_t(image, z_code, sent_emb, word_embs, text_mask)
        out_r_t = image * (1. - mask) + out_r_t * mask

        return out_r_t


class PatchDiscriminator(nn.Module):
    def __init__(self, c_img=3,
                 norm='instance', act='leaky_relu'):
        super().__init__()

        c_in = c_img + c_img
        cnum = 64
        self.eps = 1e-7
        self.discriminator = nn.Sequential(
            nn.Conv2d(c_in, cnum, 4, 2, 1),
            get_act(act),

            nn.Conv2d(cnum, cnum * 2, 4, 2, 1),
            get_norm(norm, cnum * 2),
            get_act(act),

            nn.Conv2d(cnum * 2, cnum * 4, 4, 2, 1),
            get_norm(norm, cnum * 4),
            get_act(act),

            nn.Conv2d(cnum * 4, cnum * 8, 4, 1, 1),
            get_norm(norm, cnum * 8),
            get_act(act),

            nn.Conv2d(cnum * 8, 1, 4, 1, 1))

        self.local_discriminator = nn.Sequential(
            nn.Conv2d(c_img, cnum, 4, 2, 1),
            get_act(act),
            nn.Conv2d(cnum, cnum * 2, 4, 2, 1),  # 128 32 32
            get_norm(norm, cnum * 2),
            get_act(act),

            nn.Conv2d(cnum * 2, cnum * 4, 4, 2, 1),
            get_norm(norm, cnum * 4),
            get_act(act),

            nn.Conv2d(cnum * 4, cnum * 8, 4, 2, 1),

        )

        self.gen_weight = nn.Linear(256, 512 + 1)

    def forward(self, x1, x2, word_embs):
        # 对局部区域进行特征提取
        IMAGE_SIZE = x1.shape[2]
        q1 = p1 = IMAGE_SIZE // 4
        q2 = p2 = IMAGE_SIZE - IMAGE_SIZE // 4
        local_x1 = x1[:, :, q1:q2, p1:p2]
        local_x1 = self.local_discriminator(local_x1)
        local_x1 = local_x1.mean(-1).mean(-1).unsqueeze(-1)
        # 对词矩阵计算自注意力机制（没有加上mask） 表示出每一个词
        attn = self.cal_self_attn(word_embs)
        attn = torch.transpose(attn.squeeze(1), 0, 1).contiguous()
        sim = 0
        # 计算每一个词与所有图像子区域的关系
        word_embsT = torch.transpose(word_embs, 1, 2).contiguous()
        W_cond = self.gen_weight(word_embsT)
        W_cond, b_cond = W_cond[:, :, :-1], W_cond[:, :, -1].unsqueeze(-1)
        sim = torch.sigmoid(torch.bmm(W_cond, local_x1) + b_cond).squeeze(-1)
        sim = torch.clamp(sim + self.eps, max=1).t().pow(attn).prod(0)

        # 判别真假
        x = torch.cat([x1, x2], 1)
        return self.discriminator(x), sim

    def cal_self_attn(self, word_embs):
        aver = torch.mean(word_embs, 2)
        averT = aver.unsqueeze(1)  # shape: bs * 1 * ief
        attn = torch.bmm(averT, word_embs)
        attn = F.softmax(attn, dim=2)
        return attn
