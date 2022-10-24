import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
import numpy as np
from GlobalAttention import func_attention

def label_like(label, x):
    assert label == 0 or label == 1
    v = torch.zeros_like(x) if label == 0 else torch.ones_like(x)
    v = v.to(x.device)
    return v

def zeros_like(x):
    return label_like(0, x)

def ones_like(x):
    return label_like(1, x)


def calc_gan_loss(discriminator, output, target, word_embs):
    y_pred_fake, sim_n = discriminator(output.detach(), target.detach(), word_embs.detach())
    y_pred, sim = discriminator(target.detach(), output.detach(), word_embs.detach())

    g_loss = (torch.mean((y_pred - torch.mean(y_pred_fake) + 1.) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) - 1.) ** 2))/2
    d_loss = (torch.mean((y_pred - torch.mean(y_pred_fake) - 1.) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) + 1.) ** 2))/2
    d_loss += (nn.BCELoss()(sim, ones_like(sim)) + nn.BCELoss()(sim_n, zeros_like(sim_n))) / 2
    g_loss += nn.BCELoss()(sim_n, ones_like(sim_n))
    return g_loss, d_loss


# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):

    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def sent_loss(cnn_code, rnn_code, labels, class_ids, batch_size, use_cuda=True, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if class_ids is not None:
        for i in range(batch_size):
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if use_cuda:
            masks = masks.cuda()

    # --> seq_len x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: seq_len x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * 10.0

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    if class_ids is not None:
        scores0.data.masked_fill_(masks, -float('inf'))
    scores1 = scores0.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features, words_emb, labels, cap_lens, class_ids, batch_size, use_cuda=True,):
    
    masks = []
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        if class_ids is not None:
            mask = (class_ids == class_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features
        
        weiContext, attn = func_attention(word, context, 5.0)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        row_sim.mul_(5.0).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if use_cuda:
            masks = masks.cuda()

    similarities = similarities * 10.0
    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps


class VGGFeature(nn.Module):

    def __init__(self):
        super().__init__()

        vgg16 = models.vgg16(pretrained=True)
        for para in vgg16.parameters():
            para.requires_grad = False

        self.vgg16_pool_1 = nn.Sequential(*vgg16.features[0:5])
        self.vgg16_pool_2 = nn.Sequential(*vgg16.features[5:10])
        self.vgg16_pool_3 = nn.Sequential(*vgg16.features[10:17])

    def forward(self, x):
        pool_1 = self.vgg16_pool_1(x)
        pool_2 = self.vgg16_pool_2(pool_1)
        pool_3 = self.vgg16_pool_3(pool_2)

        return [pool_1, pool_2, pool_3]


class PerceptualLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, vgg_results, vgg_targets):
        loss = 0.
        for i, (vgg_res, vgg_target) in enumerate(
                zip(vgg_results, vgg_targets)):
            for feat_res, feat_target in zip(vgg_res, vgg_target):
                loss += self.l1loss(feat_res, feat_target)
        return loss / len(vgg_results)


class StyleLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()

    def gram(self, feature):
        n, c, h, w = feature.shape

        feature = feature.view(n, c, h * w)
        featureT = torch.transpose(feature, 1, 2).contiguous()
        gram_mat = torch.bmm(feature, featureT)
        return gram_mat / (c * h * w)

    def forward(self, vgg_results, vgg_targets):
        loss = 0.
        for i, (vgg_res, vgg_target) in enumerate(
                zip(vgg_results, vgg_targets)):
            for feat_res, feat_target in zip(vgg_res, vgg_target):
                loss += self.l1loss(
                    self.gram(feat_res), self.gram(feat_target))
        return loss / len(vgg_results)


class TotalVariationLoss(nn.Module):

    def __init__(self, c_img=3):
        super().__init__()
        self.c_img = c_img

        kernel = torch.FloatTensor([
            [0, 1, 0],
            [1, -2, 0],
            [0, 0, 0]]).view(1, 1, 3, 3)
        kernel = torch.cat([kernel] * c_img, dim=0)
        self.register_buffer('kernel', kernel)

    def gradient(self, x):
        return nn.functional.conv2d(
            x, self.kernel, stride=1, padding=1, groups=self.c_img)

    def forward(self, results, mask):
        loss = 0.
        for i, res in enumerate(results):
            grad = self.gradient(res) * mask
            loss += torch.mean(torch.abs(grad))
        return loss / len(results)


class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :]-x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :,1 :]-x[:, :, :, :w_x-1]), 2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        sobel_kernel = np.repeat(sobel_kernel, 3, axis=1)
        sobel_kernel = np.repeat(sobel_kernel, 3, axis=0)
        conv_op.weight.data = torch.from_numpy(sobel_kernel)
        self.conv_op = conv_op
        self.l1loss = nn.L1Loss()

    def forward(self, results, images):
        fake_edge = self.conv_op(results)
        real_edge = self.conv_op(images)
        edge_loss = self.l1loss(real_edge, fake_edge)

        return edge_loss

class InpaintLoss(nn.Module):

    def __init__(self, w_percep=0.05, w_style=3., w_tv=0.1, w_edge=0.5):
        super().__init__()

        self.w_percep = w_percep
        self.w_style = w_style
        self.w_tv = w_tv
        self.w_edge = w_edge

        self.vgg_feature = VGGFeature()
        self.style_loss = StyleLoss()
        self.perceptual_loss = PerceptualLoss()
        self.tv_loss = TotalVariationLoss()
        self.edge_loss = EdgeLoss()

    def forward(self, result, target, mask):
        vgg_r = [self.vgg_feature(result)]
        vgg_t = [self.vgg_feature(target)]

        loss_style = self.style_loss(vgg_r, vgg_t)
        loss_percep = self.perceptual_loss(vgg_r, vgg_t)
        loss_tv = self.tv_loss([result], mask)
        loss_edge = self.edge_loss(result, target)

        return loss_percep * self.w_percep + loss_tv * self.w_tv + loss_style * self.w_style + loss_edge * self.w_edge





