import argparse
import os

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms

from nltk.tokenize import RegexpTokenizer
from model import InpaintNet, RNN_ENCODER

from datasets import TextDataset
from datasets import prepare_data
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='birds')  # birds flowers celeba15000
parser.add_argument('--root', type=str, default='F:/crossModalDataset/birds')
parser.add_argument('--CAPTIONS_PER_IMAGE', type=int, default=10)
parser.add_argument('--checkpoint', type=str, default='../../',
                    help='The filename of pickle checkpoint.')
parser.add_argument('--WORDS_NUM', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--image_size', type=int, default=256)
args = parser.parse_args()

def get_mask():
    mask = []
    IMAGE_SIZE = args.image_size

    for i in range(args.batch_size):
        q1 = p1 = IMAGE_SIZE // 4
        q2 = p2 = IMAGE_SIZE - IMAGE_SIZE // 4

        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        m[q1:q2 + 1, p1:p2 + 1] = 1
        m = np.expand_dims(m, 0)
        mask.append(m)

    mask = np.array(mask)
    mask = torch.from_numpy(mask)

    if use_cuda:
        mask = mask.float().cuda()

    return mask

def process_cap(caption):
    cap = caption.replace("\ufffd\ufffd", " ")
    # picks out sequences of alphanumeric characters as tokens
    # and drops everything else
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(cap.lower())
    tokens_new = []
    for t in tokens:
        t = t.encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0:
            tokens_new.append(t)
    return tokens_new

def get_caption(caption, wordtoix, WORDS_NUM=16):
    caption = process_cap(caption)
    ix_cap = []
    for w in caption:
        if w in wordtoix:
            ix_cap.append(wordtoix[w])
    # a list of indices for a sentence
    sent_caption = np.asarray(ix_cap).astype('int64')
    if (sent_caption == 0).sum() > 0:
        print('ERROR: do not need END (0) token', sent_caption)
    num_words = len(sent_caption)
    # pad with 0s (i.e., '<end>')
    x = np.zeros((WORDS_NUM, 1), dtype='int64')
    x_len = num_words
    if num_words <= WORDS_NUM:
        x[:num_words, 0] = sent_caption
    else:
        ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
        np.random.shuffle(ix)
        ix = ix[:WORDS_NUM]
        ix = np.sort(ix)
        x[:, 0] = sent_caption[ix]
        x_len = WORDS_NUM
    x = torch.tensor(x)
    x_len = torch.tensor([x_len])
    return x, x_len


def prepare_data(img_path, captions, lens, transform, norm):
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    img = norm(img)
    img = img.unsqueeze(0)
    if torch.cuda.is_available():
        img = Variable(img).cuda()
    else:
        img = Variable(img)

    captions = captions.squeeze()
    captions = captions.view(1, -1)
    if torch.cuda.is_available():
        captions = Variable(captions).cuda()
        lens = Variable(lens).cuda()
    else:
        captions = Variable(captions)
        lens = Variable(lens)

    return [img, captions, lens]



if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    size = (args.image_size, args.image_size)
    train_tf = transforms.Compose([
    transforms.Resize(size)
    ])

    dataset_test = TextDataset(args.root, 'test',
                           base_size=args.image_size,
                           CAPTIONS_PER_IMAGE=args.CAPTIONS_PER_IMAGE,
                           WORDS_NUM=args.WORDS_NUM,
                           transform=train_tf)
    assert dataset_test
    test_set = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, drop_last=True)

    print(len(test_set))

    ixtoword_test = dataset_test.ixtoword
    wordtoix_test = dataset_test.wordtoix

    text_encoder = RNN_ENCODER(dataset_test.n_words, nhidden=args.image_size)
    text_encoder_path = 'F:/crossModalDataset/DAMSMencoders/' + args.dataset + '/text_encoder200.pth'
    state_dict = torch.load(text_encoder_path, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    for p in text_encoder.parameters():
        p.requires_grad = False
    print('Load text encoder from:', text_encoder_path)
    text_encoder.eval()

    if use_cuda:
        text_encoder = text_encoder.cuda()

    nz = 100
    noise = Variable(torch.FloatTensor(args.batch_size, nz))
    if use_cuda:
        noise = noise.cuda()



    g_model = InpaintNet().to(device)
    g_checkpoint = torch.load(args.checkpoint + 'G_660000.pth', map_location=device)
    g_model.load_state_dict(g_checkpoint)
    g_model.eval()
    # caption = "this small bird has a yellow body, a small pointy yellow bill."
    # caption = "this small bird has a red body, a small pointy red bill."

    # caption = "this small bird has a dark blue body, a small pointy blue bill."
    # caption = "this small bird has a very black body, a small pointy black bill."
    # caption = "this small bird has a dark grey body, a small pointy grey bill."
    # caption = "this small bird has a brown body, a small pointy brown bill."
    caption = "this small bird has a grey body, a small pointy grey bill."
    # caption = "this small bird has a green body, a small pointy green bill."



    img_dir = "D:/FCJ/Brewer_Blackbird_0065_2310/Brewer_Blackbird_0065_2310"
    img_path = img_dir + ".png"
    index = "real_grey"
    caption, len = get_caption(caption, wordtoix_test)
    norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img, captions, cap_lens = prepare_data(img_path, caption, len, train_tf, norm)

    #         #
    hidden = text_encoder.init_hidden(args.batch_size)
    words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
    text_mask = (captions == 0)
    num_words = words_embs.size(2)
    if text_mask.size(1) > num_words:
        text_mask = text_mask[:, :num_words]

    mask = get_mask()
    masked = img * (1. - mask)
    #         #
    noise.data.normal_(0, 1)
    #         # # coarse_result, refine_result = g_model(masked, mask, noise, sent_emb, words_embs, text_mask)
    refine_result = g_model(masked, mask, noise, sent_emb, words_embs, text_mask)
    #
    #
    #
    for bb in range(args.batch_size):
        # only
        ims_test = refine_result.add(1).div(2).mul(255).clamp(0, 255).byte()
        ims_test = ims_test[bb].permute(1, 2, 0).data.cpu().numpy()
        ims_out = Image.fromarray(ims_test)
        fullpath = img_dir + "%s.png" % (index)
        ims_out.save(fullpath)
    #
    #             # real
    #             ims_test = img.add(1).div(2).mul(255).clamp(0, 255).byte()
    #             ims_test = ims_test[bb].permute(1, 2, 0).data.cpu().numpy()
    #             ims_out = Image.fromarray(ims_test)
    #             fullpath = '%s/%s.png' % (args.save_real + str(i * 10000), keys[bb].split('/')[-1])
    #             ims_out.save(fullpath)

