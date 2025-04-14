import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import DualTransformer, Transformer
import math
import copy
import pdb

class SCANet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config['dropout']
        self.vocab_size = config['vocab_size']
        self.sigma = config["sigma"]
        self.use_negative = config['use_negative']
        self.num_props = config['num_props']
        self.max_epoch = config['max_epoch']
        self.gamma = config['gamma']

        self.frame_fc = nn.Linear(config['frames_input_size'], config['hidden_size'])
        self.word_fc = nn.Linear(config['words_input_size'], config['hidden_size'])
        
        self.mask_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        
        self.pred_vec = nn.Parameter(torch.zeros(config['frames_input_size']).float(), requires_grad=True)
        self.mask_f_vec = nn.Parameter(torch.zeros(config['frames_input_size']).float(), requires_grad=True)

        self.attention = DualTransformer(**config['DualTransformer'])
        self.fc_comp = nn.Linear(config['hidden_size'], self.vocab_size)
        self.fc_gauss = nn.Linear(config['hidden_size'], self.num_props*2)
        self.fc_frame = nn.Linear(config['hidden_size'], config['frames_input_size'])
 
        self.word_pos_encoder = SinusoidalPositionalEmbedding(config['hidden_size'], 0, 20)
        
        self.K = 8
        self.code_book = self.make_book(config['frames_input_size'], K=self.K)

    def make_book(self, hidden_size, K=8):
        base = torch.ones((K, hidden_size)).float()
        scale = torch.arange(1, K+1).float().div(K).unsqueeze(1)  # shape: (K, 1)
        Z = base.cuda() * scale.cuda()  # shape: (K, hidden_size), broadcasting
        Z = nn.Parameter(Z, requires_grad=True)
        return Z


    def forward(self, frames_feat, frames_len, words_id, words_feat, words_len, weights, **kwargs):
        bsz, n_frames, D = frames_feat.shape
        K, d = self.code_book.shape

        alpha = kwargs["sce"].long().to(self.code_book.device)
        alpha = torch.clamp(alpha, max=self.K - 1)
        z_alpha = self.code_book[alpha].view(-1, 1, D)
        frames_feat_ref = copy.deepcopy(frames_feat)
        
        ## Input representation
        #frames_feat_z = torch.cat([frames_feat, z_alpha], dim=1)
        frames_feat = F.dropout(frames_feat, self.dropout, self.training)
        frames_feat = self.frame_fc(frames_feat)
        frames_mask = _generate_mask(frames_feat, frames_len)
        
        z_feat = self.frame_fc(z_alpha)
        frames_feat_z = torch.cat([frames_feat, z_feat], dim=1)
        frames_mask_z = _generate_mask(frames_feat_z, frames_len + 1)

        words_feat[:, 0] = self.start_vec.cuda()
        words_pos = self.word_pos_encoder(words_feat)
        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)
        words_mask = _generate_mask(words_feat, words_len + 1)
        
        _, vz = self.attention(frames_feat_z, frames_mask_z, words_feat + words_pos, words_mask, decoding=1)
        gauss_param = torch.sigmoid(self.fc_gauss(vz[:, -1])).view(bsz*self.num_props, 2)
        gauss_center = gauss_param[:, 0]
        gauss_width = gauss_param[:, 1]
        
        props_len = n_frames//4
        keep_idx = torch.linspace(0, n_frames-1, steps=props_len).long()
        frames_feat_ = frames_feat[:, keep_idx]
        frames_mask_ = frames_mask[:, keep_idx]
        props_feat = frames_feat_.unsqueeze(1).expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, props_len, -1)
        props_mask = frames_mask_.unsqueeze(1).expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        
        ## Flatten Gaussian mask 
        gauss_weight = self.Flatten_Gauss_Weight(props_len, gauss_center, gauss_width)
        
        ## Complexity-Adaptive Proposal Enhancement
        words_feat, masked_words = self._mask_words(words_feat, words_len, weights=weights)
        words_feat = words_feat + words_pos
        words_feat = words_feat[:, :-1]
        words_mask = words_mask[:, :-1]

        words_mask1 = words_mask.unsqueeze(1).expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        words_id1 = words_id.unsqueeze(1).expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        words_feat1 = words_feat.unsqueeze(1).expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, words_mask1.size(1), -1)
        
        ## Cross-modal Reconstruction
        frames_feat_src, masked_frames = self._mask_frames(frames_feat, frames_len, weights=None)
        _, frames_rec = self.attention(frames_feat_src, frames_mask, words_feat, words_mask, decoding=1)
        f_logit = self.fc_frame(frames_rec)
        
        pos_weight = gauss_weight/gauss_weight.max(dim=-1, keepdim=True)[0]
        hv, h, attn_weight = self.attention(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=pos_weight, need_weight=True)
        w_logit = self.fc_comp(h)

        ## Hierachical Contrastive Learning
        if self.use_negative:
            neg_1_weight, neg_2_weight = self.negative_proposal_mining(props_len, gauss_center, gauss_width, kwargs['epoch'])
            
            _, neg_h_1 = self.attention(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=neg_1_weight)
            neg_w_logit_1 = self.fc_comp(neg_h_1)
  
            _, neg_h_2 = self.attention(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=neg_2_weight)
            neg_w_logit_2 = self.fc_comp(neg_h_2)

            _, ref_h = self.attention(frames_feat_, frames_mask_, words_feat, words_mask, decoding=2)
            ref_w_logit = self.fc_comp(ref_h)
        else:
            neg_w_logit_1 = None
            neg_w_logit_2 = None
            ref_w_logit = None

        return {
            'neg_words_logit_1': neg_w_logit_1,
            'neg_words_logit_2': neg_w_logit_2,
            'ref_words_logit': ref_w_logit,
            'words_logit': w_logit,
            'words_id': words_id,
            'words_mask': words_mask,
            'width': gauss_width,
            'center': gauss_center,
            'gauss_weight': gauss_weight,
            'sce': kwargs['sce'], # for Dynamic Calibration
            'f_logit': f_logit,
            'masked_frames': masked_frames, 
            'frames_feat_ref': frames_feat_ref,
        }



    def Flatten_Gauss_Weight(self, props_len, center, width):
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)

        weight2 = torch.linspace(0, 1, props_len)
        weight2 = weight2.view(1, -1).expand(center.size(0), -1).to(center.device)

        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / self.sigma

        w = 0.3989422804014327

        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))
        weight = weight/weight.max(dim=-1, keepdim=True)[0]

        weight2 = w/width*torch.exp(-(weight2-center)**2/(2*width**2))
        weight2 = weight2/weight2.max(dim=-1, keepdim=True)[0]

        st = torch.clip(torch.round((center-width/2)*50), min=0, max=49)
        ed = torch.clip(torch.round((center+width/2)*50), min=0, max=49)
        st = st.to(torch.long)
        ed = ed.to(torch.long)
        a,b = weight.shape
        for i in range(a):
            stx = st[i]
            edx = ed[i]
            weight2[i,stx:edx+1] =  torch.mean(weight[i,stx:edx+1])

        return weight2

    def negative_proposal_mining(self, props_len, center, width, epoch):
        def Gauss(pos, w1, c):
            w1 = w1.unsqueeze(-1).clamp(1e-2) / (self.sigma/2)
            c = c.unsqueeze(-1)
            w = 0.3989422804014327
            y1 = w/w1*torch.exp(-(pos-c)**2/(2*w1**2))
            return y1/y1.max(dim=-1, keepdim=True)[0]

        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)

        left_width = torch.clamp(center-width/2, min=0)
        left_center = left_width * min(epoch/self.max_epoch, 1)**self.gamma * 0.5
        right_width = torch.clamp(1-center-width/2, min=0)
        right_center = 1 - right_width * min(epoch/self.max_epoch, 1)**self.gamma * 0.5

        left_neg_weight = Gauss(weight, left_center, left_center)
        right_neg_weight = Gauss(weight, 1-right_center, right_center)

        return left_neg_weight, right_neg_weight

    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)
        token = self.word_fc(token)

        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = max(l // 3, 1) 
            masked_words.append(torch.zeros([words_feat.size(1)]).byte().cuda())
            if l < 1:
                continue
            p = weights[i, :l].cpu().numpy() if weights is not None else None
            choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            masked_words[-1][choices] = 1
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1, masked_words
    
    def _mask_frames(self, frames_feat, frames_len, weights=None):

        token = self.mask_f_vec.cuda().unsqueeze(0).unsqueeze(0)
        token = self.frame_fc(token)

        masked_frames = []
        for i, l in enumerate(frames_len):
            l = int(l)
            num_masked = max(l // 10, 1)
            masked_frames.append(torch.zeros([frames_feat.size(1)], dtype=torch.uint8).cuda())
            if l < 1:
                continue
            # use weights for sampling if provided
            if weights is not None:
                p = weights[i, :l].cpu().numpy()
            else:
                p = np.ones(l) / l  # uniform distribution
            choices = np.random.choice(np.arange(0, l), num_masked, replace=False, p=p)
            masked_frames[-1][choices] = 1
        masked_frames = torch.stack(masked_frames, 0).unsqueeze(-1)  # [B, T, 1]
        masked_frames_vec = frames_feat.new_zeros(*frames_feat.size()) + token  # same shape, filled with token
        masked_frames_vec = masked_frames_vec.masked_fill_(masked_frames == 0, 0)
        masked_feat = frames_feat.masked_fill(masked_frames == 1, 0) + masked_frames_vec
        return masked_feat, masked_frames


def _generate_mask(x, x_len):
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        for l in x_len:
            mask.append(torch.zeros([x.size(1)]).byte().cuda())
            mask[-1][:l] = 1
        mask = torch.stack(mask, 0)
    return mask


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.cuda(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number
