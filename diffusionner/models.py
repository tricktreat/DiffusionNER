from collections import namedtuple
import copy
import math
import random
import torch
from torch import nn as nn
from torch.nn import functional as F
from diffusionner.modeling_albert import AlbertModel, AlbertConfig
from diffusionner.modeling_bert import BertConfig, BertModel
from diffusionner.modeling_roberta import RobertaConfig, RobertaModel
from diffusionner.modeling_xlm_roberta import XLMRobertaConfig
from transformers.modeling_utils import PreTrainedModel
from diffusionner import util
import logging

logger = logging.getLogger()

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class EntityBoundaryPredictor(nn.Module):
    def __init__(self, config, prop_drop = 0.1):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.token_embedding_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size)
        ) 
        self.entity_embedding_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size)
        ) 
        self.boundary_predictor = nn.Linear(self.hidden_size, 1)
    
    def forward(self, token_embedding, entity_embedding, token_mask):
        # B x #ent x #token x hidden_size
        entity_token_matrix = self.token_embedding_linear(token_embedding).unsqueeze(1) + self.entity_embedding_linear(entity_embedding).unsqueeze(2)
        entity_token_cls = self.boundary_predictor(torch.relu(entity_token_matrix)).squeeze(-1)
        token_mask = token_mask.unsqueeze(1).expand(-1, entity_token_cls.size(1), -1)
        entity_token_cls[~token_mask] = -1e25
        # entity_token_p = entity_token_cls.softmax(dim=-1)
        entity_token_p = F.sigmoid(entity_token_cls)
        return entity_token_p


class EntityTypePredictor(nn.Module):
    def __init__(self, config, entity_type_count):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, entity_type_count),
        )
    
    def forward(self, h_cls):
        entity_logits = self.classifier(h_cls)
        return entity_logits


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class SpanAttentionLayer(nn.Module):
    def __init__(self, d_model=768, d_ffn=1024, dropout=0.1, activation="relu", n_heads=8, self_attn = True, cross_attn = True):
        super().__init__()

        self.self_attn_bool = self_attn
        self.cross_attn_bool = cross_attn

        if self.cross_attn_bool:
            # cross attention
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout1 = nn.Dropout(dropout)
            self.norm1 = nn.LayerNorm(d_model)
        if self.self_attn_bool:
            # self attention
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(d_model)
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
        
    def forward(self, tgt, pos, src, mask):
        if self.self_attn_bool:
            # self attention
            q = k = self.with_pos_embed(tgt, pos)
            v = tgt
            tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))[0].transpose(0, 1)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
        
        if self.cross_attn_bool:
            # cross attention
            q = self.with_pos_embed(tgt, pos)
            k = v = src
            tgt2 = self.cross_attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), key_padding_mask=~mask if mask is not None else None)[0].transpose(0, 1)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class SpanAttention(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(self, tgt, pos, src, mask):
        output = tgt

        for lid, layer in enumerate(self.layers):
            output = layer(output, pos, src, mask)

        return output

def span_lw_to_lr(x):
    l, w = x.unbind(-1)
    b = [l, l + w]
    return torch.stack(b, dim=-1)


def span_lr_to_lw(x):
    l, r = x.unbind(-1)
    b = [l, r-l]
    return torch.stack(b, dim=-1)

def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end+1] = 1
    return mask


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def constant_beta_schedule(timesteps):
    scale = 1000 / timesteps
    constant = scale * 0.01
    return torch.tensor([constant] * timesteps, dtype = torch.float64)


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h

class DiffusionNER(PreTrainedModel):

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def __init__(
        self,
        model_type, 
        config, 
        entity_type_count,
        lstm_layers = 0,
        span_attn_layers = 0,
        timesteps = 1000,
        beta_schedule = "cosine",
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        sampling_timesteps = 5,
        num_proposals = 100,
        scale = 3.0,
        extand_noise_spans = 'repeat',
        span_renewal = False,
        step_ensemble = False,
        prop_drop = 0.1, 
        soi_pooling = "maxpool+lrconcat",
        pos_type = "sine",
        step_embed_type = "add",
        sample_dist_type = "normal",
        split_epoch = 0,
        pool_type = "max",
        wo_self_attn = False,
        wo_cross_attn = False):
        super().__init__(config)
        self.model_type = model_type
        self._entity_type_count = entity_type_count
        self.pool_type = pool_type
        self.span_attn_layers = span_attn_layers
        self.soi_pooling = soi_pooling
        self.pos_type = pos_type
        self.step_embed_type = step_embed_type
        self.sample_dist_type = sample_dist_type

        # build backbone
        if model_type == "roberta":
            self.roberta = RobertaModel(config)
            self.model = self.roberta

        if model_type == "bert":
            self.bert = BertModel(config)
            self.model = self.bert
            for name, param in self.bert.named_parameters():
                if "pooler" in name:
                    param.requires_grad = False

        if model_type == "albert":
            self.albert = AlbertModel(config)
            self.model = self.albert
        
        self.lstm_layers = lstm_layers
        if self.lstm_layers>0:
            self.lstm = nn.LSTM(input_size = config.hidden_size, hidden_size = config.hidden_size//2, num_layers = self.lstm_layers,  bidirectional = True, dropout = prop_drop, batch_first = True)
        
        DiffusionNER._keys_to_ignore_on_save = ["model." + k for k,v in self.model.named_parameters()]
        DiffusionNER._keys_to_ignore_on_load_missing = ["model." + k for k,v in self.model.named_parameters()]

        # build head
        self.prop_drop = prop_drop
        self.dropout = nn.Dropout(prop_drop)

        if "lrconcat" in self.soi_pooling:
            self.downlinear = nn.Linear(config.hidden_size*2, config.hidden_size)
            self.affine_start = nn.Linear(config.hidden_size, config.hidden_size)
            self.affine_end = nn.Linear(config.hidden_size, config.hidden_size)

        if "|" in soi_pooling:
            n = len(soi_pooling.split("|"))
            self.soi_pooling_downlinear = nn.Sequential(
                    nn.Linear(config.hidden_size*n, config.hidden_size),
                    nn.GELU()
                )

        if self.span_attn_layers > 0:
            if self.pos_type == "sine":
                self.pos_embeddings = nn.Sequential(
                    SinusoidalPositionEmbeddings(config.hidden_size),
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.GELU(),
                    nn.Linear(config.hidden_size, config.hidden_size),
                )
            spanattentionlayer = SpanAttentionLayer(d_model=config.hidden_size, self_attn = not wo_self_attn, cross_attn = not wo_cross_attn)
            self.spanattention = SpanAttention(spanattentionlayer, num_layers=self.span_attn_layers)
        self.left_boundary_predictor = EntityBoundaryPredictor(config)
        self.right_boundary_predictor = EntityBoundaryPredictor(config)
        self.entity_classifier = EntityTypePredictor(config, entity_type_count)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        if self.step_embed_type == 'scaleshift':
            self.step_scale_shift = nn.Sequential(nn.SiLU(), nn.Linear(config.hidden_size, config.hidden_size*2))
        self.split_epoch = split_epoch
        self.has_changed = True

        if self.split_epoch > 0:
            self.has_changed = False
            logger.info(f"Freeze bert weights from begining")
            logger.info("Freeze transformer weights")
            if self.model_type == "bert":
                model = self.bert
            if self.model_type == "roberta":
                model = self.roberta
            if self.model_type == "albert":
                model = self.albert
            for name, param in model.named_parameters():
                param.requires_grad = False

        self.init_weights()

        # build diffusion
        self.num_proposals = num_proposals
        timesteps = timesteps
        sampling_timesteps = sampling_timesteps
        self.objective = 'pred_x0'
        betas = None
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'constant':
            betas = constant_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.extand_noise_spans = extand_noise_spans

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = scale
        self.span_renewal = span_renewal
        self.step_ensemble = step_ensemble

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
    
    def model_predictions(self, span, h_token, h_token_lstm, timestep, token_masks, x_self_cond=None, clip_x_start=False):
        x_span = torch.clamp(span, min=-1 * self.scale, max=self.scale) # -scale -- +scale
        x_span = ((x_span / self.scale) + 1) / 2 # 0 -- 1
        x_span = span_lw_to_lr(x_span) # maybe r > 1 
        x_span = torch.clamp(x_span, min=0, max=1)
        outputs_logits, outputs_span, left_entity_token_p, right_entity_token_p = self.head(x_span, h_token, h_token_lstm, timestep, token_masks)
        
        token_count = token_masks.long().sum(-1,keepdim=True)
        token_count_expanded = token_count.unsqueeze(1).expand(-1, span.size(1), span.size(2))

        x_start = outputs_span  # (batch, num_proposals, 4) predict spans: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / (token_count_expanded - 1 + 1e-20)
        # x_start = x_start / token_count_expanded
        x_start = span_lr_to_lw(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(span, timestep, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_logits, outputs_span, left_entity_token_p, right_entity_token_p

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, span, h_token, h_token_lstm, time_cond, token_masks, self_cond = None, clip_denoised = True):
        preds, outputs_class, outputs_coord, left_entity_token_p, right_entity_token_p = self.model_predictions(span, h_token, h_token_lstm, time_cond, token_masks,
                                        self_cond, clip_x_start=clip_denoised)
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = span, t = time_cond)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def sample(self, h_token, h_token_lstm, token_masks):
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(h_token, h_token_lstm, token_masks)

    @torch.no_grad()
    def p_sample(self, span, h_token, h_token_lstm, t, token_masks, x_self_cond = None, clip_denoised = True):
        b, *_, device = *span.shape, span.device
        batched_times = torch.full((span.shape[0],), t, device = span.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(span, h_token, h_token_lstm, batched_times, token_masks, self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(span) if t > 0 else 0. # no noise if t == 0
        pred = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred, x_start


    @torch.no_grad()
    def p_sample_loop(self, h_token, h_token_lstm, token_masks):
        batch = token_masks.shape[0]
        shape = (batch, self.num_proposals, 2)
        span = torch.randn(shape, device=device)

        x_start = None

        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            span, x_start = self.p_sample(span, h_token, h_token_lstm, t, token_masks, self_cond)
        return span

    @torch.no_grad()
    def ddim_sample(self, h_token, h_token_lstm, token_masks, clip_denoised=True):
        batch = token_masks.shape[0]
        shape = (batch, self.num_proposals, 2)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        if self.sample_dist_type == "normal":
            span = torch.randn(shape, device=self.device)
        elif self.sample_dist_type == "uniform":
            span = (2*torch.rand(shape, device=self.device) - 1) * self.scale

        x_start = None
        step_ensemble_outputs_class = []
        step_ensemble_outputs_coord = []
        step_ensemble_left_entity_token_p = []
        step_ensemble_right_entity_token_p = []
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_coord, left_entity_token_p, right_entity_token_p  = self.model_predictions(span, h_token, h_token_lstm, time_cond, token_masks,
                                                                         self_cond, clip_x_start=clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if time_next < 0:
                span = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            if self.sample_dist_type == "normal":
                noise = torch.randn_like(span)
            elif self.sample_dist_type == "uniform":
                noise = torch.rand_like(span)

            span = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if self.span_renewal:  # filter
                score_per_span, boundary_per_span = outputs_class, outputs_coord
                threshold = 0.0
                score_per_span = F.softmax(score_per_span, dim=-1)
                value, _ = torch.max(score_per_span, -1, keepdim=False)
                keep_idx = value > threshold
                keep_idx = keep_idx * (boundary_per_span[:, :, 1] >= boundary_per_span[:, :, 0])
                num_remain = torch.sum(keep_idx)
                span[~keep_idx] = torch.randn(self.num_proposals * span.size(0) - num_remain, 2, device=span.device).double()
            
            if self.step_ensemble:
                step_ensemble_outputs_class.append(outputs_class)
                step_ensemble_outputs_coord.append(outputs_coord)
                step_ensemble_left_entity_token_p.append(left_entity_token_p)
                step_ensemble_right_entity_token_p.append(right_entity_token_p)

        output = {'pred_logits': outputs_class, 'pred_spans': outputs_coord, "pred_left": left_entity_token_p, "pred_right": right_entity_token_p}
        if self.step_ensemble:
            output = {'pred_logits': torch.cat(step_ensemble_outputs_class, dim = 1), 'pred_spans': torch.cat(step_ensemble_outputs_coord, dim = 1), 
            "pred_left": torch.cat(step_ensemble_left_entity_token_p, dim = 1), "pred_right": torch.cat(step_ensemble_right_entity_token_p, dim = 1)}
        return output


    def forward(self, 
            encodings: torch.tensor,
            context_masks: torch.tensor,
            token_masks:torch.tensor,
            context2token_masks:torch.tensor,
            pos_encoding: torch.tensor = None,
            seg_encoding: torch.tensor = None,
            entity_spans: torch.tensor = None, 
            entity_types: torch.tensor = None, 
            entity_masks: torch.tensor = None, 
            meta_doc = None,
            epoch = None):

        # Feature Extraction.
        h_token, h_token_lstm = self.backbone(encodings, 
                                context_masks, 
                                token_masks,
                                pos_encoding, 
                                seg_encoding, 
                                context2token_masks)

        # Prepare Proposals.
        if not self.training:
            results = self.ddim_sample(h_token, h_token_lstm, token_masks)
            return results

        if self.training:
            if not self.has_changed and epoch >= self.split_epoch:
                logger.info(f"Now, update bert weights @ epoch = {epoch}")
                self.has_changed = True
                for name, param in self.named_parameters():
                    param.requires_grad = True

            d_spans, noises, t = self.prepare_targets(entity_spans, entity_types, entity_masks, token_masks, meta_doc = meta_doc)
            t = t.squeeze(-1)
            outputs_class, outputs_span, left_entity_token_p, right_entity_token_p = self.head(d_spans, h_token, h_token_lstm, t, token_masks)
            output = {'pred_logits': outputs_class, 'pred_spans': outputs_span, 'pred_left': left_entity_token_p, 'pred_right': right_entity_token_p}

            return output

    def prepare_diffusion_repeat(self, gt_spans, gt_num):
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 2, device=self.device)

        num_gt = gt_num.item()
        gt_spans = gt_spans[:gt_num]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_spans = torch.as_tensor([[0., 1.]], dtype=torch.float, device=gt_spans.device)
            num_gt = 1

        num_repeat = self.num_proposals // num_gt  # number of repeat except the last gt box in one image
        repeat_tensor = [num_repeat] * (num_gt - self.num_proposals % num_gt) + [num_repeat + 1] * (
                self.num_proposals % num_gt)
        assert sum(repeat_tensor) == self.num_proposals
        random.shuffle(repeat_tensor)
        repeat_tensor = torch.tensor(repeat_tensor, device=self.device)

        gt_spans = (gt_spans * 2. - 1.) * self.scale
        x_start = torch.repeat_interleave(gt_spans, repeat_tensor, dim=0)

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_spans = span_lw_to_lr(x)
        diff_spans = torch.clamp(diff_spans, min=0, max=1)

        return diff_spans, noise, t

    def prepare_diffusion_concat(self, gt_spans, gt_num):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 2, device=self.device)
        
        num_gt = gt_num.item()
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_spans = torch.as_tensor([[0., 1.]], dtype=torch.float, device=gt_spans.device)
            num_gt = 1

        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 2,
                                          device=self.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 1:] = torch.clip(box_placeholder[:, 1:], min=1e-4)
            x_start = torch.cat((gt_spans, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_spans[select_mask]
        else:
            x_start = gt_spans

        x_start = (x_start * 2. - 1.) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_spans = span_lw_to_lr(x)
        diff_spans = torch.clamp(diff_spans, min=0, max=1)

        return diff_spans, noise, t

    def prepare_targets(self, entity_spans, entity_types, entity_masks, token_masks, meta_doc):
        diffused_spans = []
        noises = []
        ts = []
        token_count = token_masks.long().sum(-1,keepdim=True)
        for gt_spans, gt_types, entity_mask, sent_length in zip(entity_spans, entity_types, entity_masks, token_count):
            gt_num = entity_mask.sum()
            target = {}
            gt_spans = gt_spans / sent_length
            gt_spans = span_lr_to_lw(gt_spans)
            d_spans = d_noise = d_t = None
            if self.extand_noise_spans == "concat":
                d_spans, d_noise, d_t = self.prepare_diffusion_concat(gt_spans, gt_num)
            elif self.extand_noise_spans == "repeat":
                d_spans, d_noise, d_t = self.prepare_diffusion_repeat(gt_spans, gt_num)
            
            diffused_spans.append(d_spans)
            noises.append(d_noise)
            ts.append(d_t)

        return torch.stack(diffused_spans), torch.stack(noises), torch.stack(ts)

    def backbone(self, 
        encodings: torch.tensor, 
        context_masks: torch.tensor, 
        token_masks: torch.tensor,
        pos_encoding: torch.tensor = None, 
        seg_encoding: torch.tensor = None, 
        context2token_masks:torch.tensor = None):

        outputs = self.model(
                    input_ids=encodings,
                    attention_mask=context_masks,
                    # token_type_ids=seg_encoding,
                    position_ids=pos_encoding,
                    output_hidden_states=True)
        
        h = outputs.hidden_states[-1]
        h_token = util.combine(h, context2token_masks, self.pool_type)

        h_token_lstm = None
        if self.lstm_layers > 0:
            token_count = token_masks.long().sum(-1,keepdim=True)
            h_token_lstm = nn.utils.rnn.pack_padded_sequence(input = h_token, lengths = token_count.squeeze(-1).cpu().tolist(), enforce_sorted = False, batch_first = True)
            h_token_lstm, (_, _) = self.lstm(h_token_lstm)
            h_token_lstm, _ = nn.utils.rnn.pad_packed_sequence(h_token_lstm, batch_first=True)

        return h_token, h_token_lstm

    def head(self, 
        span:torch.tensor,
        h_token:torch.tensor,
        h_token_lstm:torch.tensor,
        timestep:torch.tensor,
        token_masks:torch.tensor):
        
        token_count = token_masks.long().sum(-1,keepdim=True)
        token_count_expanded = token_count.unsqueeze(1).expand(-1, span.size(1), span.size(2))

        old_span = span
        span = old_span * (token_count_expanded - 1)
        # span = old_span * token_count_expanded
        
        span_mask = None
        if "pool" in self.soi_pooling:
            span_mask = []
            for tk, sp in zip(token_count, torch.round(span).to(dtype=torch.long)):
                sp_mask = []
                for s in sp:
                    sp_mask.append(create_entity_mask(*s, tk))
                span_mask.append(torch.stack(sp_mask))
            span_mask = util.padded_stack(span_mask).to(device=h_token.device)

        timestep_embeddings = self.time_mlp(timestep)

        left_entity_token_p, right_entity_token_p, entity_logits = self.left_right_type(h_token, h_token_lstm, span_mask, timestep_embeddings, span, token_count, token_masks)
        entity_left = left_entity_token_p.argmax(dim=-1)
        entity_right = right_entity_token_p.argmax(dim=-1)
        entity_spans = torch.stack([entity_left, entity_right], dim=-1)

        return entity_logits, entity_spans, left_entity_token_p, right_entity_token_p

    def left_right_type(self, h_token, h_token_lstm, span_mask, timestep_embeddings, span, token_count, token_masks):
        N, nr_spans = span.shape[:2]

        if h_token_lstm is None:
            h_token_lstm = h_token

        entity_spans_pools = []
        if "maxpool" in self.soi_pooling:
            pool_entity_spans_pool = util.combine(h_token_lstm, span_mask, "max")
            pool_entity_spans_pool = self.dropout(pool_entity_spans_pool)
            entity_spans_pools.append(pool_entity_spans_pool)

        if "meanpool" in self.soi_pooling:
            pool_entity_spans_pool = util.combine(h_token_lstm, span_mask, "mean")
            pool_entity_spans_pool = self.dropout(pool_entity_spans_pool)
            entity_spans_pools.append(pool_entity_spans_pool)

        if "sumpool" in self.soi_pooling:
            pool_entity_spans_pool = util.combine(h_token_lstm, span_mask, "sum")
            pool_entity_spans_pool = self.dropout(pool_entity_spans_pool)
            entity_spans_pools.append(pool_entity_spans_pool)

        if "lrconcat" in self.soi_pooling:
            entity_spans_token_inner = torch.round(span).to(dtype=torch.long)
            entity_spans_token_inner[:,:,0][entity_spans_token_inner[:,:,0]<0] = 0
            entity_spans_token_inner[:,:,1][entity_spans_token_inner[:,:,1]<0] = 0
            entity_spans_token_inner[:,:,0][entity_spans_token_inner[:,:,0]>=token_count] = token_count.repeat(1,entity_spans_token_inner.size(1))[entity_spans_token_inner[:,:,0]>=token_count] - 1
            entity_spans_token_inner[:,:,1][entity_spans_token_inner[:,:,1]>=token_count] = token_count.repeat(1,entity_spans_token_inner.size(1))[entity_spans_token_inner[:,:,1]>=token_count] - 1
            start_end_embedding_inner = util.batch_index(h_token_lstm, entity_spans_token_inner)

            start_affined = self.dropout(self.affine_start(start_end_embedding_inner[:,:,0]))
            end_affined = self.dropout(self.affine_end(start_end_embedding_inner[:,:,1]))
            
            embed_inner = [start_affined, end_affined]
            lrconcat_entity_spans_pool = self.dropout(self.downlinear(torch.cat(embed_inner, dim=2)))
            entity_spans_pools.append(lrconcat_entity_spans_pool)

        if len(entity_spans_pools) > 1:
            if "|" in self.soi_pooling:
                entity_spans_pool = torch.cat(entity_spans_pools, dim=-1)
                entity_spans_pool = self.soi_pooling_downlinear(entity_spans_pool)
            if "+" in self.soi_pooling:
                entity_spans_pool = torch.stack(entity_spans_pools, dim=0).sum(dim=0)
        else:
            entity_spans_pool = entity_spans_pools[0]

        if self.span_attn_layers>0:

            pos = None

            if self.pos_type == "same":
                pos = entity_spans_pool
            elif self.pos_type == "sine":
                pos = self.pos_embeddings(torch.arange(nr_spans).to(h_token_lstm.device)).repeat(N, 1, 1)
            entity_spans_pool = self.spanattention(entity_spans_pool, pos, h_token_lstm, token_masks)

        if self.step_embed_type == "add":
            entity_spans_pool = entity_spans_pool + timestep_embeddings.unsqueeze(1).repeat(1, nr_spans, 1)
        elif self.step_embed_type == "scaleshift":
            entity_spans_pool = entity_spans_pool.reshape(N * nr_spans, -1)
            scale_shift = self.step_scale_shift(timestep_embeddings)
            scale_shift = torch.repeat_interleave(scale_shift, nr_spans, dim=0)
            scale, shift = scale_shift.chunk(2, dim=1)
            entity_spans_pool = entity_spans_pool * (scale + 1) + shift
            entity_spans_pool = entity_spans_pool.view(N, nr_spans, -1)
            
        left_entity_token_p = self.left_boundary_predictor(h_token_lstm, entity_spans_pool, token_masks)
        right_entity_token_p = self.right_boundary_predictor(h_token_lstm, entity_spans_pool, token_masks)
        entity_logits = self.entity_classifier(entity_spans_pool)

        return left_entity_token_p, right_entity_token_p, entity_logits


class BertDiffusionNER(DiffusionNER):
    
    config_class = BertConfig
    base_model_prefix = "bert"
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, *args, **kwagrs):
        super().__init__("bert", *args, **kwagrs)

class RobertaDiffusionNER(DiffusionNER):

    config_class = RobertaConfig
    base_model_prefix = "roberta"
    
    def __init__(self, *args, **kwagrs):
        super().__init__("roberta", *args, **kwagrs)
    
class XLMRobertaDiffusionNER(DiffusionNER):

    config_class = XLMRobertaConfig
    base_model_prefix = "roberta"
    
    def __init__(self, *args, **kwagrs):
        super().__init__("roberta", *args, **kwagrs)

class AlbertDiffusionNER(DiffusionNER):

    config_class = AlbertConfig
    base_model_prefix = "albert"
    
    def __init__(self, *args, **kwagrs):
        super().__init__("albert", *args, **kwagrs)


_MODELS = {
    'diffusionner': BertDiffusionNER,
    'roberta_diffusionner': RobertaDiffusionNER,
    'xlmroberta_diffusionner': XLMRobertaDiffusionNER,
    'albert_diffusionner': AlbertDiffusionNER
}

def get_model(name):
    return _MODELS[name]
