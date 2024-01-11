from typing import Any

import einops
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.bert.configuration_bert import BertConfig

from src.model.blip import init_tokenizer, load_checkpoint
from src.model.med import BertModel
from src.tools.utils import print_dist
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertConfig, BertLMHeadModel
from src.model.vit import VisionTransformer, interpolate_pos_embed

class BLIPCir(nn.Module):
    def __init__(
        self,
        loss: Any,
        vit_model="eva_clip_g",
        drop_path_rate=0,
        med_config="configs/med_config.json",
        vit_precision="fp16",
        use_grad_checkpoint=False,
        freeze_vit=True,
        image_size=384,
        vit="large",
        vit_grad_ckpt=True,
        vit_ckpt_layer=12,
        embed_dim=256,
        train_vit=False,
        num_query_token=32,
        cross_attention_freq=2,
    ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()

        self.loss = loss

        self.m = nn.MaxPool1d(3, stride=2)

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, image_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            # logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )

        self.tokenizer = init_tokenizer()

        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.train_vit = train_vit
        if not self.train_vit:
            # Do not train visual encoder
            for p in self.visual_encoder.parameters():
                p.requires_grad = False

        for p in self.vision_proj.parameters():
            p.requires_grad = False

        self.temp = 0.07

    def forward(self, batch, fabric):
        ref_img, tar_feat, caption, _ = batch

        print(ref_img.shape)
        print(tar_feat.shape)


        device = ref_img.device

        if self.train_vit:
            ref_img_embs = self.visual_encoder(ref_img)
            # tar_img_embs = self.visual_encoder(tar_feat)
        else:
            with torch.no_grad():
                ref_img_embs = self.visual_encoder(ref_img)
                # tar_img_embs = self.visual_encoder(tar_feat)

        print(ref_img_embs.shape)

        image_embeds = self.ln_vision(ref_img_embs)
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            device
        )

        text_tokens = self.tokenizer(
              caption,
              padding="max_length",
              truncation=True,
              max_length=35,
              return_tensors="pt",
          ).to(device)

        # Encode tar_feat
        tar_embeds = tar_feat.float()
        tar_atts = torch.ones(tar_embeds.size()[:-1], dtype=torch.long).to(
            device
        )


        # Image Text Matching
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                ref_img.device
            )
        
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

        output_itm = self.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        query_feat = self.m(
            self.text_proj(output_itm.last_hidden_state[:, 0, :]), dim=-1
        )

        # Image Text Contrastive

        query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_feats = F.normalize(
                self.vision_proj(query_output.last_hidden_state), dim=-1
            )

        if fabric.world_size > 1:
            # d: devices, b: batch size, e: embedding dim
            query_feat = fabric.all_gather(query_feat, sync_grads=True)
            query_feat = einops.rearrange(query_feat, "d b e -> (d b) e")

            tar_img_feat = fabric.all_gather(tar_img_feat, sync_grads=True)
            tar_img_feat = einops.rearrange(tar_img_feat, "d b e -> (d b) e")

        return self.loss(query_feat, tar_img_feat, self.temp)

    def init_vision_encoder(
        self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision
    ):
        assert model_name in [
            "eva_clip_g",
            "eva2_clip_L",
            "clip_L",
        ], "vit model must be eva_clip_g, eva2_clip_L or clip_L"
        if model_name == "eva_clip_g":
            # visual_encoder = create_eva_vit_g(
            #     img_size, drop_path_rate, use_grad_checkpoint, precision
            # )
            visual_encoder, vision_width = create_vit(
                "base", img_size, False, 0
            )
#         elif model_name == "eva2_clip_L":
#             visual_encoder = create_eva2_vit_L(
#                 img_size, drop_path_rate, use_grad_checkpoint, precision
#             )
        elif model_name == "clip_L":
            visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
        ln_vision = LayerNorm(visual_encoder.num_features)
        self.vit_name = model_name
        return visual_encoder, ln_vision

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def init_Qformer(num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        encoder_config.is_decoder = True
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

def create_vit(
    vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0
):
    assert vit in ["base", "large"], "vit parameter must be base or large"
    if vit == "base":
        vision_width = 768
        visual_encoder = VisionTransformer(
            img_size=image_size,
            patch_size=16,
            embed_dim=vision_width,
            depth=12,
            num_heads=12,
            use_grad_checkpointing=use_grad_checkpointing,
            ckpt_layer=ckpt_layer,
            drop_path_rate=0 or drop_path_rate,
        )
    elif vit == "large":
        vision_width = 1024
        visual_encoder = VisionTransformer(
            img_size=image_size,
            patch_size=16,
            embed_dim=vision_width,
            depth=24,
            num_heads=16,
            use_grad_checkpointing=use_grad_checkpointing,
            ckpt_layer=ckpt_layer,
            drop_path_rate=0.1 or drop_path_rate,
        )
    else:
        raise NotImplementedError
    return visual_encoder, vision_width



def blip_cir(model, ckpt_path, **kwargs):
    if ckpt_path:
        model, msg = load_checkpoint(model, ckpt_path)
        print_dist("missing keys:")
        print_dist(msg.missing_keys)
    return model
