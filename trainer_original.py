import torch
from contextlib import contextmanager, nullcontext

import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm.notebook import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.plms import PLMSSampler
import clip
import random
import math
from tqdm import tqdm
from ldm.modules.attention import CrossAttention
import k_diffusion as K
import wandb

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default='/home/xuser/nvme1/stableckpt/v13/config.yaml',
    help="path to config which constructs model",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default='/home/xuser/nvme1/stableckpt/v13/model.ckpt',
    help="path to checkpoint of model",
)
parser.add_argument(
    "--labels",
    type=str,
    default='/home/xuser/nvme1/workspace/finetune/BLIP/safe_e62.txt',
    help="path to BLIP classification file",
)
parser.add_argument(
    "--images",
    type=str,
    default='/home/xuser/nvme1/workspace/aero/e6/scrapedirsafe/',
    help="directory of sample images",
)
parser.add_argument(
    "--prefix",
    type=str,
    default='fursona, ',
    help="the prefix for all training samples",
)
parser.add_argument(
    "--bs",
    type=int,
    default=8,
    help="batch size for training",
)
parser.add_argument(
    "--lr",
    type=float,
    default=5e-5,
    help="batch size for training",
)
#parser.add_argument(
#    "--steps",
#    type=int,
#    default=180224,
#    help="number of steps to train",
#)
parser.add_argument(
    "--save_name",
    type=str,
    default='furry',
    help="number of steps to train",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default='sdhn',
    help="number of steps to train",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="seed to use for training",
)
parser.add_argument(
    "--eval_percentage",
    type=float,
    default=0.0,
    help="percentage of dataset to use for eval",
)
parser.add_argument(
    "--eval_every",
    type=int,
    default=0,
    help="run eval every n steps",
)
parser.add_argument(
    "--gas",
    type=int,
    default=8,
    help="how many gradient accumulation steps",
)
parser.add_argument(
    "--size",
    type=int,
    default=512,
    help="what resolution to train at",
)
parser.add_argument(
    "--scale_mode",
    type=str,
    default="lanczos",
    help="what scaling mode to use for scaling images. Accepts either lanczos or nearest",
)
parser.add_argument(
    "--save_every",
    type=int,
    default=1024,
    help="save every n steps",
)
parser.add_argument(
    "--resume",
    type=str,
    default="",
    help="resume from checkpoint"
)
parser.add_argument(
    "--sched_epoch_size",
    type=int,
    default=8,
    help="resume from checkpoint"
)
parser.add_argument(
    "--sched_warmup_epochs",
    type=int,
    default=32,
    help="resume from checkpoint"
)
parser.add_argument(
    "--sched_train_epochs",
    type=int,
    default=12288,
    help="resume from checkpoint"
)
parser.add_argument(
    "--sched_rampdown_epochs",
    type=int,
    default=8192,
    help="resume from checkpoint"
)
parser.add_argument(
    "--hypernet_model",
    type=str,
    default="standard",
    help="which hypernet model to use"
)
parser.add_argument(
    "--sample_text",
    type=str,
    default="canadian prime minister eevee in a suit speaking in parliament",
    help="what text to use when sampling"
)
parser.add_argument(
    "--image_every",
    type=int,
    default=8,
    help="How often to sample images (value is multiplied by gas)"
)
parser.add_argument(
    "--dataset_mode",
    type=str,
    default="filename",
    help="which mode to use for the dataset"
)
parser.add_argument(
    "--uncond_percent",
    type=float,
    default=0.0,
    help="how much to train in unconditional mode"
)

opt = parser.parse_args()

total_steps = (opt.sched_rampdown_epochs * opt.sched_epoch_size) + (opt.sched_warmup_epochs * opt.sched_epoch_size) + (opt.sched_train_epochs * opt.sched_epoch_size)

seed_everything(opt.seed)

config = OmegaConf.load(f"{opt.config}")
config['model']['params']['unet_config']['params']['use_checkpoint'] = False
config['model']['params']['cond_stage_trainable'] = True
print(config)
#eturn
model = load_model_from_config(config, f"{opt.ckpt}")

device = torch.device("cuda")

model.train()

class HyperLogic2(torch.nn.Module):
    logic_multiplier = 1.0
    def __init__(self, dim, heads=0):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim*2)
        self.linear2 = torch.nn.Linear(dim*2, dim)

    def forward(self, _x):
        #print(_x.shape)
        return _x + (self.linear2(self.linear1(_x)) * HyperLogic2.logic_multiplier)
        
class HyperLogicCondUncond(torch.nn.Module):
    logic_multiplier = 1.0
    def __init__(self, dim, heads=0):
        super().__init__()
        self.linear1_cond = torch.nn.Linear(dim, dim*2, bias=False)
        self.linear2_cond = torch.nn.Linear(dim*2, dim, bias=False)

        self.linear1_uncond = torch.nn.Linear(dim, dim*2)
        self.linear2_uncond = torch.nn.Linear(dim*2, dim)

        #torch.nn.init.normal_(self.linear1_uncond.weight, std=0.01)
        #torch.nn.init.normal_(self.linear2_uncond.weight, std=0.01)

        #torch.nn.init.normal_(self.linear1_uncond.bias, std=0.01)
        #torch.nn.init.normal_(self.linear2_uncond.bias, std=0.01)

        #self.linear1 = torch.nn.Linear(dim, dim*2)
        #self.linear2 = torch.nn.Linear(dim*2, dim)

    def forward(self, _x):
        #print(_x.shape)

        x_uncond = _x[:_x.shape[0]//2]
        x_cond = _x[_x.shape[0]//2:]

        print("cond shapes: ")
        print(x_cond.shape)
        print(x_uncond.shape)

        x_cond = self.linear2_cond(self.linear1_cond(x_cond))
        x_uncond = self.linear2_uncond(self.linear1_uncond(x_uncond))

        x_h = torch.cat((x_uncond, x_cond), dim=0)

        print("x_h shape")
        print(x_h.shape)



        #_x2 = torch.cat((x_uncond, x_cond), dim=0)

        return _x + (x_h * HyperLogic2.logic_multiplier)
        #return _x + (self.linear2(self.linear1(_x)) * HyperLogic2.logic_multiplier)
from collections import OrderedDict
class LayerNorm(torch.nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(torch.nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = torch.nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = torch.nn.Sequential(OrderedDict([
            ("c_fc", torch.nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", torch.nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class RABLogic(torch.nn.Module):
    logic_multiplier = 0.000000001
    def __init__(self, dims, heads=6):
        super().__init__()
        self.block = ResidualAttentionBlock(dims, heads)
        self.bias = torch.nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

    def forward(self, _x):
        return _x + (self.block(_x)*self.bias)#RABLogic.logic_multiplier

class HyperLogicSkip(torch.nn.Module):
    logic_multiplier = 1.0
    cond = None
    def __init__(self, dim, heads=0):
        super().__init__()
        #self.lin_cond = torch.nn.Linear(77*768, 10)
        self.linear1 = torch.nn.Linear(dim+768+768, (dim)*2)
        self.linear2 = torch.nn.Linear((dim)*2, dim)

    def set_cond(cond):
        HyperLogicSkip.cond = cond

    def forward(self, _x):
        #c = self.lin_cond(torch.reshape(HyperLogicSkip.cond, (HyperLogicSkip.cond.shape[0], 77*768)))
        c = torch.mean(HyperLogicSkip.cond.double(),dim=1).to(_x.dtype)
        c = c.unsqueeze(1).repeat(1,_x.shape[1],1)
        c2 = HyperLogicSkip.cond[:, -1]
        c2 = c2.unsqueeze(1).repeat(1,_x.shape[1],1)
        x = torch.cat([_x, c, c2], dim=2)
        return _x + (self.linear2(self.linear1(x)) * HyperLogic2.logic_multiplier)

class ClipSkipConnection(torch.nn.Module):
    logic_multiplier = 1.0
    cond = None
    def __init__(self, dim, num_clip_models=0, exclusive=False):
        super().__init__()
        #self.lin_cond = torch.nn.Linear(77*768, 10)
        self.linear1 = torch.nn.Linear(dim+(512*num_clip_models), (dim)*2)
        self.linear2 = torch.nn.Linear((dim)*2, dim)
        self.padx = torch.nn.ConstantPad3d((0, (512*num_clip_models), 0, 0, 0, 0), 0)
        self.pady = torch.nn.ConstantPad3d((dim, 0, 0, 0, 0, 0), 0)
        self.exclusive=exclusive

    def set_cond(cond):
        HyperLogicSkip.cond = cond

    def forward(self, _x):
        #c = self.lin_cond(torch.reshape(HyperLogicSkip.cond, (HyperLogicSkip.cond.shape[0], 77*768)))
        y = torch.cat(clip_tensors, dim=-1)
        x = self.padx(_x) + self.pady(y.repeat(1,_x.shape[1],1))
        if self.exclusive:
            return self.linear2(self.linear1(x))
        else:
            return _x + (self.linear2(self.linear1(x)) * HyperLogic2.logic_multiplier)


if opt.hypernet_model == "skip2t5":
    print("Loading t5...")
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-large").cuda()
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-large")
t5_tokens = None
t5_prior = None
clip_tensors = []

def encode_t5(text_batch):
    global t5_tokens
    enc = t5_tokenizer(text_batch, return_tensors="pt", padding=True)
    with torch.no_grad():
        t5_tokens = t5_model.encoder(input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda(),return_dict=True)
    t5_tokens = t5_prior(t5_tokens.last_hidden_state)

class T5Prior(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = torch.nn.GRU(1024, 1024, batch_first=True)
    def forward(self, x):
        _, x = self.gru(x)
        x = x.permute(1,0,2) # D,batch,H -> batch,D,H
        return x

class ClipPrior(torch.nn.Module):
    def __init__(self, clip_model_name, dim):
        super().__init__()
        self.gru = torch.nn.GRU(dim, 512, batch_first=True)
        self.model_name = clip_model_name

    def forward(self, x):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                clip_vector = clip_model_map[self.model_name][0].encode_text(clip.tokenize(x, truncate=True).cuda()).float()
        _, x = self.gru(clip_vector)
        x = x.squeeze(0)#x.permute(1,0,2).squeeze(1) # D,batch,H -> batch,D,H
        return x

class DummyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

class T5SkipConnection(torch.nn.Module):
    def __init__(self, dim, heads=0):
        super().__init__()
        self.linear1 = torch.nn.Linear(1024,512)
        self.linear2 = torch.nn.Linear(dim+512, (dim)*2)
        self.linear3 = torch.nn.Linear((dim)*2, dim)
        self.dim = dim
        
        self.padx = torch.nn.ConstantPad3d((0, 512, 0, 0, 0, 0), 0)
        self.pady = torch.nn.ConstantPad3d((dim, 0, 0, 0, 0, 0), 0)

    def forward(self, _x):
        y = self.linear1(t5_tokens)
        #y = y.unsqueeze(1).repeat(1,_x.shape[1],1)

        # OPTIMIZE THIS!!!!!!!
        #x = self.pad(_x)

        x = self.padx(_x) + self.pady(y.repeat(1,_x.shape[1],1))
        #for e in range(x.shape[1]):
        #    #print(x[:, e].shape)
        #    ##print(x[:, e][:, self.dim:].shape)
        #    #print(y.shape)
        #    ##print("\n")
        #    x[:, e][:, self.dim:] = y

        #x = torch.cat([_x, y], dim=2)
        return _x + (self.linear3(self.linear2(x)) * HyperLogic2.logic_multiplier)


class HyperLogicSkip2(torch.nn.Module):
    logic_multiplier = 1.0
    cond = None
    def __init__(self, dim, heads=0):
        super().__init__()
        #self.lin_cond = torch.nn.Linear(77*768, 10)
        self.linear1 = torch.nn.Linear(dim+768, (dim)*2)
        self.linear2 = torch.nn.Linear((dim)*2, dim)

    def set_cond(cond):
        HyperLogicSkip.cond = cond

    def forward(self, _x):
        #c = self.lin_cond(torch.reshape(HyperLogicSkip.cond, (HyperLogicSkip.cond.shape[0], 77*768)))
        c = torch.mean(HyperLogicSkip.cond.double(),dim=1).to(_x.dtype)
        #c = c.unsqueeze(1).repeat(1,_x.shape[1],1)
        #c2 = HyperLogicSkip.cond[:, -1]
        #c2 = c2.unsqueeze(1).repeat(1,_x.shape[1],1)
        x = torch.cat([_x, c], dim=2)
        return _x + (self.linear2(self.linear1(x)) * HyperLogic2.logic_multiplier)

image_cache_map = {}
wandb.init(entity='novelaix', project=opt.wandb_project, name=('%dbs-%dgas-%flr-%dsteps-%dres-'%(opt.bs, opt.gas, opt.lr, total_steps, opt.size)) + opt.save_name + "-" + str(time.time()))

# this is applied to all attentions in SD
# channels: (Q_hypernet, K_hypernet)
if opt.hypernet_model == "conduncond":
    network = {
        768: (RABLogic(768, heads=3).cuda(), RABLogic(768, heads=3).cuda()),
        1280: (RABLogic(1280, heads=2).cuda(), RABLogic(1280, heads=2).cuda()),
        640: (RABLogic(640, heads=4).cuda(), RABLogic(640, heads=4).cuda()),
        320: (RABLogic(320, heads=4).cuda(), RABLogic(320, heads=4).cuda()),
    }
elif opt.hypernet_model == "clip_multiperceptor_prior":
    clip_models = ["ViT-B/16", "ViT-B/32", "RN50x64", "RN50x16", "RN50x4", "RN101", "RN50"]

    network = {
        #768: (ClipSkipConnection(768, len(clip_models), exclusive=True).cuda(), ClipSkipConnection(768, len(clip_models), exclusive=True).cuda()),
        768: (ClipSkipConnection(768, len(clip_models)).cuda(), ClipSkipConnection(768, len(clip_models)).cuda()),
        1280: (ClipSkipConnection(1280, len(clip_models)).cuda(), ClipSkipConnection(1280, len(clip_models)).cuda()),
        640: (ClipSkipConnection(640, len(clip_models)).cuda(), ClipSkipConnection(640, len(clip_models)).cuda()),
        320: (ClipSkipConnection(320, len(clip_models)).cuda(), ClipSkipConnection(320, len(clip_models)).cuda()),
        #"clip_prior": (T5Prior().cuda(), DummyLayer().cuda())
    }
    clip_model_map = {}

    for cm in clip_models:
        print("Loading " + cm)
        device = "cuda"
        model_clip, preprocess_clip = clip.load(cm, device=device)
        clip_model_map[cm] = (model_clip, preprocess_clip)

        # get feature shape
        text_features = model_clip.encode_text(clip.tokenize(opt.sample_text, truncate=True).cuda())

        network["clip_prior_" + cm] = (ClipPrior(cm, text_features.shape[1]).cuda(), DummyLayer().cuda())

    def encode_clip(batch):
        clip_tensors.clear()
        for cm in clip_models:
            clip_tensors.append(network["clip_prior_" + cm][0](batch))

elif opt.hypernet_model == "skip2t5":
    network = {
        768: (HyperLogic2(768).cuda(), HyperLogic2(768).cuda()),
        1280: (T5SkipConnection(1280).cuda(), T5SkipConnection(1280).cuda()),
        640: (T5SkipConnection(640).cuda(), T5SkipConnection(640).cuda()),
        320: (T5SkipConnection(320).cuda(), T5SkipConnection(320).cuda()),
        "t5_prior": (T5Prior().cuda(), DummyLayer().cuda())
    }
    t5_prior = network["t5_prior"][0]
elif opt.hypernet_model == "skip":
    network = {
        768: (HyperLogic2(768).cuda(), HyperLogic2(768).cuda()),
        1280: (HyperLogicSkip(1280).cuda(), HyperLogicSkip(1280).cuda()),
        640: (HyperLogicSkip(640).cuda(), HyperLogicSkip(640).cuda()),
        320: (HyperLogicSkip(320).cuda(), HyperLogicSkip(320).cuda()),
    }
    def cond_hook(c):
        HyperLogicSkip.set_cond(c)
        return c
    LatentDiffusion.set_hook_cond(cond_hook)
else:
    network = {
        768: (HyperLogic2(768).cuda(), HyperLogic2(768).cuda()),
        1280: (HyperLogic2(1280).cuda(), HyperLogic2(1280).cuda()),
        640: (HyperLogic2(640).cuda(), HyperLogic2(640).cuda()),
        320: (HyperLogic2(320).cuda(), HyperLogic2(320).cuda()),
    }

if opt.hypernet_model == "nocond":
    del network[768]

if len(opt.resume) > 0:
    hypernet = torch.load(opt.resume)
    for k in hypernet.keys():
        network[k][0].load_state_dict(hypernet[k][0])
        network[k][1].load_state_dict(hypernet[k][1])

CrossAttention.set_noise_cond(False)
CrossAttention.set_hypernetwork(network)

layers_to_train = network.values()

for layer in layers_to_train:
    for sublayer in layer:
        for param in sublayer.parameters():
            param.requires_grad = True

params_to_train = []
for layer in layers_to_train:
    for sublayer in layer:
        params_to_train = params_to_train + list(sublayer.parameters())
        sublayer.train()

def rampupdown(x):
    if x < opt.sched_warmup_epochs:
        x = float(x+1) / float(opt.sched_warmup_epochs+1)
        x = (math.sin((x - 0.5) * math.pi) / 2) + 0.5
    elif x < opt.sched_warmup_epochs + opt.sched_train_epochs:
        x = 1.0
    else:
        x = ((x-(opt.sched_warmup_epochs + opt.sched_train_epochs))/opt.sched_rampdown_epochs)
        x = (math.sin((x + 0.5) * math.pi) / 2) + 0.5

    return x
#rampupdown = lambda x: float(x+1) / float(opt.sched_warmup_epochs+1) if x < opt.sched_warmup_epochs else (1.0 if x < opt.sched_warmup_epochs + opt.sched_train_epochs else 1-((x-(opt.sched_warmup_epochs + opt.sched_train_epochs))/opt.sched_rampdown_epochs))

optim = torch.optim.AdamW(params_to_train, lr=opt.lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=rampupdown)

ids = open(opt.labels,"r").read().split("\n")
ids = [ id.split("\t") for id in ids ]

ids_eval = []

db_random = random.Random(opt.seed)

if opt.eval_percentage > 0:
    db_random.shuffle(ids)
    cut_id = int(opt.eval_percentage * len(ids))
    ids_eval = ids[:cut_id]
    ids = ids[cut_id:]

id_db = {"train": ids, "eval": ids_eval}

import pickle
e621_tags = pickle.load(open("datasets/post_tagmap_new_pandas.pkl", "rb"))
e621_tag_types = pickle.load(open("datasets/e621_tag_types.pkl", "rb"))
invalidations = pickle.load(open("datasets/invalidations.pkl", "rb"))

#invalidations = [mammal] = {canid, canis, fox, dog}, [canid] = {fox, dog, etc}

def invalidates(species1, species2):
    return (species1 in invalidations) and (species2 in invalidations[species1])
    #id,antecedent_name,consequent_name,created_at,status

def find_highest_species(species):
    new_species = []

    for s in species:
        bad = False
        for s2 in species:
            if invalidates(s, s2):
                bad = True
                break
        if not bad:
            new_species.append(s)

    return new_species

def get_special_prefix_v2(file, blip_description):
    tags = e621_tags[file.split("/")[-1].split(".")[-2]].split(" ")
    artists = []
    species = []
    characters = []
    for tag in tags:
        tag_type = e621_tag_types[tag]
        if "_(" in tag:
            tag = tag.split("_(")[0]
        
        #if tag_type == '5':
        #    tag = tag #+ " species"
        if tag_type == '4':
            characters.append(tag)
        elif tag_type == '1' and tag != "conditional_dnp" and tag != "unknown_artist":
            artists.append(tag)
        elif tag_type == '5':
            species.append(tag)
    
    characters += find_highest_species(species)
    db_random.shuffle(characters)
    db_random.shuffle(artists)
    s = ((" ".join(characters)) + ", by " + (" and ".join(artists)) + ", " + blip_description).replace("_", " ")
    print(s)
    return s


def get_special_prefix(file):
    tags = e621_tags[file.split("/")[-1].split(".")[-2]].split(" ")
    random.shuffle(tags)
    #print(tags)
    """
    <option value="0">General</option>
    <option selected="selected" value="5">Species</option>
    <option value="4">Character</option>
    <option value="3">Copyright</option>
    <option value="1">Artist</option>
    <option value="6">Invalid</option>
    <option value="8">Lore</option>
    <option value="7">Meta</option>
    """
    ret_str = ""

    for tag in tags:
        tag_type = e621_tag_types[tag]
        if "_(" in tag:
            tag = tag.split("_(")[0]
        
        #if tag_type == '5':
        #    tag = tag #+ " species"
        if tag_type == '4':
            tag = "character named " + tag
        elif tag_type == '1':
            tag = "art by " + tag
        if not (tag.startswith("generation_") and tag.endswith("_pokemon")) and tag != "art by conditional_dnp" and tag != "art by unknown_artist":
            tag = tag.replace("_", " ")
            if tag_type in ['5', '1', '4']:
                ret_str = ret_str + tag + ", "

    if len(ret_str) > 2:
        ret_str = ret_str[:-2]
        print(ret_str)

    return ret_str

def next_batch(size=512, db_key="train", in_idx=-1):
    images = []
    masks = []
    labels = []
    bs = int(opt.bs)
    ids = id_db[db_key]

    scale_mode = Image.LANCZOS if opt.scale_mode == 'lanczos' else Image.NEAREST

    while(len(images)<bs):
        try:
            if in_idx == -1:
                file, label = db_random.choice(ids)
            else:
                file, label = ids[in_idx]
                in_idx = in_idx + 1
                if in_idx >= len(ids):
                    in_idx = 0
            #labels.append(opt.prefix + get_special_prefix(file) + label)
            #labels.append(get_special_prefix_v2(file, label))
            #labels.append(opt.prefix + label)
            labels.append(label)
            
            if file in image_cache_map:
                image = image_cache_map[file]
            else:
                if True: #MTG mode
                    if opt.dataset_mode == "filename":
                        image = Image.open(opt.images + file)
                    else: # full path
                        image = Image.open(file)
                    image = image.resize((384, 536), scale_mode)
                    image = np.array(image).astype(np.float32) / 127.5 - 1
                    if(len(image.shape) == 2):
                        image = np.stack([image]*3, -1)
                    image = image[:, :, :3]
                    rhc = int(random.random() * (640 - 536))
                    rhc = rhc // 8
                    rhc = rhc * 8
                    img2 = np.random.randn(640, 384, 3)
                    img2[rhc:rhc+536, :, :] = image
                    image = img2

                    mask = np.zeros((640//8, 384//8, 1), dtype=np.float32)
                    mask[rhc//8:rhc//8+536//8, :, :] = np.ones((536//8, 384//8, 1), dtype=np.float32)

                    masks.append(mask)

                    #384 534   (536)
                else:
                    if opt.dataset_mode == "filename":
                        image = Image.open(opt.images + file)
                    else: # full path
                        image = Image.open(file)
                    ih,iw = image.size

                    if iw == ih:
                        image = image.resize((size, size), scale_mode)
                    elif iw > ih:
                        crophalf = int((iw - ih) * db_random.random())
                        image = image.crop((crophalf, 0, crophalf+ih, ih))
                        image = image.resize((size, size), scale_mode)
                    else:
                        crophalf = int((ih - iw) * db_random.random())
                        image = image.crop((0, crophalf, iw, crophalf+iw))
                        image = image.resize((size, size), scale_mode)

                    image = np.array(image).astype(np.float32) / 127.5 - 1
                    if(len(image.shape) == 2):
                        image = np.stack([image]*3, -1)
                    image = image[:, :, :3]

            images.append(image)
        except:
            print("Error")
            import traceback
            traceback.print_exc()
            import time
            time.sleep(0.1)
    images = torch.tensor(np.array(images)).cuda()
    masks = torch.tensor(np.array(masks)).cuda()

    if opt.hypernet_model == "conduncond":
        labels = labels[bs//2:]
        labels = (["", ] * (bs//2)) + labels
        print("Conduncond labels shape : " + str(len(labels)))
        #labels = labels[:bs//2] + (["", ] * (bs//2))
        #labels = (["", ] * (bs//2)) + 

    return {"jpg": images, "txt": labels, "in_idx": in_idx, "mask": masks}

read_queue = []
import time
import threading
def read_batches():
    while True:
        while len(read_queue) > 10:
            time.sleep(1)
        for e in range(10):
            try:
                read_queue.append(next_batch(size=opt.size))
            except:
                print("err")

x = threading.Thread(target=read_batches, args=(), daemon=True)
x.start()

def run_eval():
    in_idx = 0
    eval_total = 0.0
    eval_total_batches = 0
    progress = tqdm(range(int(1+(len(id_db['eval'])/opt.bs))))
    seed_everything(opt.seed)
    for e in progress:
        batch = next_batch(db_key="eval", in_idx=in_idx, size=opt.size)

        if batch['in_idx'] < in_idx:
            break
        else:
            in_idx = batch['in_idx']
        with torch.no_grad():
            eval_total += float(model.training_step(batch, e, log=False, mask=batch["mask"] if "mask" in batch else None))
        eval_total_batches += 1
        progress.set_description("eval: %.5f" % (eval_total / float(eval_total_batches)))
    return eval_total / float(eval_total_batches)

progress = tqdm(range(total_steps))

print("Dataset info:\nTrain image count: %d\nEval image count: %d\nEpochs: %f" % (len(id_db['train']), len(id_db['eval']), float(total_steps) / float(len(id_db['train']))))

if opt.eval_every > 0:
    CrossAttention.set_hypernetwork(None)
    print("\nEval vanilla: %f\n" % run_eval())
    CrossAttention.set_hypernetwork(network)
    eval_val = run_eval()
    print("\nEval @0: %f\n" % eval_val)
else:
    eval_val = 0

wandb.log({"loss": float(eval_val), "eval": float(eval_val)})

optim.zero_grad()
seed_everything(opt.seed)

def get_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return float(total_norm)

with model.ema_scope():

    def sample_images(text, arch):
        with torch.no_grad():
            # edit settings
            opt = argparse.Namespace(C=4, H=512, W=512, ckpt='/home/xuser/nvme1/stableckpt/v13/model.ckpt', config='/home/xuser/nvme1/stableckpt/v13/config.yaml', ddim_eta=0.0, ddim_steps=50, dyn=None, f=8, fixed_code=False, from_file=None, manual=True, n_iter=1, n_rows=0, n_samples=16, outdir='../aerogentest5', plms=True, precision='autocast', prompt='Toddlers playing in lava', scale=11.0, seed=199266, skip_grid=False, skip_save=False, strength=0.7)

            opt.plms = True
            if True: # mtg mode
                opt.W = 384
                opt.H = 640
            else:
                opt.W = 512#384
                opt.H = 512#640
            opt.prompt = text
            opt.scale = 11
            opt.seed = 3128017969+19+9+9+9+9+9+9+9+9+9+9+9+9+2+9+9+9+9+9+9+9+9+9+9+9+9+9+9+9+9+9+9#+1754
            opt.n_samples = 9
            opt.strength = 0.54
            opt.ddim_steps = 50
            print(config)

            seed_everything(opt.seed)

            if arch == "skip2t5":
                encode_t5([opt.prompt, ] * (opt.n_samples * 2))
            if arch == "clip_multiperceptor_prior":
                encode_clip([opt.prompt, ] * (opt.n_samples * 2))
        
            batch_size = opt.n_samples
            n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
            if not opt.from_file:
                prompt = opt.prompt
                assert prompt is not None
                data = [batch_size * [prompt]]
            else:
                print(f"reading prompts from {opt.from_file}")
                with open(opt.from_file, "r") as f:
                    data = f.read().splitlines()
                    data = list(chunk(data, batch_size))

            batch_size_late = 1
            data_late = [batch_size_late * [prompt]]


            import torch.nn.functional as F
            if opt.plms:
                sampler = PLMSSampler(model)
            else:
                sampler = DDIMSampler(model)

            def clean_dict(d):
                ret_dict = {}
                for k in d.keys():
                    if "loss." not in k:
                        ret_dict[k] = d[k]
                return ret_dict

            def make_seed_noise(seed, res=1):
                seed_everything(seed)
                return torch.randn([opt.C, (opt.H*res) // opt.f, (opt.W*res) // opt.f], device=device)

            start_code_high = torch.stack([make_seed_noise(opt.seed + e, res=2) for e in range(opt.n_samples)], dim=0)   #torch.randn([opt.n_samples, opt.C, (opt.H * 2) // opt.f, (opt.W*2) // opt.f], device=device)

            start_code = F.interpolate(start_code_high, (start_code_high.shape[2]//2, start_code_high.shape[3]//2), mode='nearest')

            def fix_batch(tensor, bs):
                #tensor[0] = network[0][0].learned_embedding
                return torch.stack([tensor.squeeze(0)]*bs, dim=0)

            # mix conditioning vectors for prompts
            def prompt_mixing(model, prompt_body, batch_size):
                if False:
                    print(fix_batch(model.get_learned_conditioning([prompt_body]), batch_size).shape)
                    print(fix_batch(clip_model.encode_text(clip.tokenize(["a diagram"]).to(device)), batch_size).shape)
                    print( fix_batch(torch.stack([clip_model.encode_image(preprocess(Image.open("International-Clown-Week-640x514.jpg")).unsqueeze(0).to(device))]*77, dim=0), batch_size).squeeze(2).shape)
                    return fix_batch(torch.stack([clip_model.encode_image(preprocess(Image.open("International-Clown-Week-640x514.jpg")).unsqueeze(0).to(device))]*77, dim=0), batch_size).squeeze(2)
                if "|" in prompt_body:
                    prompt_parts = prompt_body.split("|")
                    prompt_total_power = 0
                    prompt_sum = None
                    for prompt_part in prompt_parts:
                        prompt_power = 1
                        if ":" in prompt_part:
                            prompt_sub_parts = prompt_part.split(":")
                            try:
                                prompt_power = float(prompt_sub_parts[1])
                                prompt_part = prompt_sub_parts[0]
                            except:
                                print("Error parsing prompt power! Assuming 1")
                        prompt_vector = model.get_learned_conditioning([prompt_part])
                        if prompt_sum is None:
                            prompt_sum = prompt_vector * prompt_power
                        else:
                            prompt_sum = prompt_sum + (prompt_vector * prompt_power)
                        prompt_total_power = prompt_total_power + prompt_power
                    return fix_batch(prompt_sum / prompt_total_power, batch_size)
                else:
                    return fix_batch(model.get_learned_conditioning([prompt_body]), batch_size)

            k_model = K.external.CompVisDenoiser(model)
            class StableInterface(torch.nn.Module):
                def __init__(self, model, thresholder = None):
                    super().__init__()
                    self.inner_model = model
                    self.sigma_to_t = model.sigma_to_t
                    self.thresholder = thresholder
                    self.get_sigmas = model.get_sigmas

                @torch.no_grad()
                def forward(self, x, sigma, uncond, cond, cond_scale):
                    x_two = torch.cat([x] * 2)
                    sigma_two = torch.cat([sigma] * 2)
                    cond_full = torch.cat([uncond, cond])
                    uncond, cond = self.inner_model(x_two, sigma_two, cond=cond_full).chunk(2)
                    x_0 = uncond + (cond - uncond) * cond_scale
                    if self.thresholder is not None:
                        x_0 = self.thresholder(x_0)

                    return x_0
            k_model = StableInterface(k_model)

            precision_scope = autocast if opt.precision=="autocast" else nullcontext
            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        tic = time.time()
                        all_samples = list()
                        for n in trange(opt.n_iter, desc="Sampling"):
                            for prompts in tqdm(data, desc="data"):
                                uc = None
                                if opt.scale != 1.0:
                                    uc = model.get_learned_conditioning(batch_size * [""])
                                if isinstance(prompts, tuple):
                                    prompts = list(prompts)
                                c = prompt_mixing(model, prompts[0], batch_size)#(model.get_learned_conditioning(prompts) + model.get_learned_conditioning(["taken at night"])) / 2
                                HyperLogicSkip.set_cond(c)
                                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                                if False: # ddim
                                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                                        conditioning=c,
                                                                        batch_size=opt.n_samples,
                                                                        shape=shape,
                                                                        verbose=False,
                                                                        unconditional_guidance_scale=opt.scale,
                                                                        unconditional_conditioning=uc,
                                                                        eta=opt.ddim_eta,
                                                                        dynamic_threshold=opt.dyn,
                                                                        x_T=start_code)

                                else: # k euler ancestral
                                    sigmas = k_model.get_sigmas(opt.ddim_steps)

                                    start_code = start_code * sigmas[0]

                                    extra_args = {'cond': c, 'uncond': uc, 'cond_scale': opt.scale}
                                    samples_ddim = K.sampling.sample_euler_ancestral(k_model, start_code, sigmas, extra_args=extra_args)


                                x_samples_ddim = model.decode_first_stage(samples_ddim)
                                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                                #if not opt.skip_save:
                                #    for x_sample in x_samples_ddim:
                                #        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                #        Image.fromarray(x_sample.astype(np.uint8)).save(
                                #            os.path.join(sample_path, f"{base_count:05}.png"))
                                #        base_count += 1
                                all_samples.append(x_samples_ddim)
            all_samples_base = all_samples
            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=3)
            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            return Image.fromarray(grid.astype(np.uint8))




    for e in progress:
        while len(read_queue) < 1:
            time.sleep(0.01)

        batch = read_queue.pop()

        if opt.uncond_percent > 0 and random.random() < opt.uncond_percent:
            # unconditional pass
            batch["txt"] = ["",] * opt.bs
        if opt.hypernet_model == "conduncond":
            RABLogic.logic_multiplier = (float(e) / 1024.0) if e < 1024 else 1.0#HyperLogic2.logic_multiplier = (float(e) / 1024.0) if e < 1024 else 1.0
        if opt.hypernet_model == "skip2t5":
            encode_t5(batch["txt"])
        if opt.hypernet_model == "clip_multiperceptor_prior":
            encode_clip(batch["txt"])

        loss = model.training_step(batch, e, log=False, mask=batch["mask"] if "mask" in batch else None)

        loss.backward()
        
        progress.set_description("loss: %.5f" % (float(loss)))
        if (e % opt.gas == opt.gas - 1) or e == 0:
            g_norm_320_k, g_norm_320_q = get_grad_norm(network[320][0]), get_grad_norm(network[320][1])
            g_norm_640_k, g_norm_640_q = get_grad_norm(network[640][0]), get_grad_norm(network[640][1])
            g_norm_1280_k, g_norm_1280_q = get_grad_norm(network[1280][0]), get_grad_norm(network[1280][1])

            ld = {
                "loss": float(loss),
                "gradnorm_320_k": g_norm_320_k, "gradnorm_320_q": g_norm_320_q,
                "gradnorm_640_k": g_norm_640_k, "gradnorm_640_q": g_norm_640_q,
                "gradnorm_1280_k": g_norm_1280_k, "gradnorm_1280_q": g_norm_1280_q,
                "lr": scheduler.get_last_lr()[0],
            }

            if opt.hypernet_model == "skip2t5":
                g_t5_prior = get_grad_norm(network["t5_prior"][0])
                ld["gradnorm_t5_prior"] = g_t5_prior

            if opt.hypernet_model != "nocond":
                g_norm_768_k, g_norm_768_q = get_grad_norm(network[768][0]), get_grad_norm(network[768][1])
                ld["gradnorm_768_k"] = g_norm_768_k
                ld["gradnorm_768_q"] = g_norm_768_q

            if ((e // opt.gas) % opt.image_every == opt.image_every-1) or e == 0:
                img = wandb.Image(sample_images(opt.sample_text, opt.hypernet_model), caption=opt.sample_text)
                ld["samples"] = img
                seed_everything(opt.seed + e)

            wandb.log(ld)

            if e != 0:
                optim.step()
                optim.zero_grad()
        
        if e % opt.sched_epoch_size == (opt.sched_epoch_size-1):
            scheduler.step()

        if opt.eval_every > 0 and (e % opt.eval_every) == (opt.eval_every - 1):
            eval_val = run_eval()
            print("\nEval @%d: %f\n" % (e+1, eval_val))
            wandb.log({"loss": float(loss), "eval": float(eval_val)})
            seed_everything(opt.seed + e)
        elif not (e % opt.gas == opt.gas - 1):
            wandb.log({"loss": float(loss)})

        if (e+1) % opt.save_every == opt.save_every - 1:
            full_state_dict = {}
            for k in network.keys():
                full_state_dict[k] = (network[k][0].state_dict(), network[k][1].state_dict())

            torch.save(full_state_dict, opt.save_name + ("-%e.pt"%e))

if opt.eval_every > 0:
    print("\nFinal eval: %f\n" % run_eval())

full_state_dict = {}
for k in network.keys():
    full_state_dict[k] = (network[k][0].state_dict(), network[k][1].state_dict())

torch.save(full_state_dict, opt.save_name + ".pt")
