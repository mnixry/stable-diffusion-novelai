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
import pickle

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
model = load_model_from_config(config, f"{opt.ckpt}")

device = torch.device("cuda")
model.train()
        
class DummyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

image_cache_map = {}
wandb.init(entity='novelaix', project=opt.wandb_project, name=('%dbs-%dgas-%flr-%dsteps-%dres-'%(opt.bs, opt.gas, opt.lr, total_steps, opt.size)) + opt.save_name + "-" + str(time.time()))

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
                    
            labels.append(label)
            
            if file in image_cache_map:
                image = image_cache_map[file]

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

def sample_images(model_path, module_path, module_name):
    return

with model.ema_scope():
    for e in progress:
        if opt.uncond_percent > 0 and random.random() < opt.uncond_percent:
            # unconditional pass
            batch["txt"] = ["",] * opt.bs

        loss = model.training_step(batch, e, log=False, mask=batch["mask"] if "mask" in batch else None)
        loss.backward()
        
        progress.set_description("loss: %.5f" % (float(loss)))
        #LOG HERE
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
