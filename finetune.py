import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.optim as optim
from pathlib import Path
from torch.utils import data
from basedformer import optimizer, utils, lm_utils, dataset
import yaml
import sys
from tqdm import tqdm
import time
import wandb
import numpy as np
import os
import torch.distributed as dist
from dotmap import DotMap
from inference import *
import random
import pickle

def clean_dict(d):
    ret_dict = {}
    for k in d.keys():
        if "loss." not in k:
            ret_dict[k] = d[k]
    return ret_dict

def setup(rank, world_size):
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(backend="nccl")
    if dist.is_initialized():
        print("Initialized process group")
    else:
        print("Failed to initialize process group")

def cleanup():
    dist.destroy_process_group()

def get_rank():
    if dist.is_initialized():
        return dist.get_rank()

def get_world():
    if dist.is_initialized():
        return dist.get_world_size()

def process_tags(tags, min_tags=1, max_tags=32, type_dropout = 0.75, keep_jpeg_artifacts=True, sort_tags=True):
    if isinstance(tags, str):
        tags = tags.split(" ")
    final_tags = {}

    tag_dict = {tag: True for tag in tags}
    for bad_tag in ["absurdres", "highres", "translation_request", "translated", "commentary", "commentary_request", "commentary_typo", "character_request", "bad_id", "bad_link", "bad_pixiv_id", "bad_twitter_id", "bad_tumblr_id", "bad_deviantart_id", "bad_nicoseiga_id", "md5_mismatch", "cosplay_request", "artist_request", "wide_image", "author_request"]:
        if bad_tag in tag_dict:
            del tag_dict[bad_tag]

    if "rating:questionable" in tag_dict or "rating:explicit" in tag_dict:
        final_tags["nsfw"] = True

    tag_count = min(random.randint(min_tags, max_tags), len(tag_dict.keys()))
    chosen_tags = random.sample(list(tag_dict.keys()), tag_count)
    if sort_tags:
        chosen_tags = sorted(chosen_tags)
    for tag in chosen_tags:
        tag = tag.replace(",", "").replace("_", " ")
        if random.random() < type_dropout:
            if tag.startswith("artist:"):
                tag = tag[7:]
            elif tag.startswith("copyright:"):
                tag = tag[10:]
            elif tag.startswith("character:"):
                tag = tag[10:]
            elif tag.startswith("general:"):
                tag = tag[8:]
        if tag.startswith("meta:"):
            tag = tag[5:]
        final_tags[tag] = True

    skip_image = False
    for bad_tag in ["comic", "panels", "everyone", "sample_watermark", "text_focus", "tagme"]:
        if bad_tag in tag_dict:
            skip_image = True
    if not keep_jpeg_artifacts and "jpeg_artifacts" in tag_dict:
        skip_image = True

    return "Tags: " + ", ".join(list(final_tags.keys()))

def fsdp_train(args, model, train_loader, opt):
    bs = args["bs"]
    gas = args["gas"]
    global_rank = get_rank()
    rank = int(os.environ["LOCAL_RANK"])
    world_size = get_world()
    model.train()
    ddp_loss = torch.zeros(1).cuda()
    if rank == 0:
        t = tqdm(train_loader)
    else:
        t = train_loader

    with open("/mnt/storageserver/workspace/kuru/sdfinetune/dataset/db_tags.pkl", "rb") as f:
        db_tags = pickle.load(f)

    scaler = torch.cuda.amp.GradScaler()
    counter = 0
    for images, ids in t:
        timex = time.perf_counter()
        tags_batch = []
        for id in ids:
            tags = db_tags[id.item()]
            tags = process_tags(tags, min_tags=1, max_tags=16)
            tags_batch.append(tags)

        #copy
        tags_batch_original = tags_batch.copy()
        ucg = 0.1
        if ucg:
            ucg = int(len(tags_batch) * ucg)
            numbers = random.sample(range(len(tags_batch)), ucg)
            for i in numbers:
                tags_batch[i] = ""

        loss = 0
        batch = {}
        images = images.to(rank)

        batch["jpg"] = images
        batch["txt"] = tags_batch
        for x in range(args["gas"]):
            with torch.cuda.amp.autocast(enabled=args["amp"], dtype=args["cast_to"]):
                gas_loss = model.model.training_step({'jpg': batch["jpg"][x*bs:(x+1)*bs, ...], 'txt': batch['txt'][x*bs:(x+1)*bs]}, None, log=False)

            if args["loss_scale"]:
                with opt.optimizer.no_sync():
                    scaler.scale(gas_loss).backward()
            else:
                with opt.optimizer.no_sync():
                    gas_loss.backward()

            loss += gas_loss.item()

        loss = loss / gas
        opt.optimizer.grad_sync()
        if args["loss_scale"]:
            scaler.unscale_(opt.optimizer)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if args["loss_scale"]:
            opt.step(scaler=scaler)
        else:
            opt.step()

        if args["loss_scale"]:
            scaler.update()

        opt.zero_grad()
        #model.zero_grad(set_to_none=True)
        sec_per_step = (time.perf_counter() - timex)
        step_per_sec = (1. / sec_per_step)
        batch_size = bs * gas * world_size
        ddp_loss[0] = loss
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        if global_rank == 0:
            wandb.log({
                "train_loss": ddp_loss[0] / world_size,
                "train/sec_per_step": sec_per_step,
                "train/step_per_sec": step_per_sec, 
                "train/lr": opt.curr_lr,
                "train/batch_size": batch_size,
                "train/loss_scale": scaler.get_scale(),
                })
        
        if counter % args["save_every"] == 0:
            if global_rank == 0:
                torch.save(model.model.state_dict(), f"{args['save_path']}/{args['run_name']}-{counter}.pt")

        if counter % args["sample_every"] == 0:
            if global_rank == 0:
                torch_random_state = torch.get_rng_state()
                numpy_random_state = np.random.get_state()
                python_random_state = random.getstate()

                req = model.get_default_config
                req.n_samples = 4
                req.steps = 35
                req.prompt = "Anime girl looking at the sky"
                req.seed = 42
                images = model.sample(req)
                imagesx = [Image.fromarray(image) for image in images]
                imagesx = [wandb.Image(image, caption=req.prompt) for image in imagesx]

                req = model.get_default_config
                req.n_samples = 4
                req.steps = 35
                req.prompt = "Tags: 1girl, long hair, smile, blue eyes, open mouth, brown hair"
                req.seed = 42
                images = model.sample(req)
                imagesy = [Image.fromarray(image) for image in images]
                imagesy = [wandb.Image(image, caption=req.prompt) for image in imagesy]

                req = model.get_default_config
                req.n_samples = 4
                req.steps = 35
                req.prompt = "Tags: red bikini"
                req.seed = 42
                images = model.sample(req)
                imagesz = [Image.fromarray(image) for image in images]
                imagesz = [wandb.Image(image, caption=req.prompt) for image in imagesz]

                req = model.get_default_config
                req.n_samples = 4
                req.steps = 35
                req.prompt = "Tags: red bikini, hatsune miku"
                req.seed = 42
                images = model.sample(req)
                imageso = [Image.fromarray(image) for image in images]
                imageso = [wandb.Image(image, caption=req.prompt) for image in imageso]

                imageslog = imagesx + imagesy + imagesz + imageso
                
                req = model.get_default_config
                req.n_samples = 4
                req.steps = 35
                req.prompt = tags_batch_original[0]
                req.seed = 42
                images = model.sample(req)
                imagesrandom = [Image.fromarray(image) for image in images]
                imagesrandom = [wandb.Image(image, caption=req.prompt) for image in imagesrandom]
                wandb.log({"examples": imageslog, "examples_random": imagesrandom})
                
                torch.set_rng_state(torch_random_state)
                np.random.set_state(numpy_random_state)
                random.setstate(python_random_state)

            dist.barrier()

        counter += 1

def inner_transform(data):
    data = dataset.CPUTransforms.scale(data, 512)
    data = dataset.CPUTransforms.randomcrop(data, 512)
    return data

# we need 250 batch size to train the small GPT.
def main(rank, global_rank, world_size, args):
    bs = args["bs"]
    gas = args["gas"]
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    random.seed(args["seed"])
    setup(rank, world_size)
    Path(args["save_path"]).mkdir(parents=True, exist_ok=True)

    #model = lm_utils.load_from_path("/home/xuser/nvme1/pretrained/gpt-j-base").half().to(rank)
    outer_model = no_init(lambda: StableDiffusionModel("/mnt/storageserver/workspace/kuru/sdfinetune/models/v14", None)).float()
    outer_model = outer_model.to(rank)
    #outer_model.model.first_stage_model.load_state_dict(clean_dict(torch.load("/home/xuser/nvme1/workspace/aero/stable/sdfinetune/stable-diffusion/logs/animefinetune94525916/checkpoints/last.ckpt")['state_dict']))
    #outer_model.model.first_stage_model = outer_model.model.first_stage_model.to(rank).float()
    #fsdp_model = DDP(model, device_ids=[rank], output_device=rank, gradient_as_bucket_view=True)
    #fsdp_model = DDP(model)
    fsdp_model = outer_model
    #for param in fsdp_model.model.model.parameters():
    #    param.requires_grad = True
    for param in fsdp_model.model.model.parameters():
        param.requires_grad = True
    utils.print_parameters(fsdp_model)
    print("model loaded")
    opt = optimizer.BasedOptimizer(fsdp_model.model.model.parameters(), args, "zero2")
    # TODO: Add load, add evals, add FP16 AMP, and Data Parallel, outputting hidden states from the get_logits function.
    print(opt.curr_step)

    train_dataset = dataset.ShardedImageDataset(dataset_path="/mnt/storageserver/workspace/kuru/sdfinetune/dataset/700kdanbooru", name="smalldanbooru", shuffle=False,
    bsz=bs*gas, threads=8, inner_transform=inner_transform, world_size=world_size, local_rank=rank, global_rank=global_rank)

    train_loader = data.DataLoader(train_dataset, batch_size=None, shuffle=False, num_workers=0, )
    if global_rank == 0:
        wandb.init(project="basedformer-tests", name=args["run_name"], entity="novelaix", config={**args})
        wandb.watch(fsdp_model.model.model, log_freq=200)
        
    fsdp_train(args, fsdp_model, train_loader, opt)
    if args["do_save"]:
        if global_rank == 0:
            torch.save(fsdp_model.model.state_dict(), f"{args['save_path']}/{args['run_name']}-last.pt")
    #lm_utils.save(fsdp_model, Path(args["save_path"]) / "final")
    dist.barrier()
    cleanup()

if __name__ == "__main__":
    train_config = {
        "data_path": "/home/xuser/nvme1/dataset/sigurd-1G.map",
        "save_path": "/mnt/storageserver/workspace/kuru/sdfinetune/checkpoints/anime700k-128bs-01ucg",
        "do_save": True,
        "run_name": "anime700k-128bs-01ucg",
        "lr": 1e-5,
        "end_lr": 8e-6,
        "warmup_steps": 100,
        "anneal_steps": 7850,
        "bs": 8,
        "gas": 2,
        "seed": 69,
        "save_every": 1000,
        "amp": True,
        "loss_scale": True,
        "cast_to": torch.float16,
        "contrastive_loss": False,
        "sample_every": 250,
        "beta1": 0.95,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0.0,
    }

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    main(rank, global_rank, world_size, DotMap(train_config))