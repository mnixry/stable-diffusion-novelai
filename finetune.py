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

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}

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

def process_tags(tags, min_tags=1, max_tags=32, type_dropout=0.75, keep_important=1.00, keep_jpeg_artifacts=True, sort_tags=False):
    if isinstance(tags, str):
        tags = tags.split(" ")
    final_tags = {}

    tag_dict = {tag: True for tag in tags}
    pure_tag_dict = {tag.split(":", 1)[-1]: tag for tag in tags}
    for bad_tag in ["absurdres", "highres", "translation_request", "translated", "commentary", "commentary_request", "commentary_typo", "character_request", "bad_id", "bad_link", "bad_pixiv_id", "bad_twitter_id", "bad_tumblr_id", "bad_deviantart_id", "bad_nicoseiga_id", "md5_mismatch", "cosplay_request", "artist_request", "wide_image", "author_request", "artist_name"]:
        if bad_tag in pure_tag_dict:
            del tag_dict[pure_tag_dict[bad_tag]]

    if "rating:questionable" in tag_dict or "rating:explicit" in tag_dict:
        final_tags["nsfw"] = True

    base_chosen = []
    for tag in tag_dict.keys():
        parts = tag.split(":", 1)
        if parts[0] in ["artist", "copyright", "character"] and random.random() < keep_important:
            base_chosen.append(tag)
        if len(parts[-1]) > 1 and parts[-1][0] in ["1", "2", "3", "4", "5", "6"] and parts[-1][1:] in ["boy", "boys", "girl", "girls"]:
            base_chosen.append(tag)
        if parts[-1] in ["6+girls", "6+boys", "bad_anatomy", "bad_hands"]:
            base_chosen.append(tag)

    tag_count = min(random.randint(min_tags, max_tags), len(tag_dict.keys()))
    base_chosen_set = set(base_chosen)
    chosen_tags = base_chosen + [tag for tag in random.sample(list(tag_dict.keys()), tag_count) if tag not in base_chosen_set]
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
        if bad_tag in pure_tag_dict:
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
            tags = process_tags(tags, min_tags=args["min_tags"], max_tags=args["max_tags"])
            tags_batch.append(tags)

        #copy
        tags_batch_original = tags_batch.copy()
        ucg = args["ucg"]
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
            with torch.cuda.amp.autocast(enabled=args["amp"], dtype=dtype_map[args["cast_to"]]):
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
        #do ema
        if args["use_ema"]:
            model.model.model_ema(model.model.model)

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
        
        if args["do_save"] and counter % args["save_every"] == 0:
            if global_rank == 0:
                torch.save(model.model.state_dict(), f"{args['save_path']}/{args['run_name']}-{counter}.pt")

        if args["do_save"] == False:
            if global_rank == 0:
                print("Alert, you are NOT SAVING THE MODEL!")

        if counter % args["sample_every"] == 0:
            if global_rank == 0:
                torch_random_state = torch.get_rng_state()
                numpy_random_state = np.random.get_state()
                python_random_state = random.getstate()

                req = model.get_default_config
                req.n_samples = 4
                req.steps = 35
                req.prompt = "An anime girl with black shorts laying down on the bed, looking at you, trending on pixiv"
                req.seed = 42
                images = model.sample(req, ema=False)
                imagesx = [Image.fromarray(image) for image in images]
                imagesx = [wandb.Image(image, caption=req.prompt) for image in imagesx]
                images = model.sample(req)
                imagesx_ema = [Image.fromarray(image) for image in images]
                imagesx_ema = [wandb.Image(image, caption=req.prompt + ":ema") for image in imagesx_ema]

                req = model.get_default_config
                req.n_samples = 4
                req.steps = 35
                req.prompt = "Tags: 1girl, long hair, smile, blue eyes, open mouth, brown hair"
                req.seed = 42
                images = model.sample(req, ema=False)
                imagesy = [Image.fromarray(image) for image in images]
                imagesy = [wandb.Image(image, caption=req.prompt) for image in imagesy]
                images = model.sample(req)
                imagesy_ema = [Image.fromarray(image) for image in images]
                imagesy_ema = [wandb.Image(image, caption=req.prompt + ":ema") for image in imagesy_ema]

                req = model.get_default_config
                req.n_samples = 4
                req.steps = 35
                req.prompt = "Tags: red bikini"
                req.seed = 42
                images = model.sample(req, ema=False)
                imagesz = [Image.fromarray(image) for image in images]
                imagesz = [wandb.Image(image, caption=req.prompt) for image in imagesz]
                images = model.sample(req)
                imagesz_ema = [Image.fromarray(image) for image in images]
                imagesz_ema = [wandb.Image(image, caption=req.prompt + ":ema") for image in imagesz_ema]

                req = model.get_default_config
                req.n_samples = 4
                req.steps = 35
                req.prompt = "Tags: red bikini, hatsune miku"
                req.seed = 42
                images = model.sample(req, ema=False)
                imageso = [Image.fromarray(image) for image in images]
                imageso = [wandb.Image(image, caption=req.prompt) for image in imageso]
                images = model.sample(req)
                imageso_ema = [Image.fromarray(image) for image in images]
                imageso_ema = [wandb.Image(image, caption=req.prompt + ":ema") for image in imageso_ema]
        
                imageslog = imagesx + imagesy + imagesz + imageso + imagesx_ema + imagesy_ema + imagesz_ema + imageso_ema
                req = model.get_default_config
                req.n_samples = 4
                req.steps = 35
                req.prompt = tags_batch_original[0]
                req.seed = 42
                images = model.sample(req, ema=False)
                imagesrandom = [Image.fromarray(image) for image in images]
                imagesrandom = [wandb.Image(image, caption=req.prompt) for image in imagesrandom]
                images = model.sample(req)
                imagesrandom_ema = [Image.fromarray(image) for image in images]
                imagesrandom_ema = [wandb.Image(image, caption=req.prompt + ":ema") for image in imagesrandom_ema]
                imagesrandom = imagesrandom + imagesrandom_ema
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

    config_path = args["config_path"] if "config_path" in args else None
    #model = lm_utils.load_from_path("/home/xuser/nvme1/pretrained/gpt-j-base").half().to(rank)
    outer_model = no_init(lambda: StableDiffusionModel(args["model_path"], mode=args["mode"], config_path=config_path)).float()
    outer_model = outer_model.to(rank)
    outer_model.model.cond_stage_model.return_layer = -2
    outer_model.model.cond_stage_model.do_final_ln = True
    outer_model.model.cond_stage_model.inference_mode = False
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

    train_dataset = dataset.ShardedImageDataset(dataset_path=args["data_path"], index_path=args["index_path"], name="danbooru", shuffle=False,
    bsz=bs*gas, threads=8, inner_transform=inner_transform, world_size=world_size, local_rank=rank, global_rank=global_rank)
    train_dataset.shard(shuffle=True, epoch=args["epoch"], seed=args["seed"])

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
    '''
    train_config = {
        "data_path": "/mnt/storageserver/workspace/kuru/sdfinetune/dataset/fulldanbooru",
        "index_path": "/mnt/storageserver/workspace/kuru/sdfinetune/gsq.index",
        #"model_path": "/mnt/storageserver/workspace/kuru/sdfinetune/models/anime700k-64bs-0.1ucg-penultimate-clip-1epoch-ema-restore",
        "model_path": "/mnt/storageserver/workspace/kuru/sdfinetune/models/v14",
        "save_path": "/mnt/storageserver/workspace/kuru/sdfinetune/checkpoints/animeno-e-64bs-0.1ucg-penultimate-clip-6epoch-1-22prompt",
        "do_save": True,
        "run_name": "animeno-e-64bs-0.1ucg-penultimate-clip-6epoch-1-22prompt",
        "lr": 1e-5,
        "end_lr": 5e-6,
        "warmup_steps": 100,
        #"anneal_steps": 80000,
        #"anneal_steps": 7850*2,
        "anneal_steps": 69000 * 6,
        "bs": 8,
        "gas": 1,
        "seed": 69,
        #"seed": 42,
        "save_every": 1000*2,
        "amp": False,
        "loss_scale": False,
        "cast_to": torch.float16,
        "contrastive_loss": False,
        "sample_every": 250*2,
        "beta1": 0.95,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0.0,
        "use_ema": True,
        "ucg": 0.1,
        "min_tags": 1,
        "max_tags": 22, 
        "mode": "stable",
        "epoch": 6,
    }
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-c", "--config", type=str, required=True)

    args = argparser.parse_args()
    #read train config from yaml with OmegaConf
    train_config = OmegaConf.load(args.config)
    print(train_config)

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    main(rank, global_rank, world_size, DotMap(train_config))
