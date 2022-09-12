import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import base64
from torch import autocast
from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.attention import CrossAttention, HyperLogic
import time
from PIL import Image
import k_diffusion as K
import argparse

def no_init(loading_code):
    def dummy(self):
        return
    
    modules = [torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm]
    original = {}
    for mod in modules:
        original[mod] = mod.reset_parameters
        mod.reset_parameters = dummy
    
    result = loading_code()
    for mod in modules:
        mod.reset_parameters = original[mod]
    
    return result

#inherate from dict
class ConfigClass(dict):
    def __init__(self, config):
        #set all the key and values in config to attributes of this class
        for key, value in config.items():
           self.__setitem__(key, value)

    def __getattr__(self, __name: str):
        return self.__getitem__(__name)

    def __setattr__(self, __name: str, __value) -> None:
        return self.__setitem__(__name, __value)

def load_modules(path):
    path = Path(path)
    modules = {}
    if not path.is_dir():
        return

    for file in path.iterdir():
        module = load_module(file, "cuda")
        modules[file.stem] = module
        print(f"Loaded module {file.stem}")

    return modules

def load_module(path, device):
    path = Path(path)
    if not path.is_file():
        print("Module path {} is not a file".format(path))

    network = {
        768: (HyperLogic(768).to(device), HyperLogic(768).to(device)),
        1280: (HyperLogic(1280).to(device), HyperLogic(1280).to(device)),
        640: (HyperLogic(640).to(device), HyperLogic(640).to(device)),
        320: (HyperLogic(320).to(device), HyperLogic(320).to(device)),
    }

    state_dict = torch.load(path)
    for key in state_dict.keys():
        network[key][0].load_state_dict(state_dict[key][0])
        network[key][1].load_state_dict(state_dict[key][1])

    return network

def pil_upscale(image, scale=1):
    device = image.device
    dtype = image.dtype
    image = Image.fromarray((image.cpu().permute(1,2,0).numpy().astype(np.float32) * 255.).astype(np.uint8))
    if scale > 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)), resample=Image.LANCZOS)
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2.*image - 1.
    image = repeat(image, '1 ... -> b ...', b=1)
    return image.to(device)

def fix_batch(tensor, bs):
    return torch.stack([tensor.squeeze(0)]*bs, dim=0)

# mix conditioning vectors for prompts
# @aero
def prompt_mixing(model, prompt_body, batch_size):
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

def sample_start_noise(seed, C, H, W, f, device="cuda"):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    noise = torch.randn([C, (H) // f, (W) // f], device=device).unsqueeze(0)
    return noise

def sample_start_noise_special(seed, request, device="cuda"):
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    noise = torch.randn([request.latent_channels, request.height // request.downsampling_factor, request.width // request.downsampling_factor], device=device).unsqueeze(0)
    return noise

@torch.no_grad()
def encode_image(image, model):
    if isinstance(image, Image.Image):
        image = np.asarray(image)
        image = torch.from_numpy(image).clone()
    
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    #gets image as numpy array and returns as tensor
    def preprocess_vqgan(x):
        x = x / 255.0
        x = 2.*x - 1.
        return x

    #image = image.permute(2, 0, 1).unsqueeze(0).float().cuda()
    image = preprocess_vqgan(image)
    image = model.encode(image).sample()

    return image

@torch.no_grad()
def decode_image(image, model):
    def custom_to_pil(x):
        x = x.detach().float().cpu()
        x = torch.clamp(x, -1., 1.)
        x = (x + 1.)/2.
        x = x.permute(1,2,0).numpy()
        x = (255*x).astype(np.uint8)
        x = Image.fromarray(x)
        if not x.mode == "RGB":
            x = x.convert("RGB")
        return x
    
    image = model.decode(image)
    image = image.squeeze(0)
    image = custom_to_pil(image)
    return image

class StableInterface(nn.Module):
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

class StableDiffusionModel(nn.Module):
    def __init__(self, model_path, module_path=None, dtype="float32", device="cuda"):
        nn.Module.__init__(self)
        if module_path:
            self.premodules = load_modules(module_path)

        model, model_config = self.from_folder(model_path)
        if dtype == "float16":
            typex = torch.float16
        else:
            typex = torch.float32
        self.model = model.to(device).to(typex)
        self.k_model = K.external.CompVisDenoiser(model)
        self.k_model = StableInterface(self.k_model)
        self.device = device
        self.model_config = model_config
        self.plms = PLMSSampler(model)
        self.ddim = DDIMSampler(model)
        self.sampler_map = {
            'plms': self.plms.sample,
            'ddim': self.ddim.sample,
            'k_euler': K.sampling.sample_euler,
            'k_euler_ancestral': K.sampling.sample_euler_ancestral,
            'k_heun': K.sampling.sample_heun,
            'k_dpm_2': K.sampling.sample_dpm_2,
            'k_dpm_2_ancestral': K.sampling.sample_dpm_2_ancestral,
            'k_lms': K.sampling.sample_lms,
        }

    def from_folder(self, folder):
        folder = Path(folder)
        model_config = OmegaConf.load(folder / "config.yaml")
        #model_config['model']['params']['unet_config']['params']['use_checkpoint'] = False
        #model_config['model']['params']['cond_stage_trainable'] = True
        #model_config['model']['params']['unet_config']['params']['use_checkpoint'] = False
        if (folder / "pruned.ckpt").is_file():
            model_path = folder / "pruned.ckpt"
        else:
            model_path = folder / "model.ckpt"
        model = self.load_model_from_config(model_config, model_path)
        return model, model_config

    def load_model_from_config(self, config, ckpt, verbose=False):
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

        model.eval()
        return model
    
    @property
    def get_default_config(self):
        dict_config = {
            'steps': 30,
            'sampler': "k_euler_ancestral",
            'n_samples': 1,
            'image': None,
            'fixed_code': False,
            'ddim_eta': 0.0,
            'height': 512,
            'width': 512,
            'latent_channels': 4,
            'downsampling_factor': 8,
            'scale': 12.0,
            'dynamic_threshold': None,
            'seed': None,
            'stage_two_seed': None,
            'module': None,
            'masks': None,
            'output': None,
        }
        return ConfigClass(dict_config)

    @torch.no_grad()
    @torch.autocast("cuda", enabled=True, dtype=torch.float16)
    def sample(self, request):
        if request.module is not None:
            if request.module == "vanilla":
                pass

            else:
                module = self.premodules[request.module]
                CrossAttention.set_hypernetwork(module)

        if request.seed is None:
            request.seed = torch.randint(0, 2**32, (1,)).item()

        if request.seed is not None:
            torch.manual_seed(request.seed)
            np.random.seed(request.seed)

        if request.image is not None:
            request.steps = 50
            #request.sampler = "ddim_img2img" #enforce ddim for now
            if request.sampler == "plms":
                request.sampler = "k_lms"
            if request.sampler == "ddim":
                request.sampler = "k_lms"

            self.ddim.make_schedule(ddim_num_steps=request.steps, ddim_eta=request.ddim_eta, verbose=False)
            start_code = encode_image(request.image, self.model.first_stage_model).to(self.device)
            start_code = self.model.get_first_stage_encoding(start_code)
            start_code = torch.repeat_interleave(start_code, request.n_samples, dim=0)

            main_noise = []
            start_noise = []
            for seed in range(request.seed, request.seed+request.n_samples):
                main_noise.append(sample_start_noise(seed, request.latent_channels, request.height, request.width, request.downsampling_factor, self.device))
                start_noise.append(sample_start_noise(None, request.latent_channels, request.height, request.width, request.downsampling_factor, self.device))

            main_noise = torch.cat(main_noise, dim=0)
            start_noise = torch.cat(start_noise, dim=0)

            start_code = start_code + (start_noise * request.noise)
            t_enc = int(request.strength * request.steps)

        if request.sampler.startswith("k_"):
            sampler = "k-diffusion"
        
        elif request.sampler == 'ddim_img2img':
            sampler = 'img2img'

        else:
            sampler = "normal"

        if request.image is None:
            main_noise = []
            for seed_offset in range(request.n_samples):
                if request.masks is not None:
                    noise_x = sample_start_noise_special(request.seed, request, self.device)
                else:
                    noise_x = sample_start_noise_special(request.seed+seed_offset, request, self.device)
                    
                if request.masks is not None:
                    for maskobj in request.masks:
                        mask_seed = maskobj["seed"]
                        mask = maskobj["mask"]
                        mask = np.asarray(mask)
                        mask = torch.from_numpy(mask).clone().to(self.device).permute(2, 0, 1)
                        mask = mask.float() / 255.0
                        # convert RGB or grayscale image into 4-channel
                        mask = mask[0].unsqueeze(0)
                        mask = torch.repeat_interleave(mask, request.latent_channels, dim=0)
                        mask = (mask < 0.5).float()

                        # interpolate start noise
                        noise_x = (noise_x * (1-mask)) + (sample_start_noise_special(mask_seed+seed_offset, request, self.device) * mask)

                main_noise.append(noise_x)

            main_noise = torch.cat(main_noise, dim=0)
            start_code = main_noise
        
        prompt = [request.prompt] * request.n_samples
        prompt_condition = prompt_mixing(self.model, prompt[0], request.n_samples)
        if hasattr(self, "prior") and request.mitigate:
            prompt_condition = self.prior(prompt_condition)

        uc = None
        if request.scale != 1.0:
            uc = self.model.get_learned_conditioning(request.n_samples * [""])

        shape = [
            request.latent_channels,
            request.height // request.downsampling_factor,
            request.width // request.downsampling_factor
        ]
        if sampler == "normal":
            #with self.model.ema_scope():
            samples, _ = self.sampler_map[request.sampler](
                S=request.steps,
                conditioning=prompt_condition,
                batch_size=request.n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=request.scale,
                unconditional_conditioning=uc,
                eta=request.ddim_eta,
                dynamic_threshold=request.dynamic_threshold,
                x_T=start_code,
            )

        elif sampler == "k-diffusion":
            #with self.model.ema_scope():
            sigmas = self.k_model.get_sigmas(request.steps)
            if request.image is not None:
                noise = main_noise * sigmas[request.steps - t_enc - 1]
                start_code = start_code + noise
                sigmas = sigmas[request.steps - t_enc - 1:]

            else:
                start_code = start_code * sigmas[0]

            extra_args = {'cond': prompt_condition, 'uncond': uc, 'cond_scale': request.scale}
            samples = self.sampler_map[request.sampler](self.k_model, start_code, sigmas, extra_args=extra_args)

        x_samples_ddim = self.model.decode_first_stage(samples)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

        images = []
        for x_sample in x_samples_ddim:
            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            x_sample = x_sample.astype(np.uint8)
            x_sample = np.ascontiguousarray(x_sample)
            images.append(x_sample) 

        if request.seed is not None:
            torch.seed()
            np.random.seed()

        #set hypernetwork to none after generation
        CrossAttention.set_hypernetwork(None)
        if request.output is not None:
            for i, img in enumerate(images):
                #numpy to PIL
                img = Image.fromarray(img)
                path = Path(request.output)
                path.mkdir(parents=True, exist_ok=True)
                img.save(path / f"{request.prompt}-{request.seed+i}.png")

        return images

if __name__ == '__main__':
    #parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model-path', type=str, required=False, default="/home/xuser/nvme1/stableckpt/v13", help='model folder')
    parser.add_argument('--module-path', type=str, required=False, default=None, help='path for module folder')
    parser.add_argument('--module', type=str, required=False, default=None, help='module name to use')
    parser.add_argument("-p", '--prompt', type=str, required=True, help='prompt')
    parser.add_argument("-n", '--n-samples', type=int, required=False, default=1, help='number of samples')
    parser.add_argument("-s", '--steps', type=int, required=False, default=30, help='number of steps')
    parser.add_argument("-x", '--sampler', type=str, required=False, default="k_euler_ancestral", help='sampler')
    parser.add_argument("-r", '--seed', type=int, required=False, default=None, help='seed')
    parser.add_argument("-o", '--output', type=str, required=False, default="output", help='output folder relative to the current folder')

    args = parser.parse_args()
    model = no_init(lambda: StableDiffusionModel(args.model_path, args.module_path))
    config = model.get_default_config
    config.n_samples = args.n_samples
    config.steps = args.steps
    config.sampler = args.sampler
    config.prompt = args.prompt
    config.seed = args.seed
    config.module = args.module
    config.output = args.output
    print(config)
    images = model.sample(config)