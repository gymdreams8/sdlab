from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import safety_checker
import torch
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
import os

import datetime as dt


def safety_check_rewrite(self, clip_input, images):
    """
    Remove NSFW checks
    :param self:
    :param clip_input:
    :param images:
    :return:
    """
    return images, [False for i in images]


safety_checker.StableDiffusionSafetyChecker.forward = safety_check_rewrite


class SchedulerLibrary:
    """
    Scheduler Library
    """
    repo_id = "runwayml/stable-diffusion-v1-5"
    subfolder = "scheduler"

    @classmethod
    def get_euler_a(cls):
        return EulerAncestralDiscreteScheduler.from_pretrained(cls.repo_id, subfolder=cls.subfolder)


class Text2Image:
    default_prompt: str = "man"

    def __init__(
            self,
            model_path: str = None,
            prompt: str = None,
            use_cuda: bool = False,
            guidance_scale: int = 5,
            steps: int = 20,
            outputs_path: str = None,
    ):
        self.model_path = os.path.expanduser(model_path)

        # You can enable cuda if you need to
        self.use_cuda = use_cuda

        if not prompt:
            prompt = self.default_prompt
        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.outputs_path = os.path.expanduser(outputs_path)

    def make_folders(self):
        folders = [
            'txt2img-images',
            'txt2img-grids',
            'logs-images',
            'img2img-images',
            'img2img-grids',
            'extras-images',
        ]
        if self.outputs_path:
            for folder in folders:
                os.makedirs(os.path.join(self.outputs_path, folder), exist_ok=True)

    def save_img(self, img):
        """
        Save image
        :param img:
        :return:
        """
        if not self.outputs_path:
            return

        dst_folder = os.path.join(self.outputs_path, 'txt2img-images')
        os.makedirs(dst_folder, exist_ok=True)

        only_files = [
            f for f in os.listdir(dst_folder)
            if all([
                os.path.isfile(os.path.join(dst_folder, f)),
                not f.startswith('.')
            ])
        ]
        file_count = len(only_files)
        file_id = file_count
        timestamp = dt.datetime.now().timestamp()
        filename = f'{file_id:05d}-{timestamp}.png'

        dst_path = os.path.join(dst_folder, filename)
        img.save(dst_path)

    def run(self):
        pipe = StableDiffusionPipeline.from_single_file(
            self.model_path,
            torch_dtype=torch.float32,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
            scheduler=SchedulerLibrary.get_euler_a(),
        )

        if self.use_cuda:
            pipe.to('cuda')

        image = pipe(
            self.prompt,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.steps,
        ).images[0]
        self.save_img(image)
        image.show()


def main():
    """
    Parameters for dev, using Airfuck’s Brute Mix
    :return:
    """
    txt2img = Text2Image(
        model_path='~/_StableDiffusion_Models/Stable-diffusion/airfucksBruteMix_v10.safetensors',
        outputs_path='~/Dropbox/StableDiffusion/outputs_code',
        prompt='man',
        use_cuda=False,
    )
    txt2img.run()


if __name__ == '__main__':
    main()
