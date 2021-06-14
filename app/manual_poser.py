import os
import sys

sys.path.append(os.getcwd())

import PIL.Image
import PIL.ImageTk
import numpy
import torch

from poser.morph_rotate_combine_poser import MorphRotateCombinePoser256Param6
from poser.poser import Poser
from tha.combiner import CombinerSpec
from tha.face_morpher import FaceMorpherSpec
from tha.two_algo_face_rotator import TwoAlgoFaceRotatorSpec
from util import extract_pytorch_image_from_filelike, rgba_to_numpy_image


class ManualPoserApp:
    def __init__(self,
                 poser: Poser,
                 torch_device: torch.device):
        super().__init__()
        self.poser = poser
        self.torch_device = torch_device
        self.pose_size = len(self.poser.pose_parameters())
        self.source_image = None
        self.posed_image = None
        self.current_pose = None
        self.last_pose = None
        self.count = 0
        self.TOTAL_IMAGE: int = 39

    def load_image(self):
        file_name = "data/illust/waifu_00_256.png"
        if len(file_name) > 0:
            image = PIL.Image.open(file_name)
            if image.size[0] != self.poser.image_size() or image.size[1] != self.poser.image_size():
                message = "The loaded image has size %dx%d, but we require %dx%d." \
                          % (image.size[0], image.size[1], self.poser.image_size(), self.poser.image_size())
                print(message)
            self.source_image = extract_pytorch_image_from_filelike(file_name).to(self.torch_device).unsqueeze(dim=0)

    def main_loop(self):
        self.load_image()
        while self.count <= self.TOTAL_IMAGE:
            self.update_image()
            self.count += 1
        get_gif("result/result", self.TOTAL_IMAGE, 0.5)

    def update_pose(self):
        if self.current_pose is None:
            self.current_pose = torch.zeros(self.pose_size, device=self.torch_device)
        for i in range(self.pose_size):
            self.current_pose[i] += 0.02
        return self.current_pose.unsqueeze(dim=0)

    def update_image(self):
        self.update_pose()
        posed_image = self.poser.pose(self.source_image, self.update_pose()).detach().cpu()
        numpy_image = rgba_to_numpy_image(posed_image[0])
        pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(numpy_image * 255.0)), mode='RGBA')
        res_file = "result/result_%d.png" % self.count
        pil_image.save(res_file)


def get_gif(pics_dir, n, t=0.1):
    imgs = []
    for i in range(n):
        pic_name = '{}_{}.png'.format(pics_dir, i)
        temp = PIL.Image.open(pic_name)
        imgs.append(temp)
    for img in reversed(imgs):
        imgs.append(img)
    save_name = '{}.gif'.format(pics_dir)
    imgs[0].save(save_name, save_all=True, append_images=imgs, duration=t, loop=0)
    return save_name


if __name__ == "__main__":
    cuda = torch.device('cuda')
    poser = MorphRotateCombinePoser256Param6(
        morph_module_spec=FaceMorpherSpec(),
        morph_module_file_name="data/face_morpher.pt",
        rotate_module_spec=TwoAlgoFaceRotatorSpec(),
        rotate_module_file_name="data/two_algo_face_rotator.pt",
        combine_module_spec=CombinerSpec(),
        combine_module_file_name="data/combiner.pt",
        device=cuda)
    app = ManualPoserApp(poser=poser, torch_device=cuda)
    app.main_loop()
