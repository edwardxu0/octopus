import os
import torch
import numpy as np

from torchvision import datasets, transforms
from torchvision.utils import save_image


class Artifact:
    def __init__(self, **kwargs):
        self.data_mode = kwargs["data_mode"]
        self.batch_size = kwargs["batch_size"]
        self.test_batch_size = kwargs["test_batch_size"]
        self.use_cuda = kwargs["use_cuda"]

    def get_data_loader(self):
        if self.data_mode == "nature":
            self._gen_natural_data_loader()
        elif self.data_mode == "pgd":
            self._gen_pgd_data_loader()
        elif self.data_mode == "vae":
            self._gen_vae_data_loader()
        else:
            assert False

        return self.train_loader, self.test_loader

    def _gen_natural_data_loader(self):
        train_kwargs = {"batch_size": self.batch_size}
        test_kwargs = {"batch_size": self.test_batch_size}
        if self.use_cuda:
            cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((self._mean,), (self._std,))]
        )

        # turn on download flag if dataset not downloaded
        dataset1 = eval(f"datasets.{self.name}")(
            "data", train=True, download=True, transform=transform
        )
        dataset2 = eval(f"datasets.{self.name}")(
            "data", train=False, transform=transform
        )

        self.train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        self.test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # adversarial training PGD

    def _gen_pgd_data_loader():
        raise NotImplementedError

    def _gen_vae_data_loader():
        raise NotImplementedError

    def gen_property(self, prop_id, epsilon, prop_dir):
        # select image from test_loader
        data, _ = next(iter(self.test_loader))
        while len(data) < prop_id + 1:
            data = torch.cat((data, next(iter(self.test_loader))[0]))
        img = data[prop_id]

        # save image to disk
        save_image(img, os.path.join(prop_dir, f"{self.name}_{prop_id}.png"))
        npy_img_path = os.path.join(prop_dir, f"{self.name}_{prop_id}.npy")
        img_npy = img.numpy()
        img_npy = img_npy.reshape((1, *img_npy.shape))
        np.save(npy_img_path, img_npy)

        property_path = os.path.join(prop_dir, f"{self.name}_{prop_id}_{epsilon}.py")

        property_lines = [
            "from dnnv.properties import *\n",
            "import numpy as np\n\n",
            'N = Network("N")\n',
            f'x = Image("{npy_img_path}")\n',
            f"epsilon = {epsilon}\n",
            "Forall(\n",
            "    x_,\n",
            "    Implies(\n",
            "        ((x - epsilon) < x_ < (x + epsilon)),\n",
            "        np.argmax(N[:](x_)) == np.argmax(N[:](x)),\n",
            "    ),\n",
            ")\n",
        ]

        with open(property_path, "w+") as property_file:
            property_file.writelines(property_lines)

        return property_path
