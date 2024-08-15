from collections import OrderedDict
from typing import Dict, Tuple

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from hydra.utils import instantiate
import numpy as np
from model import test, train
from pytorch_cifar.models.resnet import ResNet18
import pyzfp
import fpzip
import copy

from scipy.fftpack import dct, idct


class FlowerClient(fl.client.NumPyClient):
    """A standard FlowerClient."""

    def __init__(self, trainloader, vallodaer, cfg) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = vallodaer
        self.cfg = cfg
        # For further flexibility, we don't hardcode the type of model we use in
        # federation. Here we are instantiating the object defined in `conf/model/net.yaml`
        # (unless you changed the default) and by then `num_classes` would already be auto-resolved
        # to `num_classes=10` (since this was known right from the moment you launched the experiment)
        self.model = ResNet18()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        a = []
        og_bytes = []
        comp_bytes = []
        comp_ratio = []
        reconstruction_loss = []
        for _, val in self.model.state_dict().items():
            b = val.cpu().detach()
            c = copy.deepcopy(
                b.numpy()
                if not (b.numel() == 1 and not b)
                else np.array([b], dtype=np.float64)
            )
            if self.cfg.method.name == "zfp":
                c1 = pyzfp.compress(
                    c,
                    tolerance=self.cfg.method.tolerance,
                    precision=self.cfg.method.precision,
                    rate=self.cfg.method.rate,
                )
                c2 = pyzfp.decompress(
                    c1,
                    c.shape,
                    c.dtype,
                    tolerance=self.cfg.method.tolerance,
                    precision=self.cfg.method.precision,
                    rate=self.cfg.method.rate,
                )
                comp_ratio.append(c1.tobytes().__len__() / c.tobytes().__len__())
                reconstruction_loss.append(copy.deepcopy(np.linalg.norm(c - c2).item()))
                # calculate compression ratio
            elif self.cfg.method.name == "fpzip":
                c1 = fpzip.compress(c, precision=self.cfg.method.precision, order="C")
                c2 = fpzip.decompress(c1, order="C")
                c2 = c2.reshape(c.shape)
                comp_ratio.append(c1.__len__() / c.tobytes().__len__())
                og_bytes.append(c.tobytes().__len__())
                comp_bytes.append(c1.__len__())
                if comp_ratio[-1] > 1:
                    c2 = c
                    comp_ratio[-1] = 1
                reconstruction_loss.append(copy.deepcopy(np.linalg.norm(c - c2).item()))
            elif self.cfg.method.name == "dct":
                c1 = dct(dct(c, axis=0, norm="ortho"), axis=1, norm="ortho")
                c2 = idct(idct(c1, axis=0, norm="ortho"), axis=1, norm="ortho")
                comp_ratio.append(c1.tobytes().__len__() / c.tobytes().__len())
                reconstruction_loss.append(copy.deepcopy(np.linalg.norm(c - c2).item()))
            elif self.cfg.method.name == "quantize":
                if self.cfg.method.acc == 32:
                    c1 = c.astype(np.float32)
                else:
                    c1 = c.astype(np.float16)
                c2 = c1
                comp_ratio.append(c1.tobytes().__len__() / c.tobytes().__len())
                reconstruction_loss.append(copy.deepcopy(np.linalg.norm(c - c2).item()))
            elif self.cfg.method.name == "sparsify":
                mask = np.random.random(c.shape) < self.cfg.method.sparsity
                c1 = copy.deepcopy(c)
                c1[mask] = 0
                c2 = c1
                comp_ratio.append(self.cfg.method.sparsity)
                reconstruction_loss.append(copy.deepcopy(np.linalg.norm(c - c2).item()))
            else:
                c2 = c
                comp_ratio.append(1)
                reconstruction_loss.append(0)
            c = c2
            a.append(c)
        print("comp_ratio", comp_ratio)
        print("reconstruction_loss", reconstruction_loss)
        print("og_bytes", og_bytes)
        print("comp_bytes", comp_bytes)
        return a

    def fit(self, parameters, config):
        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        # You could also set this optimiser from a config file. That would make it
        # easy to run experiments considering different optimisers and set one or another
        # directly from the command line (you can use as inspiration what we did for adding
        # support for FedAvg and FedAdam strategies)
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # do local training
        # similarly, you can set this via a config. For example, imagine you have different
        # experiments using wildly different training protocols (e.g. vision, speech). You can
        # toggle between different training functions directly from the config without having
        # to clutter your code with if/else statements all over the place :)
        train(self.model, self.trainloader, optim, epochs, self.device)

        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {"accuracy": accuracy}


def generate_client_fn(trainloaders, valloaders, cfg):
    """Return a function to construct a FlowerClient."""

    def client_fn(cid: str):
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            vallodaer=valloaders[int(cid)],
            cfg=cfg,
        ).to_client()

    return client_fn
