from torch import manual_seed
from torch.utils.data import DataLoader as DL
from bionlp.utils import Datasets
import os
import requests

manual_seed(1)


class DataLoader(DL):
    def __init__(self, name, data_dir, **kwargs):
        self.name = name
        self.data_dir = data_dir
        self.data = self.get_data()
        super().__init__(self.data, **kwargs)

    def get_data(self):
        if self.name == "GARD":
            path = os.sep.join([self.data_dir, "GARD_qdecomp.master.03.qtd.xml"])
            if not os.path.isfile(path):
                r = requests.get("https://ceb.nlm.nih.gov/ridem/infobot_docs/GARD_qdecomp.master.03.qtd.xml")
                with open(path, "w") as f:
                    f.write(r.text)
            return Datasets.GARD(path)

        elif self.name == "CADEC":
            text_path = os.sep.join([self.data_dir, "text"])
            ann_path = os.sep.join([self.data_dir, "original"])
            if not os.path.isfile(text_path):
                print("Could not locate {0}".format(text_path))
                print("The data can be downloaded from https://data.csiro.au/dap/landingpage?pid=csiro%3A10948")
            if not os.path.isfile(ann_path):
                print("Could not locate {0}".format(text_path))
                print("These can be downloaded from: https://data.csiro.au/dap/landingpage?pid=csiro%3A10948")
            return Datasets.CADEC(text_path, ann_path)

        else:
            return None


if __name__ == "__main__":
    kwargs = {"batch_size": 10,
              "shuffle": True}

    dataloader = DataLoader("GARD", "/home/will/projects/GARD/data", **kwargs)
    for index, text in enumerate(dataloader.dataset.data):
        dataloader.dataset.data[index] = index
    for i_batch, sample_batched in enumerate(dataloader):
        print(len(sample_batched[0]))