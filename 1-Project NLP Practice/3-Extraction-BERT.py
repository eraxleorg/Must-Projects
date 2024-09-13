import inspect
import torch
import lightning as L
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_dataset



class ResumeDataModule(L.LightningDataModule):
     def __init__(self, `batch_size`=16):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        
        
        parent_init_signature = inspect.signature(ResumeDataModule.__init__)
        # Print the parameters
print(parent_init_signature)

print(dir(L.LightningDataModule))
help(L.LightningDataModule.__init__)
