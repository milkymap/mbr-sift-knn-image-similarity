from torch.utils.data import Dataset
from utilities.utils import * 

class Source(Dataset):
	def __init__(self, filepaths):
		self.filepaths = filepaths

	def __len__(self):
		return len(self.filepaths)

	def __getitem__(self, index):
		current_path = self.filepaths[index]
		return read_image_for_vgg16(current_path)



