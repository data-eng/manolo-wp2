import os
import shutil

from manolo.base.wrappers.other_packages import json
from manolo.base.wrappers.pytorch import torch

def create_exp_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)
	print('Experiment dir : {}'.format(path))


def load_pretrained_model(model, pretrained_dict):
	model_dict = model.state_dict()
	# 1. filter out unnecessary keys
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict) 
	# 3. load the new state dict
	model.load_state_dict(model_dict)

def save_checkpoint(state, is_best, save_root, qz_map=None):
	save_path = os.path.join(save_root, 'checkpoint.pth.tar')
	torch.save(state, save_path)
	if qz_map!=None:
		save_qz_path = os.path.join(save_root, 'checkpoint_qz_map.pth.tar')
		with open(save_qz_path, 'w') as f:
			json.dump(qz_map, f)
	
	if is_best:
		best_save_path = os.path.join(save_root, 'model_best.pth.tar')
		shutil.copyfile(save_path, best_save_path)