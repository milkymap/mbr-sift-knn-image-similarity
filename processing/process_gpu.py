import os 
import zmq 

import click 
import pickle 

import time 

import numpy as np 
import operator as op 
import itertools as it, functools as ft 

import multiprocessing as mp 

import torch as th 

from os import path 
from glob import glob 
from tqdm import tqdm 
from logger.log import logger 
from processing.loader import Source
from torch.utils.data import DataLoader 
from utilities.utils import *

class GPUFeaturesExtractor:
	def __init__(self, source_path, target, vgg16_path, dim, batch_size=2):
		self.dim = dim 
		self.vgg16_path = vgg16_path
		self.source_path = source_path
		self.target = target 	 
		self.batch_size = batch_size
		self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
		logger.info(f'The model will be load on {self.device}')

	def start(self):
		self.filepaths = pull_files_from(self.source_path, '*')
		vgg16_FE = th.load(self.vgg16_path)
		vgg16_FE.to(self.device)
		source = Source(self.filepaths)
		loader = DataLoader(dataset=source, batch_size=self.batch_size, shuffle=False)
		global_features_accumulator = []
		global_filepaths_accumulator = []
		for batch_images, batch_paths in loader:
			logger.debug(f'worker process {len(batch_paths):03d} images')
			batch_features = process_batch(batch_images.to(self.device), vgg16_FE)
			global_features_accumulator.append(batch_features)
			global_filepaths_accumulator.append(batch_paths)

		features_matrix, pca_reducer = reduction(global_features_accumulator, self.dim)
		descriptors = {
			'reducer': pca_reducer,
			'features': features_matrix,
			'filepaths': list(it.chain(*global_filepaths_accumulator))
		}
		pickle.dump(descriptors, self.target)
		duration = time.time() - start
		logger.success('The features were stored ...!')
		logger.debug(f'This process took : {duration} s')

	
@click.command()
@click.option('-s', '--source', help='path to images directory', type=click.Path(True))
@click.option('-t', '--target', help='path to target location where features are stored', type=click.File('wb'))
@click.option('-m', '--model_path', help='path to vgg16 model', type=click.Path(False))
@click.option('-d', '--dim', help='output dim(reduction)', type=int)
@click.option('--batch_size', type=int, default=4)
def process_images(source, target, model_path, dim, batch_size):
	try:
		logger.debug('process images one by one by using vgg16')
		if not path.isfile(model_path):
			logger.debug('The model will be downloaded ...!')
			get_model(render=False)

		gpu_FE = GPUFeaturesExtractor(source, target, model_path, dim, batch_size)
		gpu_FE.start()
	except KeyboardInterrupt as e:
		pass 

if __name__ == '__main__':
	logger.debug(' ... [processing] ... ')
	process_images()
