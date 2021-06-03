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
	def __init__(self, source_path, target, vgg16_path, dim, pca_number, batch_size=2):
		self.dim = dim 
		self.vgg16_path = vgg16_path
		self.source_path = source_path
		self.target = target 	 
		self.batch_size = batch_size
		self.pca_number = pca_number
		self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
		logger.info(f'The model will be load on {self.device}')

	def start(self):
		self.filepaths = pull_files_from(self.source_path, '*')
		nb_images = len(self.filepaths)
		vgg16_FE = th.load(self.vgg16_path)
		vgg16_FE.to(self.device)
		source = Source(self.filepaths)
		loader = DataLoader(dataset=source, batch_size=self.batch_size, shuffle=False)
		global_features_accumulator = []
		global_filepaths_accumulator = []
		start = time.time()
		total = 0
		pca_reducer = None
		for batch_images, batch_paths in loader:
			total += batch_images.shape[0]
			logger.debug(f'worker process {total:07d} images >> remainder {nb_images:07d}')
			batch_features = process_batch(batch_images.to(self.device), vgg16_FE)
			if total < self.pca_number:
				global_features_accumulator.append(batch_features)
			else: 
				if pca_reducer is None:
					logger.info('The pca will be computed ...!')
					features_matrix, pca_reducer = reduction(global_features_accumulator, self.dim)
					global_features_accumulator = [features_matrix]
				else:
					features_matrix = pca_reducer.transform(batch_features)
					global_features_accumulator.append(batch_features)	
						
			global_filepaths_accumulator.append(batch_paths)


		descriptors = {
			'reducer': pca_reducer,
			'features': np.vstack(global_features_accumulator),
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
@click.option('--pca_number', type=int, help='number of sample for pca computation')
@click.option('--batch_size', type=int, default=4)
def process_images(source, target, model_path, dim, pca_numer, batch_size):
	try:
		logger.debug('process images one by one by using vgg16')
		if not path.isfile(model_path):
			logger.debug('The model will be downloaded ...!')
			get_model(render=False)

		gpu_FE = GPUFeaturesExtractor(source, target, model_path, dim, pca_number, batch_size)
		gpu_FE.start()
	except KeyboardInterrupt as e:
		pass 

if __name__ == '__main__':
	logger.debug(' ... [processing] ... ')
	process_images()
