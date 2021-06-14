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

class ZMQFeaturesExtractor:
	def __init__(self, source_path, target, nb_workers, vgg16_path, dim, aggregator_port, batch_size=2):
		self.dim = dim 
		self.nb_workers = nb_workers
		self.vgg16_path = vgg16_path
		self.source_path = source_path
		self.target = target 
		self.aggregator_port = aggregator_port 
		self.batch_size = batch_size
		
	def start(self):
		self.filepaths = pull_files_from(self.source_path, '*')
		nb_images = len(self.filepaths)
		nb_images_per_worker = nb_images // self.nb_workers

		barrier = mp.Barrier(self.nb_workers)
		controller = mp.Value('i', 0)
		workers = []
		for idx in range(self.nb_workers):
			begin = idx * nb_images_per_worker 
			end = (idx + 1) * nb_images_per_worker
			filepaths_scope = self.filepaths[begin:end] 
			workers.append(
				mp.Process(target=self.worker, args=[idx, barrier, controller, filepaths_scope])
			)
			workers[-1].start()

		aggregator = mp.Process(target=self.aggregate, args=[controller]) 
		aggregator.start()

	def worker(self, pid, barrier, controller, filepaths_scope):		
		try:
			with controller.get_lock():
				controller.value = controller.value + 1

			ctx = zmq.Context()
			pusher = ctx.socket(zmq.PUSH)
			pusher.connect(f'tcp://localhost:{self.aggregator_port}')
			barrier.wait() 

			vgg16_FE = th.load(self.vgg16_path)
			source = Source(filepaths_scope)
			loader = DataLoader(dataset=source, batch_size=self.batch_size, shuffle=False)
			for batch_images, batch_paths in loader:
				logger.debug(f'worker n° {pid} process {len(batch_paths):03d} images')
				batch_features = process_batch(batch_images, vgg16_FE)
				
				pusher.send_pyobj({
					'features': batch_features, 
					'filepaths': batch_paths
				})

			with controller.get_lock():
				controller.value = controller.value - 1
		except Exception as e:
			logger.warning(e)
		finally:
			pusher.close()
			ctx.term()

	def aggregate(self, controller):
		try:
			start = time.time()
			ctx = zmq.Context()
			puller = ctx.socket(zmq.PULL)
			puller.bind(f'tcp://*:{self.aggregator_port}')
			poller = zmq.Poller()
			poller.register(puller, zmq.POLLIN)

			global_features_accumulator = []
			global_filepaths_accumulator = []
			keep_reducing = True 
			while keep_reducing:
				events = dict(poller.poll(100))
				if puller in events:
					if events[puller] == zmq.POLLIN: 
						incoming_data = puller.recv_pyobj()
						global_features_accumulator.append(incoming_data['features'])
						global_filepaths_accumulator.append(incoming_data['filepaths'])
				keep_reducing = controller.value > 0

			# reduce ...!
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

		except Exception as e:
			logger.warning(e)
		finally:
			poller.unregister(puller)
			puller.close()
			ctx.term()

@click.command()
@click.option('-s', '--source', help='path to images directory', type=click.Path(True))
@click.option('-t', '--target', help='path to target location where features are stored', type=click.File('wb'))
@click.option('-m', '--model_path', help='path to vgg16 model', type=click.Path(False))
@click.option('-d', '--dim', help='output dim(reduction)', type=int)
@click.option('-n', '--nb_workers', help='number of workers', type=int, default=4)
@click.option('-p', '--port', help='port of the aggregator', default=8500, type=int)
@click.option('--batch_size', type=int, default=4)
def process_images(source, target, model_path, dim, nb_workers, port, batch_size):
	try:
		logger.debug('process images one by one by using vgg16')
		if not path.isfile(model_path):
			logger.debug('The model will be downloaded ...!')
			get_model(render=False)

		zmq_FE = ZMQFeaturesExtractor(source, target, nb_workers, model_path, dim, port, batch_size)
		zmq_FE.start()
	except KeyboardInterrupt as e:
		pass 

if __name__ == '__main__':
	logger.debug(' ... [processing] ... ')
	process_images()
