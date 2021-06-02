import os 
import cv2
import zmq 

import click  
import pickle 
import joblib

import operator as op 
import itertools as it, functools as ft 

import torch as th 

from logger.log import logger 
from utilities.utils import * 
from sklearn.neighbors import NearestNeighbors as KNN 
from multiprocessing import Process, Barrier, Lock, Condition, Value, Queue  

@click.group(invoke_without_command=True, chain=True)
@click.option('--debug/--no-debug', default=False)
@click.pass_context
def main(ctx, debug):
	if not ctx.invoked_subcommand:
		logger.debug(' ... [main command] ... ')
	ctx.obj['debug'] = debug 

@click.command()
@click.option('-n', '--k_nearest', help='number of nearest neighbors', type=int)
@click.option('-d', '--descriptor_path', help='path to descriptor', type=click.File('rb'))
@click.option('-k', '--knn_path', help='path where the knn model will be stored', type=click.File('wb'))
@click.option('--save/--no-save', help='should save or not the knn model', default=True)
@click.pass_context
def train(ctx, k_nearest, descriptor_path, knn_path, save):
	descriptor = pickle.load(descriptor_path)
	knn_agent = KNN(
        n_neighbors=k_nearest, algorithm="brute", metric="euclidean"
    ).fit(descriptor['features'])
	if save:
		joblib.dump(knn_agent, knn_path)
	ctx.obj['knn'] = knn_agent

@click.command()
@click.option('-n', '--k_nearest', help='number of nearest neighbors', type=int)
@click.option('-s', '--source', help='path to source image', type=click.Path(True))
@click.option('-k', '--knn_path', help='path to knn model', type=click.File('rb'))
@click.option('-m', '--vgg16_path', help='path to vgg16 model', type=click.Path(True))
@click.option('-d', '--descriptor_path', help='path to descriptor', type=click.File('rb'))
@click.pass_context
def nearest(ctx, k_nearest, source, knn_path, vgg16_path, descriptor_path):
	if 'knn' in ctx.obj:
		knn_agent = ctx.obj['knn']
	else:
		knn_agent = joblib.load(knn_path)
	
	descriptor = pickle.load(descriptor_path)
	vgg16_FE = th.load(vgg16_path)
	features = process_image(source, vgg16_FE)
	reduced_features = descriptor['reducer'].transform(features[None, :])
	distances, indices = knn_agent.kneighbors(reduced_features, k_nearest)
	response = list(op.itemgetter(*indices[0])(descriptor['filepaths']))
	print('\n'.join(response))
	ctx.obj['source'] = source 
	ctx.obj['target_paths'] = response

@click.command()
@click.option('--size', help='size of resizing version', type=click.Tuple([int, int]))
@click.option('--score', help='threshold value for weights metrics', default=0.5, type=float)
@click.option('--threshold', help='threshold value for SIFT keypoint match', default=0.75, type=float)
@click.option('--nb_workers', help='number of workers', type=int, default=8)
@click.pass_context
def search_map(ctx, size, score, threshold, nb_workers):
	source = ctx.obj['source']
	target_paths = ctx.obj['target_paths']
	window_size = len(target_paths) // nb_workers
	barrier = Barrier(nb_workers)
	mutex = Lock()
	condition = Condition(mutex)
	shared_queue = Queue()
	controller = Value('i', 0)

	for idx in range(nb_workers):
		current_window = target_paths[idx*window_size:(idx+1)*window_size]
		p = Process(
			target=search, 
			args=[
				idx, 
				source, 
				current_window, 
				size, 
				score, 
				threshold, 
				barrier, 
				controller, 
				mutex, 
				condition, 
				shared_queue
			])
		p.start()
	mutex.acquire()
	logger.debug('main process wait for worker to finish ...!')
	condition.wait_for(lambda: controller.value == 0)

	sink_accumulator = []
	while not shared_queue.empty():
		sink_accumulator.append(shared_queue.get())
	sink_accumulator = list(it.chain(*sink_accumulator))
	print(sink_accumulator)


main.add_command(train)
main.add_command(nearest)
main.add_command(search_map)

if __name__ == '__main__':
	logger.debug(' ... [modelization] ... ')
	main(obj={})
