import os 

import click 
import pickle 


from os import path 
from glob import glob 
from tqdm import tqdm 
from logger.log import logger 
from utilities.utils import * 

@click.command()
@click.option('-s', '--source', help='path to images directory', type=click.Path(True))
@click.option('-t', '--target', help='path to target location where features are stored', type=click.File('wb'))
@click.option('-f', '--fmt', help='image file format', default='*.jpg')
@click.option('-m', '--model_path', help='path to vgg16 model', type=click.Path(False))
@click.option('-d', '--dim', help='output dim(reduction)', type=int)
def process_dir(source, target, fmt, model_path, dim):
	logger.debug('process images one by one by using vgg16')

	cwd = get_cwd(__file__)
	path_to_source = path.join(cwd, '..', source)
	image_filepaths = pull_files_from(path_to_source, fmt)

	if path.isfile(model_path):
		logger.debug('The model is already present ...!')
		vgg16_FE = th.load(model_path)
	else: 
		logger.debug('The model will be downloaded ...!')
		vgg16_FE = get_model()

	vgg16_FE.eval()

	features_accumulator = []
	filepaths_accumulator = []
	
	for crr_path in tqdm(image_filepaths):
		features = process_image(crr_path, vgg16_FE)
		features_accumulator.append(features)
		filepaths_accumulator.append(crr_path)

	features_matrix, pca_reducer = reduction(features_accumulator, dim)
	descriptors = {
		'reducer': pca_reducer,
		'features': features_matrix,
		'filepaths': filepaths_accumulator
	}
	pickle.dump(descriptors, target)
	logger.success('The features were stored ...!')


if __name__ == '__main__':
	logger.debug(' ... [processing] ... ')
	process_dir()
