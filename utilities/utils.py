import os
import cv2

import numpy as np 
import operator as op 
import itertools as it, functools as ft

import torch as th 
import torch.nn as nn 
import torchvision as tv 
import torchvision.transforms as T 

from os import path, listdir  
from PIL import Image 
from glob import glob 
from logger.log import logger 
from torchvision import models
from sklearn.decomposition import PCA 

Image.MAX_IMAGE_PIXELS = None

def get_cwd(filename):
    return path.dirname(path.realpath(filename))

def read_image(image_path, size):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

def pull_files_from(location, extension='*'):
    contents = listdir(location)
    accumulator = []
    for item in contents:
        path_to_image = path.join(location, item)
        accumulator.append(path_to_image)
    return accumulator

def get_model(render=True):
    # vgg16 has 7 layers on the sequential part
    # head   => [linear, relu, dropout, linear, relu, dropout, linear]
    # retain => [linear, relu, dropout, linear]
    # remove => [relu, dropout, linear]

    cwd = get_cwd(__file__)
    path_to_models = path.join(cwd, '..', 'models')
    logger.debug('download vgg16 with pretrained weights')
    vgg16 = models.vgg16(pretrained=True, progress=True)

    vgg16.classifier = vgg16.classifier[:-3]  
    if not path.isdir(path_to_models):
        os.mkdir(path_to_models)
    th.save(vgg16, path.join(path_to_models, 'vgg16.pt'))
    if render:
    	return vgg16

def get_mapper():
    return T.Compose(
        [
            T.Resize((256, 256)),
            T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

def read_image_for_pil(image_path):
	image = cv2.imread(image_path, cv2.IMREAD_COLOR)
	image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	return image

def read_image_for_vgg16(image_path):
	image = read_image_for_pil(image_path)
	mapper = get_mapper()
	prepared_image = mapper(image)
	return prepared_image

def process_image(image_path, vgg16_FE):
    image = read_image_for_pil(image_path)
    mapper = get_mapper()
    prepared_image = mapper(image)
    with th.no_grad():
        features_1x4096 = vgg16_FE(prepared_image[None, ...])
        return th.squeeze(features_1x4096).numpy()

def process_pil_image(image, vgg16_FE):
    mapper = get_mapper()
    prepared_image = mapper(image)
    with th.no_grad():
        features_1x4096 = vgg16_FE(prepared_image[None, ...])
        return th.squeeze(features_1x4096).numpy()

def process_batch(input_batch, vgg16_FE):
	with th.no_grad():
		features_Nx4096 = vgg16_FE(input_batch)
		return th.squeeze(features_Nx4096.to('cpu')).numpy()

def reduction(features, target_dim):
    pca_reducer = PCA(n_components=target_dim)
    new_features = pca_reducer.fit_transform(np.vstack(features))
    return new_features, pca_reducer

def get_SIFT(image, extractor):
    keypoints, descriptor = extractor.detectAndCompute(image, None) 
    return keypoints, descriptor

def map_SIFT2MBRSIFT(descriptor):
    sink = []
    for row in descriptor:
        breaked_row = np.split(row, 4)
        breaked_row.reverse()
        chunks_acc = []
        for groups in breaked_row:
            breaked_chunk = np.split(groups, 4)
            for chunk in breaked_chunk:
                head, *remainder = chunk
                new_chunk = np.hstack([head, np.flip(remainder)])
                chunks_acc.append(new_chunk)
            
        sink.append(row)
        sink.append(np.hstack(chunks_acc))
    return np.vstack(sink) 

def compare_descriptors(source_des, target_des, threshold):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(source_des, target_des, k=2)

    valid_matches = 0 
    for left, right in matches:
        if left.distance < threshold * right.distance:
            valid_matches = valid_matches + 1 

    return valid_matches


def search(pid, source, target_paths, size, score, threshold, barrier, controller, mutex, condition, queue):
    with controller.get_lock():
        controller.value += 1  # increament the controller 
    logger.debug(f'worker : {pid} wait on the barrier')
    barrier.wait()  # wait for other worker to be ready
    sift = cv2.SIFT_create()
    source_img = read_image(source, size)
    source_keypoints, source_features = get_SIFT(source_img, sift)
    extended_source_features = map_SIFT2MBRSIFT(source_features)

    accumulator = []
    for current_path in target_paths:
        target_img = read_image(current_path, size)
        target_keypoints, target_features = get_SIFT(target_img, sift)
        extended_target_features = map_SIFT2MBRSIFT(target_features)
        metrics = compare_descriptors(extended_source_features, extended_target_features, threshold)
        weights = metrics / np.maximum(len(source_keypoints), len(target_keypoints))
        logger.debug(f'worker : {pid} >> mbr-sift score:{weights:07.3f}')
        if weights > score:
            accumulator.append((current_path, weights))

    queue.put(accumulator)
    with controller.get_lock():
        controller.value -= 1
    
    mutex.acquire()
    condition.notify()
    mutex.release()