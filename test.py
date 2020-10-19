#!/usr/bin/env python

import os
import time
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

import cv2
import numpy as np
import random
from random import randint
from tqdm import tqdm
import matplotlib.pyplot as plt

ROOT_PATH = os.path.join("C:/Users/Dylan/Desktop/build/python_projects/comma/comma-programming-challenge/")
MODEL_PATH = os.path.join("C:/Users/Dylan/Desktop/build/python_projects/comma/comma-programming-challenge/models/")

if torch.cuda.is_available():
	print("CUDA: True. Running on the GPU.")
else:
	print("CUDA: False. Slow on the CPU.")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 1
IMG_SIZE = (200, 66)
IMG_CHANNELS = 3 # images taken in from video
INPUT_CHANNELS = 3 # images converted and input to network

log_dir = './test/logs'

FONT = cv2.FONT_HERSHEY_SIMPLEX

def opticalflow(img1, img2): # DOF
	
	hsv = np.zeros_like(img1)
	hsv[...,1] = 255

	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	# flow_data = cv2.calcOpticalFlowFarneback(prev, nxt, None, 0.4, 1, 12, 2, 8, 1.2, 0)
	flow = cv2.calcOpticalFlowFarneback(gray1, gray2, flow=None, pyr_scale=0.5, 
                                        levels=3, winsize=15, iterations=3, 
                                        poly_n=5, poly_sigma=1.2, flags=0)

	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

	hsv[:,:,0] = ang*(180/np.pi/2)
	hsv[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
	# hsv[...,2] = (mag * 15).astype(int)
	
	rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

	# plt.imshow(rgb)
	# plt.show()

	return rgb

def test_model(model_file, video_file, gt_file=None):
	model = torch.load(MODEL_PATH+model_file, map_location=torch.device('cpu'))

	cap = cv2.VideoCapture(video_file)

	frame_idx = 0

	prev = None
	current = None

	speeds = []
	# get ground truthing data if known
	if gt_file is not None:
		groundtruth_file = open(gt_file, "r")
		for line in groundtruth_file:
			speed = float(line.split('\n')[0])
			speeds.append(speed)

	preds = []

	while 1:
			ret, frame = cap.read()

			if ret:
				# frame exists
				if current is not None:
					prev = current

				current = frame

				# waited until second frame to get a prev and current
				if prev is not None:
					img_batch = np.empty((BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[0], IMG_CHANNELS))
					
					flow_img_bgr = opticalflow(prev, current)
					flow_img = cv2.resize(flow_img_bgr, (IMG_SIZE[0], IMG_SIZE[1]), interpolation=cv2.INTER_AREA)
					flow_img = flow_img / 127.5 - 1.0

					# make input batch tensor with each image
					img_batch[0] = flow_img
					img_batch = np.reshape(img_batch, (BATCH_SIZE, INPUT_CHANNELS, IMG_SIZE[0], IMG_SIZE[1]))
					x = torch.from_numpy(img_batch)

					pred = model(x)
					
					speed = pred.item()
					
					preds.append(speed);

					test_frame = np.hstack((frame, flow_img_bgr))
					cv2.putText(test_frame, f"Predicted Speed: { round(speed, 2) }", (5,35), FONT, 0.55, (20, 255, 20), 2)
					cv2.putText(test_frame, f"Exit: q", (5,470), FONT, 0.55, (255, 255, 255), 1)

					if gt_file is not None:
						cv2.putText(test_frame, f"Actual Speed: {round(speeds[frame_idx], 2)}", (5, 65), FONT, 0.55, (255, 255, 255), 2)
						error = speeds[frame_idx] - speed
						cv2.putText(test_frame, f"Error: {round(error, 2)}", (5,95), FONT, 0.55, (0, 0, 255), 2)
						print(f"[f={frame_idx}] Predicted speed: {round(speed,3)}, actual speed: {round(speeds[frame_idx],3)}.")
					
					cv2.imshow('Testing Frame', test_frame)

					frame_idx += 1

					k = cv2.waitKey(1)
					if k == ord('q'):
						break

			else:
				# frames ended
				break

	# write all predictions out to outputs file
	with open("test.txt", "w") as f:
		for i in range(len(preds)):
			f.write(f"{preds[i]}\n")

	cap.release()
	cv2.destroyAllWindows()

# if __name__ == "__main__":
# 	test_model(video_file="data/train.mp4", model_path=MODEL_PATH, gt_file="data/train.txt")

# 	alpha = 0.7
# 	img_batch = img_batch[:,:,:,[0,2]]

# 	flow_overlay = cv2.addWeighted(frame[:,:,:], alpha, flow_img_overlay[:,:,:], 1-alpha, 0)
# 	frame[:,:] = flow_overlay
	