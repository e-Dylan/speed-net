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
import argparse

from test import test_model

# GPU training for google colab
# from google.colab import drive
# from google.colab import files
# drive.mount('/content/drive')
# root_path= '/content/drive/My Drive/Colab/comma/comma-speed-challenge/'

MODEL_SAVE_PATH = os.path.join("C:/Users/Dylan/Desktop/build/python_projects/comma/comma-programming-challenge/models/")

# tensorboard
log_dir = "./logs"
writer = SummaryWriter(log_dir)

def train_val_split(data, val_size):

	val_size = int(len(data) * val_size)

	training_data = data[:-val_size]
	validation_data = data[-val_size:]

	print(f"Training data length: {len(training_data)}")
	print(f"validation data length:  {len(validation_data)}")

	return training_data, validation_data

def augment_brightness(img, brightness_factor):
	# augment an rgb image with a specified brightness factor, cvt back to rgb and return img as np.array.
	hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	hsv_image[:, :, 2] = hsv_image[:, :, 2] * brightness_factor
	img = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
	return img

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

# opticalflow(train_data[0][0], train_data[1][0])

class SNet(nn.Module):
	LEARNING_RATE = 0.001
	EPOCHS = 12
	BATCH_SIZE = 32
	INPUT_CHANNELS = 3
	IMG_SIZE = (200, 66)

	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels=self.INPUT_CHANNELS, out_channels=24, kernel_size=5, stride=2)
		self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2)
		self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2)
		self.conv3_drop = nn.Dropout2d(p=0.5)
		self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3)
		self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
		self.conv_flatten = nn.Flatten()

		self.fc1 = nn.Linear(in_features=1152, out_features=100)
		self.fc2 = nn.Linear(in_features=100, out_features=50)
		# self.fc2_drop = nn.Dropout2d(p=0.5)
		self.fc3 = nn.Linear(in_features=50, out_features=10)
		self.fc4 = nn.Linear(in_features=10, out_features=1)

	def main(self, args):
		self.loss_function = nn.MSELoss()
		self.optimizer = optim.Adam(self.parameters(), lr=self.LEARNING_RATE)

		if args.mode == "train":
			if args.training_data_file is None:
				print("Specify a training data file when training a model.\nIf you do not have one, make one with --mode=make_data, and specify a video and speed file to make data from.")
			else:
				self.train_model(args.model_file, args.training_data_file, args.val_size)

		elif args.mode == "test":
			if args.model_file is None or args.video_file is None:
				print("To test, specify: --model_file --video_file --speed_file.")
			test_model(args.model_file, args.video_file, args.speed_file)

		elif args.mode == "make_data":
			self.make_training_data(args.video_file, args.speed_file)
		else:
			print("Specify a mode: train, test, or make_data.")
			return

	# tests:
		# - remove dropout from conv layer
		# - decrease parameters in dense layers with dropouts to prevent overfit
		# 5 conv layers, 2 dense

	# next to do
		# - add back third channel of rgb-hsv data rather than extracting it, 3 input channels to network
		# - crop input image data (remove steering wheel, etc)
		# - 5 conv nets, 3 dense nets, 

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = self.conv3_drop(x)
		x = F.relu(self.conv4(x))
		x = F.relu(self.conv5(x))
		x = self.conv_flatten(x)

		# x = x.view(-1, x[0].shape[0]*x[0].shape[1]*x[0].shape[2]) # reshape to 128, in features of first nn.Linear layer.
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		# x = self.fc2_drop(x)
		x = F.relu(self.fc3(x)) # no activation function
		x = self.fc4(x)
		return x

	def make_training_data(video_file, speed_file):
		training_data = [] # training_data array filled with image tensors ([0]) and speed labels ([1])
		speeds = [] # speed labels taken from test.txt for each frame
		frame_idx = 0 # index of training_data array to feed in each frame
		cap = cv2.VideoCapture(video_file)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

		# fill speeds array with speed for each frame, corresponding to frame_idx
		speeds_file = open(speed_file, "r")
		for line in speeds_file:
			speed = float(line.split('\n')[0])
			speeds.append(speed)

		# begin video reading
		while cap.isOpened():
			ret, frame = cap.read()

			if frame is None:
				# reached end of video
				np.save(f"training_data-{time.time()}.npy", training_data)
				break

			if ret:
				img = cv2.resize(frame, (320, 160))
				training_data.append([np.asarray(img), speeds[frame_idx]]) # image tensor, label (speed float)

				cv2.imshow('Frame', img)
				cv2.waitKey(1)

				frame_idx += 1

	def make_batches(self, data, batch_size, shuffle):

		x_data = []
		y_data = []

		data_both = []
		flow_data = []
		speed_data = []

		img_channels = 3 # img channels for making flow images with cv2 input is 3 to keep hsv.
						# INPUT_CHANNELS = 2 -> hsv images are converted to 2-channel np arrays to input for network.

		# generate all dense optical flow images from data images and their associated labels
		for i in range(0, len(data)-1, 1):
			# 2 imgs per step = 32 images for batch=32.
			img1 = data[i][0]
			# print(i)
			img2 = data[i+1][0]
			
			# preprocessing images to optimize generalization
			brightness_factor = np.random.uniform() + 0.7
			img1 = augment_brightness(img1, brightness_factor)
			img2 = augment_brightness(img2, brightness_factor)

			# plt.imshow(img2)
			# plt.show()

			flow_img = opticalflow(img1, img2)
			flow_img = cv2.resize(flow_img, (self.IMG_SIZE[0], self.IMG_SIZE[1]), interpolation=cv2.INTER_AREA)
			flow_img = flow_img / 127.5 - 1.0
			
			# flip image and make another flow image
			img1_flip = np.flip(img1, 1)
			img2_flip = np.flip(img2, 1)
			flow_img_flip = opticalflow(img1_flip, img2_flip)
			flow_img_flip = cv2.resize(flow_img_flip, (self.IMG_SIZE[0], self.IMG_SIZE[1]), interpolation=cv2.INTER_AREA)
			flow_img_flip = flow_img_flip / 127.5 - 1.0

			speed1 = data[i][1]
			speed2 = data[i+1][1]
			mean_speed = [np.mean([speed1, speed2])]

			# make array containing both data for shuffling
			data_both.append([flow_img, mean_speed])
			data_both.append([flow_img_flip, mean_speed])

		if shuffle:
			random.shuffle(data_both)
		for i in range(len(data_both)):
			flow_data.append(data_both[i][0])
			speed_data.append(data_both[i][1])

		# sort all DOF data into batches=32
		image_batch = np.empty((batch_size, self.IMG_SIZE[1], self.IMG_SIZE[0], img_channels), dtype="float64")
		label_batch = np.empty((batch_size, 1))

		idx = 0
		for j in range(len(flow_data)):
			if idx % batch_size == 0 and idx != 0:
				# finished a batch
				img_batch = image_batch
				# img_batch = img_batch[:,:,:,[0,2]] # extract hue and value channels with flow data
				img_batch = np.reshape(img_batch, (batch_size, self.INPUT_CHANNELS, self.IMG_SIZE[0], self.IMG_SIZE[1]))
				x_data.append(copy.deepcopy(torch.from_numpy(img_batch)))
				y_data.append(copy.deepcopy(torch.DoubleTensor(label_batch)))
				idx = 0

			image_batch[idx] = flow_data[j]			# idx loops 0-31, makes batches of 32.
			label_batch[idx] = speed_data[j]		# j loops len of flow_data
			idx += 1

		# x_data = array of 32-batch image tensors, [0] is a batch, [1] is a batch, etc.
		# y_data = array of 32-batch labels
		return zip(x_data, y_data)

		# TRAINING
	def train_model(self, model_file, data_file, val_size):
		full_data = np.load(data_file, allow_pickle=True) # small training data file data[0] == img, data[1] = label

		train_start_time = time.time()

		training_data, validation_data = train_val_split(full_data, val_size=val_size)

		train_batch_data = list(self.make_batches(training_data, self.BATCH_SIZE, shuffle=True))
		print(f"Finished loading {len(train_batch_data)} batches (b={self.BATCH_SIZE}) of training samples.")

		validation_batch_data = list(self.make_batches(validation_data, self.BATCH_SIZE, shuffle=False))
		print(f"Finished loading {len(validation_batch_data)} batches (b={self.BATCH_SIZE}) of validation samples.")

		total_steps = 0
		total_accuracy = 0
		total_loss = 0
		val_loss = 0

		prev_mean_epoch_val_loss = 0
		val_epoch_streak = 0

		model = torch.load(MODEL_SAVE_PATH+model_file)
		# model = net
		optimizer = optim.Adam(model.parameters(), lr=self.LEARNING_RATE)

		num_steps_print = 1
		num_steps_test = 25

		for epoch in tqdm(range(self.EPOCHS)):
			running_loss = 0.0
			epoch_val_running_loss = 0
			num_validation_tests = 1

			for i in tqdm(range(0, len(train_batch_data), 1)): # loop over each batch, begin at 0: i = idx, data = data[img, label]
				total_steps += 1

				data = train_batch_data[i]

				b_imgs, b_labels = data
				b_imgs, b_labels = b_imgs.to(device), b_labels.to(device)

				outputs = model(b_imgs)
				# outputs are ([1, 32]) == labels ([1, 32]) for loss function.

				accuracy = torch.mean( outputs / b_labels ).item() # every batch
				total_accuracy += torch.mean( outputs / b_labels ).item() # entirety of training

				loss = self.loss_function(outputs, b_labels)
				loss.backward()
				self.optimizer.step()
				self.optimizer.zero_grad()

				running_loss += loss.item()
				total_loss += loss.item()

				# test on validation data
				if i % num_steps_test == num_steps_test-1:
					idx = randint(0, len(validation_batch_data) - 1)
					test_x, test_y = validation_batch_data[idx] # [0] == imgs
					val_outputs = model(test_x)
					val_loss = loss_function(val_outputs, test_y).item()	
					val_accuracy = torch.mean(val_outputs / test_y).item()

					# tensorboard log validation
					writer.add_scalar('Loss/validation', val_loss, (i+1)*(epoch+1))
					writer.add_scalar('Accuracy/validation', val_accuracy, (i+1)*(epoch+1))
					print(f"[{epoch + 1}/{self.EPOCHS},   {i + 1}/{len(train_batch_data)}], [VALIDATION]: x = {round(val_outputs[0].item(), 3)} y = {round(test_y[0].item(), 3)}, accuracy: {val_accuracy}, loss: {val_loss}")

				if i % num_steps_print == num_steps_print-1: # print update every 10 steps
					print(f"[e={epoch + 1},   b={i + 1}/{len(train_batch_data)}]: x = {round(outputs[0].item(), 3)} y = {round(b_labels[0].item(), 3)}, accuracy: {accuracy}, loss: {running_loss/num_steps_print}")
					writer.add_scalar('Loss/train', running_loss/num_steps_print, (i+1)*(epoch+1))
					writer.add_scalar('Accuracy/train', accuracy, (i+1)*(epoch+1))
					running_loss = 0

			# end each epoch, check if mean validation loss is higher that last epoch, if so, stop training to stop overfit.
			if num_validation_tests > 1:
				num_validation_tests -= 1
			mean_epoch_val_loss = epoch_val_running_loss / num_validation_tests
			if mean_epoch_val_loss > prev_mean_epoch_val_loss and prev_mean_epoch_val_loss != 0: # validation loss increased this epoch
				val_epoch_streak += 1
				if val_epoch_streak >= 2:
					print("Validation loss higher than previous epoch. Stopping training to prevent overfitting.")
					break
			else:
				# val mse decreased for an epoch, reset streak.
				val_epoch_streak = 0

			prev_mean_epoch_val_loss = mean_epoch_val_loss # set new previous val loss

		# finished epochs
		train_time = time.time() - train_start_time
		# torch.save(model, f"MODEL-(={self.EPOCHS}e)-DOF-[full-highway+city-overtrain]-{time.time()}.pth")
		torch.save(model, f"MODEL-(=55e)-DOF-[pass]-{time.time()}.pth")
		print(f"Finished training model: epochs: {self.EPOCHS}. train time: {round(train_time, 4)} sec ({round(train_time/60, 4)} min.). total accuracy: {round((total_accuracy/total_steps),4)}, MSE loss: {round((total_loss/total_steps),4)}. Saved model at {MODEL_SAVE_PATH}")

if __name__ == "__main__":

	if torch.cuda.is_available():
		print("CUDA: True. Running on the GPU.")
	else:
		print("CUDA: False. Slow on the CPU.")
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	parser = argparse.ArgumentParser()
	parser.add_argument("--model_file", help="model to train or use for testing (.pth)", nargs="?", default="MODEL-(=55e)-DOF-[full-highway-vid1]-1601720148.4695241.pth")
	parser.add_argument("--video_file", help="video file name (.mp4). video file data must match speed file data.")
	parser.add_argument("--speed_file", help="training speed file name (.txt) speed file data must match video file data.")
	parser.add_argument("--training_data_file", help="training data file (.npy) of saved video frames and associated speeds.", nargs="?", default="data/training_small.npy")
	parser.add_argument("--val_size", type=float, nargs="?", default=0.2, help="percentage of training data taken for validation.")
	parser.add_argument("--epochs", type=int, nargs = "?", default=40, help="number of epochs for training model.")
	parser.add_argument("--mode", choices=["train", "test", "make_data"], default="test", help="train or test the model, or make training data using a video (.mp4) and speed file (.txt).")
	args = parser.parse_args()
	print("Running SNet.")
	net = SNet().to(device)
	net = net.double()

	net.main(args)








