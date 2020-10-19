import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

learning_rate = 0.001
img_size = (320, 160, 3)

MODEL_SAVE_PATH = os.path.join("C:/Users/Dylan/Desktop/build/python_projects/comma/comma-programming-challenge/models")

class dNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 24, kernel_size=5, stride=2)
		self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
		self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
		# self.conv3_drop = nn.Dropout2d()
		self.conv4 = nn.Conv2d(48, 64, kernel_size=5, stride=2)
		self.conv5 = nn.Conv2d(64, 64, kernel_size=5, stride=2)
		self.fc1 = nn.Linear(448, 400) # 448 = x.shape[0] * x.shape[1] * x.shape[2] after passing conv layers.
		self.fc2 = nn.Linear(400, 100)
		self.fc3 = nn.Linear(100, 50)
		self.fc4 = nn.Linear(50, 10)
		self.fc5 = nn.Linear(10, 1)

	def convs(self, x):
		x = F.elu(self.conv1(x))
		x = F.elu(self.conv2(x))
		x = F.elu(self.conv3(x))
		x = F.elu(self.conv4(x))
		x = F.elu(self.conv5(x))
		return x

	def forward(self, x):
		x = self.convs(x)
		x = x.view(-1, 448) # reshape to 448 to feed into fc1 layer (448 is shape of (320, 160) image tensor after passing conv layers)
		x = F.elu(self.fc1(x))
		x = F.elu(self.fc2(x))
		x = F.elu(self.fc3(x))
		x = F.elu(self.fc4(x))
		x = self.fc5(x) # no activation function
		return x

net = dNet()
loss_function = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# reads in all text.txt labels and fills array.
# reads in live video frames from train.mp4, then feeds with each frame + label live.
def get_training_data():
	training_data = [] # training_data array filled with image tensors ([0]) and speed labels ([1])
	speeds = [] # speed labels taken from test.txt for each frame
	frame_idx = 0 # index of training_data array to feed in each frame
	quarter = 1
	cap = cv2.VideoCapture("data/train2.mp4")
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	# fill speeds array with speed for each frame, corresponding to frame_idx
	speeds_file = open("data/train2.txt", "r")
	for line in speeds_file:
		speed = float(line.split('\n')[0])
		speeds.append(speed)

	# begin video reading
	while cap.isOpened():
		ret, frame = cap.read()

		# if frame is None:
		# 	# reached end of video
		# 	np.save(f"training_full_quarter{quarter}.npy", training_data)
		# 	break

		if ret:
			img = cv2.resize(frame, (320, 160))
			training_data.append([np.asarray(img), speeds[frame_idx]]) # image tensor, label (speed float)

			cv2.imshow('Frame', img)
			cv2.waitKey(1)
			print(training_data[frame_idx])

			frame_idx += 1

			if frame_idx % int(total_frames/4) == 0 and frame_idx is not 0:
				# save video as npy data files in batches of quarters
				np.save(f"training_city_full.npy", training_data)
				print(f"Saving data file (quarter: {quarter})")
				training_data = []
				frame_idx = 0
				quarter += 1

BATCH_SIZE = 32
EPOCHS = 5

def train(training_data):

	x = torch.Tensor([i[0] for i in training_data]).view(-1, 1, 320, 160) # all imgs in a tensor
	y = torch.Tensor([i[1] for i in training_data]) # all labels

	val_size = int(len(x) * 0.10) # 10% of training data is validation

	train_x = x[:-val_size] # 90% of imgs are training data
	train_y = y[:-val_size] # "
	
	for epoch in range(EPOCHS):
		print(epoch)
		for i in tqdm(range(0, len(train_x), BATCH_SIZE)):

			batch_x = train_x[i:i + BATCH_SIZE]
			batch_y = train_y[i:i + BATCH_SIZE]

			# x = torch.Tensor(training_data[i][0]).view(-1, 1, 320, 160) # imgs = [0] is image, [1] is label, EACH index is 1 image + label array.
			# y = torch.Tensor([training_data[i][1]])

			optimizer.zero_grad()

			outputs = net(batch_x)
			
			loss = loss_function(outputs, batch_y)

			message = f"outputs: {outputs}, loss: {loss}\n"
			print(message)

			loss.backward()
			optimizer.step()

	torch.save(net, os.path.join(MODEL_SAVE_PATH, 'med-data-batch8-model.pth'))

def test(model, testing_data):
	x = torch.Tensor([i[0] for i in testing_data]).view(-1, 1, 320, 160) # all imgs in a tensor
	y = torch.Tensor([i[1] for i in testing_data]) # all labels

	val_size = int(len(x) * 0.10) # 10% of training data is validation

	val_x = x[-val_size:] # last 10% of imgs are validation data
	val_y = y[-val_size:] # "

	preds = []
	losses = []

	with open("test_preds.txt", "w") as f:
		for epoch in range(EPOCHS):
			print(epoch)
			for i in tqdm(range(0, len(val_x), BATCH_SIZE)):
				batch_x = val_x[i:i + BATCH_SIZE]
				batch_y = val_y[i:i + BATCH_SIZE]

				outputs = model(batch_x)
				loss = loss_function(outputs, batch_y)
				
				arr = outputs.detach().numpy()
				sum = 0
				for i in range(len(arr)):
					sum += arr[i][0]
				avg_output = sum / len(arr)

				message = f"outputs: {avg_output}, loss: {loss}\n"
				print(message)
				f.write(message)
			
if __name__ == "__main__":
	get_training_data()
	# t_data = np.load("training.npy", allow_pickle=True)

	# plt.imshow(training_data[12][0], cmap="gray")
	# plt.show()
	
	# train(t_data)

	# model = torch.load(MODEL_SAVE_PATH+"/med-data-batch8-model.pth")
	# test(model, t_data)
	# print(model)

	# with open("data/train.txt") as speed_file:
	# 	speeds = speed_file.readlines()

	# print(len(speeds))


	
