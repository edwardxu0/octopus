import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import numpy as np
import copy
import csv
import os
import time
import argparse
import pickle
import DropNet.models

from DropNet.mask import *



def train(args,train_loader,num_of_class,mask=None):
	###Function for training a network
	if mask==None:
		model, mask, activation=initialize_model(args.network, num_of_class, args.layers)
	else:
		model, mask, activation=initialize_model(args.network, num_of_class, args.layers,mask)
	cross_el = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) #e-1
	for ep in range(args.epochs):
		#print(ep)
		model.train()
		for data in train_loader:
		    x, y = data
		    optimizer.zero_grad()
		    if args.network=='Mnist_fc':
		        outputs = model(x.view(x.size(0), -1))
		    else:
		        outputs = model(x)
		    #loss=F.nll_loss(outputs, y)
		    loss = cross_el(outputs, y)
		    loss.backward()
		    optimizer.step()
	return model, mask, activation

def test(args, model,data_loader):
	##Function for testing a network
	correct=0
	total=0
	#count=0
	with torch.no_grad():
		for data in data_loader:
		    x, y = data
		    if args.network=='Mnist_fc':
		        outputs = model(x.view(x.size(0), -1))
		    else:
		        outputs = model(x)
		    for idx, i in enumerate(outputs):
		        if torch.argmax(i) == y[idx]:
		            correct +=1
		        total +=1
		    # pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
		    # correct += pred.eq(y.view_as(pred)).sum().item()
	#print(" correct= ",correct, "total= ",total)
	accuracy= round(correct/total, 3)
	return accuracy


def make_param_matrix(model, mask):
	'''
	Function for making weight matrix for reconstructed model
	input:model: from which reconstructed model will be made
		  mask: matrix containing 0 for pruned and 1 for unpruned neurons of the model
	output: weight matrix: containing weights of all unpruned neurons
		 	bias matrix: containing bias of all unpruned neurons
	'''
	weight=[]
	bias = []

	for i in mask:
		lst_w = []
		lst_b = []
		

		for j in range(len(mask[i])):
			for name, param in model.named_parameters():
				if name == f"{i}.weight" and mask[i][j]!=0:
					lst_w.append(param[j].reshape(1,-1))
					
				elif name == f"{i}.bias" and mask[i][j]!=0:
					lst_b.append(param[j])


		w=torch.cat(lst_w,dim=0)
		b=torch.tensor(lst_b)
		weight.append(w)
		bias.append(b)

	for name, param in model.named_parameters():
			if "fcout" in name and "weight" in name:
				weight.append(param)
			elif "fcout" in name and "bias" in name:
				bias.append(param)


	final_weight=[]
	final_weight.append(weight[0])

	l=1
	keys=[k for k in mask.keys()]
	while l<=len(mask):
		k=keys[l-1]
		lst_w = []
		for i in range(len(weight[l])):
			temp=[]
			t=weight[l][i]
			for j in range(len(mask[k])):
				if mask[k][j]!=0:
					temp.append(t[j])
			temp = torch.tensor(temp)
			lst_w.append(temp.reshape(1,-1))

		w = torch.cat(lst_w,dim=0)
		final_weight.append(w)

		l=l+1


	return final_weight, bias


def make_new_model(model, mask):
	'''
	function for making reconstructed model
	input:model: from which reconstructed model will be made
		  mask: matrix containing 0 for pruned and 1 for unpruned neurons of the model
	output: reconstructed model
	'''
	model.eval()

	weight, bias = make_param_matrix(model, mask)

	layers = [int(sum(mask[i])) for i in mask]
	

	network=md.Mnist_fc(layers,10)
	network.eval()
	
	with torch.no_grad():
		i=0
		w=0
		b=0

		for name, param in network.named_parameters():
			if "weight" in name:
				for j in range(len(weight[i])):
					#print("weight shape: ", weight[i][j].shape)
					param[j].copy_(weight[i][j])
					
					#print("param shape: ",param[j].shape)
				w=1
			elif "bias" in name:
				param.copy_(bias[i])
				#param=bias[i]
				#print("bias shape: ", bias[i].shape)
				#print("param shape: ",param.shape)
				b=1

			if w==1 and b==1:
				i=i+1
				w=0
				b=0

	return network




def prune( args):
	'''
	Function for pruning and generating new as well as reconstruced models
	'''
	if args.dataset == 'mnist':
		transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

		mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
		train_set, val_set = torch.utils.data.random_split(mnist_trainset, [50000, 10000])
		train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
		val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True)


		mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
		test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=args.test_batch_size, shuffle=True)


	#epoch = args.epochs
	pr_ratio=args.pruning_ratio
	num_of_class=10

	

	
	
	#count=0
	model, mask, activation=train(args,train_loader,num_of_class)

	up_activation = copy.deepcopy(activation)

	train_accuracy=test(args, model,train_loader)

	val_accuracy = test(args, model,val_loader)

	test_accuracy = test(args, model,test_loader)



	percent=percent_mask(mask)
	print("percentage remaining: ", percent)
	print("Sparsity: ", (1-percent))
	print('Layer nodes:', [torch.sum(mask[i]) for i in mask.keys()])
	print('main train accuracy:', train_accuracy)
	print('main validation accuracy:', test_accuracy)

	fields = ["pruning rate", "remaining frac", "sparsity","train accuracy", "test accuracy", "test accuracy of completely neuron removed network","accuracy drop from original model's test accuracy(%)"]
	layer=[float(torch.sum(mask[i])) for i in mask.keys()]
	for i in range(len(layer)):
		fields.append(f"Layer f{i+1}")

	value=[0, float(percent),(1-float(percent)), float(train_accuracy), float(test_accuracy), 0, 0]
	value.extend(layer)
	# name of csv file 
	strg="DropNet_"+args.name
	path=f"./result/{strg}"
	os.makedirs(path)
	os.mkdir(f"{path}/activation")
	os.mkdir(f"{path}/mask")
	#os.mkdir(f"{path}/model")
	filename = f"./result/{strg}/{strg}_data.csv"

	# writing to csv file 
	with open(filename, 'w') as csvfile: 
		# creating a csv writer object 
		csvwriter = csv.writer(csvfile)
		csvwriter.writerow(fields)
		csvwriter.writerow(value)

	#save the original network

	dummy_input = torch.randn(1, 1, 28, 28)
	torch.onnx.export(model, dummy_input, f'{path}/original.onnx', verbose=True)
	#torch.save(model,f"{path}/model/original.pth")
	pickle.dump(up_activation, open(f'{path}/activation/original.pkl','wb'))


	#Produce pruned network

	prev_percent=0
	percent_count=0

	while percent>0.1 and pr_ratio > 0.01:
		activation = copy.deepcopy(up_activation)
		prev_activation= copy.deepcopy(up_activation)

		prev_mask=copy.deepcopy(mask)

		mask=update_mask(mask, activation, pr_ratio)
		model, mask, activation=train(args,train_loader,num_of_class,mask)

		#print("checkpoint 2")   

		up_activation = copy.deepcopy(activation)

		#print("correct= ",correct, "total= ",total)
		t_accuracy=test(args, model,train_loader)

		v_accuracy = test(args, model,val_loader)

		accuracy = test(args, model,test_loader)

		network = make_new_model(model, mask)
		r_accuracy = test(args, network,test_loader)

		#print("checkpoint 3")

		percent=percent_mask(mask)
		print("percentage remaining: ", percent)
		print("Sparsity: ", (1-percent))
		print('Layer nodes:', [torch.sum(mask[i]) for i in mask.keys()])
		print('train accuracy:', t_accuracy)
		print('test accuracy:', accuracy)
		print("test accuracy of reduced network: ",r_accuracy)
		print('accuracy drop: ', ((test_accuracy-accuracy)/test_accuracy)*100)
		acc_drop=((test_accuracy-accuracy)/test_accuracy)*100

		layer=[float(torch.sum(mask[i])) for i in mask.keys()]
		value=[pr_ratio, float(percent),(1-float(percent)), float(t_accuracy), float(accuracy), float(r_accuracy), acc_drop]
		value.extend(layer)

		# writing to csv file 
		with open(filename, 'a') as csvfile: 
		    # creating a csv writer object 
		    csvwriter = csv.writer(csvfile)
		    csvwriter.writerow(value)

		if prev_percent==percent:
		    percent_count+=1
		    if percent_count==4:
		        break
		else:
		    prev_percent=percent
		    percent_count=0

		#print("final_Activation: ",activation)
		if (test_accuracy-accuracy)/test_accuracy>args.tolerance:
		    mask= copy.deepcopy(prev_mask)
		    up_activation= copy.deepcopy(prev_activation)
		    pr_ratio=pr_ratio*0.5
		    print("Accuracy drops more, retraiving....")
		    print(f'new pruning ratio is: {pr_ratio}')
		else:
			#save all parameters
		    p=float(1-percent)
		    p=int(p * 10000)
		    dummy_input = torch.randn(1, 1, 28, 28)
		    torch.onnx.export(model, dummy_input, f'{path}/{strg}_per_{p}.onnx', verbose=True)

		    
		    torch.onnx.export(network, dummy_input, f'{path}/{strg}_Reduced:per_{p}.onnx', verbose=True)
		    #torch.save(model,f"{path}/model/per_{p}.pth")
		    pickle.dump(up_activation, open(f'{path}/activation/per_{p}.pkl','wb'))
		    pickle.dump(mask, open(f'{path}/mask/per_{p}.pkl','wb'))
