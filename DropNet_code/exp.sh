#!/bin/bash

for i in {1..5} 
do
	python main.py --network=NetS --name=NetS_${i}
done

for i in {4..5} 
do
	python main.py --network=NetM --name=NetM_${i}
done

for i in {1..5} 
do
	python main.py --network=NetL --name=NetL_${i}
done