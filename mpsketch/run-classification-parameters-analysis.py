#!/usr/bin/env python

import os

datasets = ['m10'];

iterations = [1,2,3,4,5]
ks = [100,150,250,300]
turns = [1,2,3,4,5]

method = 'mpsketch'
os.system("rm "+method);
os.system("g++ -std=c++11 -lm -O3 -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result "+method+".cpp -o "+method+" -lgsl -lm -lgslcblas");

for turn in turns:
	for i_data, data in enumerate(datasets):

		print(data)

		path = "../results/" + data + "/"
		folder = os.path.exists(path)
	 
		if not folder:          
			os.makedirs(path) 
	
		for k in ks:
			for iteration in iterations:
				os.system("./"+method+" -network ../data/" + data + "/network.adjlist " + \
					" -feature ../data/" + data + "/features.txt -hashdim " + str(k) + " -iteration " + str(iteration) + \
					" -embedding ../results/" + data + "/" + data + "." +method + ".iteration." +str(iteration) + ".k."+ str(k) + ".embedding.turn." +str(turn) + \
					" -time ../results/" + data + "/time."+ method+ ".iteration." + str(iteration) + ".k."+ str(k) + ".txt.turn." +str(turn));

