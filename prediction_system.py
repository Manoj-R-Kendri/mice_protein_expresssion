
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

loaded_model=pickle.load(open('trained_model3.sav'))

input_data=(0.447506385,	0.62817583	,0.36738809	,2.38593897,	4.807635435,	0.218577766,	0.176233365	,2.14128243	,0.195187525	,1.442398172	,0.566339562	,0.289823901	,0.363892996	,0.26683694,	0.85912085,	0.521306627	,1.538244388	,1.968275306,	0.495899987	,0.672402205,	0.36940449	,0.357171663,	0.179728458,	1.227449926	,2.956983466	,1.447909665	,0.250840167,	0.284043554,	0.704395752,	0.156875924	,0.391047184	,2.467132679	,0.327597795,	0.404489851,	0.296276381	,0.674418605	,0.539723081,	0.354214276	,0.51431644	,0.347224089	,0.303132141,	0.412824304	,0.382578304	,0.162330317,	0.77969457,	0.186792986	,1.634615385,	0.28803733,	0.332367081	,1.12344457	,0.175692873	,0.150593891	,0.183823529,	0.106476244	,0.13956448,	0.174844457	,0.130514706	,0.115243213	,0.236849548	,0.13645362	,0.478577489	,0.244485294	,1.507777149,	2.003535068	,0.120687217,	0.920178167	,0.843679299,	0.190469457,	0.131575226	,	0.106476244	,0.109445701,	0.439833145	,0.11665724,	0.140766403	,0.14218043,	1.816388575,	0,0,0,0)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction =loaded_model.predict(input_data_reshaped)
print(prediction)