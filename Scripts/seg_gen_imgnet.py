import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab
from sklearn.preprocessing import normalize
caffe_root = '/home/zshen5/GitHub/SegNet-Tutorial/caffe-segnet/' 			# Change this to the absolute directoy to SegNet Caffe
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()

caffe.set_mode_gpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

print 'inilization done'
for i in range(0, args.iter):

	net.forward()
	image = net.blobs['data'].data
	label = net.blobs['label'].data
	predicted = net.blobs['prob'].data
	image = np.squeeze(image[0,:,:,:])
	output = np.squeeze(predicted[0,:,:,:])
	ind = np.argmax(output, axis=0)

	r = ind.copy()
	g = ind.copy()
	b = ind.copy()
	r_gt = label.copy()
	g_gt = label.copy()
	b_gt = label.copy()
	
	wall = [139,181,248]
	building = [251,209,244]
	sky = [44,230,121]
	floor = [156,40,149]
	tree = [166,219,98]
	ceiling = [35,229,138]
	road = [143,56,194]
	bed  = [144,223,70]
	windowpane = [200,162,57]
	grass = [120,225,199]
	cabinet = [87,203,13]
	sidewalk = [185,1,136]
	person = [16,167,16]
	earth = [29,249,241]
	door = [17,192,40]
	table = [199,44,241]
	mountain = [193,196,159]
	plant = [241,172,78]
	curtain = [56,94,128]
	chair = [231,166,116]
	car = [50,209,252]
	water = [217,56,227]
	painting = [168,198,178]
	sofa = [77,179,188]
	shelf = [236,191,103]
	house = [248,138,151]
	sea = [214,251,89]
	mirror = [208,204,187]
	rug = [115,104,49]
	field = [29,202,113]
	armchair = [159,160,95]
	seat = [78,188,13]
	fence = [83,203,82]
	desk = [8,234,116]
	rock = [80,159,200]
	wardrobe = [124,194,2]
	lamp = [192,146,237]
	bathtub = [64,3,73]
	railing = [17,213,58]
	cushion = [106,54,105]
	base = [125,72,155]
	box = [202,36,231]
	column = [79,144,4]
	signboard = [118,185,128]
	chest = [138,61,178]
	counter = [23,182,182]
	sand = [154,114,4]
	sink = [201,0,83]
	skyscraper = [21,134,53]
	fireplace = [194,77,237]
	refrigerator = [198,81,106]
	grandstand = [37,222,181]
	path = [203,185,14]
	stairs = [134,140,113]
	runway = [220,196,79]
	case = [64,26,68]
	pooltable = [128,89,2]
	pillow = [199,228,65]
	screen = [62,215,111]
	stairway = [124,148,166]
	river = [221,119,245]
	bridge = [68,57,158]
	bookcase = [80,47,26]
	blind = [143,59,56]
	coffeetable = [14,80,215]
	toilet = [212,132,31]
	flower = [2,234,129]
	book = [134,179,44]
	hill = [53,21,129]
	bench = [80,176,236]
	countertop = [154,39,168]
	stove = [221,44,139]
	palm = [103,56,185]
	kitchenisland = [224,138,83]
	computer = [243,93,235]
	swivelchair = [80,158,63]
	boat = [81,229,38]
	bar = [116,215,38]
	arcademachine = [103,69,182]
	hovel = [66,81,5]
	bus = [96,157,229]
	towel = [164,49,170]
	light = [14,42,146]
	truck = [164,67,44]
	tower = [108,116,151]
	chandelier = [144,8,144]
	awning = [85,68,228]
	streetlight = [16,236,72]
	booth = [108,7,86]
	television = [172,27,94]
	airplane = [119,247,193]
	dirttrack = [155,240,152]
	apparel = [49,158,204]
	pole = [23,193,204]
	land = [228,66,107]
	bannister = [69,36,163]
	escalator = [238,158,228]
	ottoman = [202,226,35]
	bottle = [194,243,151]
	buffet = [192,56,76]
	poster = [16,115,240]
	stage = [61,190,185]
	van = [7,134,32]
	ship = [192,87,171]
	fountain = [45,11,254]
	conveyerbelt = [179,183,31]
	canopy = [181,175,146]
	washer = [13,187,133]
	plaything = [12,1,2]
	swimmingpool = [63,199,190]
	stool = [221,248,32]
	barrel = [183,221,51]
	basket = [90,111,162]
	waterfall = [82,0,6]
	tent = [40,0,239]
	bag = [252,81,54]
	minibike = [110,245,152]
	cradle = [0,187,93]
	oven = [163,154,153]
	ball = [134,66,99]
	food = [123,150,242]
	step = [38,144,137]
	tank = [59,180,230]
	tradename = [144,212,16]
	microwave = [132,125,200]
	pot = [26,3,35]
	animal = [199,56,92]
	bicycle = [83,223,224]
	lake = [203,47,137]
	dishwasher = [74,74,251]
	screen = [246,81,197]
	blanket = [168,130,178]
	sculpture = [136,85,200]
	hood = [186,147,103]
	sconce = [170,21,85]
	vase = [104,52,182]
	trafficlight = [166,147,202]
	tray = [103,119,71]
	ashcan = [74,161,165]
	fan = [14,9,83]
	pier = [129,194,43]
	crtscreen = [7,100,55]
	plate = [13,12,170]
	monitor = [30,21,22]
	bulletinboard = [224,189,139]
	shower = [40,77,25]
	radiator = [194,14,94]
	glass = [178,8,231]
	clock = [234,166,8]
	flag = [248,25,7]	
	unlabelled = [0,0,0]

	label_colours = np.array([wall,building,sky,floor,tree,ceiling,road,bed ,windowpane,grass,cabinet,sidewalk,person,earth,door,table,mountain,plant,curtain,chair,car,water,painting,sofa,shelf,house,sea,mirror,rug,field,armchair,seat,fence,desk,rock,wardrobe,lamp,bathtub,railing,cushion,base,box,column,signboard,chest,counter,sand,sink,skyscraper,fireplace,refrigerator,grandstand,path,stairs,runway,case,pooltable,pillow,screen,stairway,river,bridge,bookcase,blind,coffeetable,toilet,flower,book,hill,bench,countertop,stove,palm,kitchenisland,computer,swivelchair,boat,bar,arcademachine,hovel,bus,towel,light,truck,tower,chandelier,awning,streetlight,booth,television,airplane,dirttrack,apparel,pole,land,bannister,escalator,ottoman,bottle,buffet,poster,stage,van,ship,fountain,conveyerbelt,canopy,washer,plaything,swimmingpool,stool,barrel,basket,waterfall,tent,bag,minibike,cradle,oven,ball,food,step,tank,tradename,microwave,pot,animal,bicycle,lake,dishwasher,screen,blanket,sculpture,hood,sconce,vase,trafficlight,tray,ashcan,fan,pier,crtscreen,plate,monitor,bulletinboard,shower,radiator,glass,clock,flag,unlabelled])
	for l in range(0,150):
		r[ind==l] = label_colours[l,0]
		g[ind==l] = label_colours[l,1]
		b[ind==l] = label_colours[l,2]
		r_gt[label==l] = label_colours[l,0]
		g_gt[label==l] = label_colours[l,1]
		b_gt[label==l] = label_colours[l,2]

	rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
	rgb[:,:,0] = r/255.0
	rgb[:,:,1] = g/255.0
	rgb[:,:,2] = b/255.0
	rgb_gt = np.zeros((ind.shape[0], ind.shape[1], 3))
	rgb_gt[:,:,0] = r_gt/255.0
	rgb_gt[:,:,1] = g_gt/255.0
	rgb_gt[:,:,2] = b_gt/255.0

	image = image/255.0

	image = np.transpose(image, (1,2,0))
	output = np.transpose(output, (1,2,0))
	image = image[:,:,(2,1,0)]                    
	scipy.misc.toimage(rgb,cmin=0.0,cmax=255).save('/home/zshen5/Data/ImageNet2016/ADEChallengeData2016/predictions_camvid38kv2/rgb/ADE_val_0000'+ str('%04d' %(i+1)) +'.png')
	scipy.misc.toimage(ind,cmin=0.0,cmax=150).save('/home/zshen5/Data/ImageNet2016/ADEChallengeData2016/predictions_camvid38kv2/img/ADE_val_0000'+ str('%04d'%(i+1)) +'.png')

	# plt.figure()
	# plt.imshow(image,vmin=0, vmax=1)
	# plt.figure()
	# plt.imshow(rgb_gt,vmin=0, vmax=1)
	# plt.figure()
	# plt.imshow(rgb,vmin=0, vmax=1)
	# plt.show()


print 'Success!'

