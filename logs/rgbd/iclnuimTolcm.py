from lcm import EventLog 
from lcm import Event
from eflcm.Frame import Frame

from PIL import Image
from PIL import ImageOps

import numpy as np 

import os, sys, getopt, datetime, re, string
from collections import defaultdict

import cv2 

base_name = 'iclnuim.lcmlog.'
extension = ".lcm"

def file_namer():
	date = datetime.date.today().strftime("%m.%d.%Y.")	
	name = base_name + date
	count = str(len([f for f in os.listdir("./") if os.path.isfile(f) and f.startswith(name)]))
	name += count + extension
	return name 

def main(argv):

	iclnuim_png_dir = ''
	lcm_channel = ''
	trackOnly = False
	path = file_namer()

	lcm_log = EventLog(path, 'w', True)

	try:
		opts, args = getopt.getopt(argv,"hd:c:t",["directory=", "channel=", "trackOnly"])
	except getopt.GetoptError:
		print 'iclnuimTolcm.py -d <iclnum_png_directory> -c <lcm_channel> -t'
		sys.exit(2)

	for opt, arg in opts:
		if opt == '-h':
			print 'iclnuimTolcm.py -d <iclnum_png_directory> -c <lcm_channel> -t'
			sys.exit()
		if opt in ("-d", "--directory"):
			iclnum_png_directory = arg
		if opt in ("-c", "--channel"):
			lcm_channel = arg
		if opt in ("-t", "--trackOnly"):
			trackOnly = True

	filenames = [f for f in os.listdir(iclnum_png_directory)]
	
	file_map = defaultdict(list)
	for f in filenames:
		parts = re.split(r'[_\.]+', f)

		if parts[3] in ('png', 'depth'):
			file_map[int(parts[2])].append(f)

	print 'writing to file: {}'.format(path)

	indexes = sorted(file_map)

	for i in indexes:
		print 'writing frame: {}'.format(i)
		files = sorted(file_map[i])
		image = Image.open(os.path.join(iclnum_png_directory, files[1]))
		image = image.convert('RGB')
		depth = np.loadtxt(os.path.join(iclnum_png_directory, files[0])) #* 1000 
		depth = depth.astype(np.dtype("uint16"))

		f = Frame()
		f.trackOnly = trackOnly
		f.compressed = False
		f.last = i == indexes[-1]
		f.depthSize = 640 * 480 * 2
		f.imageSize = 640 * 480 * 3
		f.depth = np.array(depth).flatten()
		f.image = np.array(image).flatten()
		f.timestamp = i# * 33000000
		f.frameNumber = i
		f.senderName = path
		
		e = Event(i, i * 33000, lcm_channel, f.encode())#f.timestamp / 1000
		
		lcm_log.write_event(e.timestamp, e.channel, e.data)

	lcm_log.close()

if __name__ == '__main__':
	main(sys.argv[1:])