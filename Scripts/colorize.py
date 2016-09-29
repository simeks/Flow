from scipy import ndimage
import flow
import numpy as np
from matplotlib import cm
from function_node import numpy_func, node_func


@node_func('GrayScale', 'Visualization/ColorMap')
def GrayScale(In):
	# Output is a 0-255 grayscale image
	# If all values are positive:
	#	Map 0 to black and largest to white
	# If image consists of both positive and negative values:
	#	Map zero to 127 and scale accordingly

	data = In.to_array()
	dmin = data.min()
	dmax = data.max()

	if dmin >= 0 and dmax >= 0:
		out = 255 * data / dmax
	else:
		out = 126 + 126 * data / max(-dmin, dmax)

	return flow.Image(out.astype('uint8'), 'uint8')

@node_func('Jet', 'Visualization/ColorMap')
def Jet(In, Min, Max):
	data = In.to_array()
	
	out = cm.jet((data - Min) / float(Max - Min)) * 255
	
	return flow.Image(out.astype('uint8'), 'vec4u8')

@node_func('DivergingScale', 'Visualization/ColorMap')
def DivergingScale(In):
	data = In.to_array()
	dmin = data.min()
	dmax = data.max()

	out = cm.bwr(((data / dmax) + 1.0)*0.5) * 255

	return flow.Image(out.astype('uint8'), 'vec4u8')
