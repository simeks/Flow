from scipy import ndimage
import flow
import numpy as np
from matplotlib import cm
from function_node import numpy_func, node_func

def grid_img(w, h, step, thickness):
    img_data = np.zeros((h, w)).astype('uint8')
    
    for y in range(0, int(h / step)+1):
        for x in range(0, w):
            for t in range(0, min(thickness, h - y*step - 1)): 
                img_data[y*step+t,x] = 1
    for x in range(0, int(w / step)+1):
        for y in range(0, h):
            for t in range(0, min(thickness, w - x*step - 1)): 
                img_data[y,x*step+t] = 1
                
    img = flow.Image(img_data, 'uint8')
    return img

@node_func('DeformationColor', 'Visualization')
def CreateGrid2D(Size, Thickness=4, Step=10):
    return grid_img(Size[0], Size[1], Step, Thickness)

@numpy_func('ColorGrid2D', 'Visualization')
def ColorGrid2D(Grid, ColorMap):
    size = Grid.shape[::-1]
    
    Grid = Grid.reshape(np.product(size), 1)
    Grid = np.hstack((Grid, Grid, Grid, Grid))

    ColorMap = ColorMap.reshape(np.product(size), 4)
    ColorMap = ColorMap * Grid
    ColorMap = ColorMap.reshape([size[1], size[0], 4])
    return flow.Image(ColorMap, 'vec4f')

@node_func('DeformationColor', 'Visualization')
def DeformationColor(Deformation, ZSlice):
    size = Deformation.size()
    data = Deformation.to_array()[ZSlice]

    data = data.reshape((np.product(size[:2]), 3))
    x, y, z = np.hsplit(data, 3)

    # Determine magnitude in world coordinates
    spacing = Deformation.spacing()
    x = x * spacing[0]
    y = y * spacing[1]
    z = z * spacing[2]
    m = np.sqrt(x*x + y*y + z*z)
    
    # Normalize and color
    max = 100 # np.max(m)
    m = cm.jet(m / max)
    m = m.reshape((size[1], size[0], 4))

    return flow.Image(m.astype('float32'), 'vec4f')
