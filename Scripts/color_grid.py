from scipy import ndimage
import flow
import numpy as np
from matplotlib import cm

def grid_img(w, h, step, thickness):
    img_data = np.zeros((h, w)).astype('uint8')
    
    for y in range(0, int(h / step)+1):
        for x in range(0, w):
            for t in range(0, thickness): 
                img_data[y*step+t,x] = 1
    for x in range(0, int(w / step)+1):
        for y in range(0, h):
            for t in range(0, thickness): 
                img_data[y,x*step+t] = 1
                
    img = flow.Image(img_data, 'uint8')
    return img

class CreateGrid2D(flow.Node):
    title = 'CreateGrid2D'
    category = 'Visualization'

    def __init__(self, other=None):
        super(CreateGrid2D, self).__init__(other)
        if other == None:
            self.add_pin('Size', 'in')
            self.add_pin('Thickness', 'in')
            self.add_pin('Step', 'in')
            self.add_pin('Grid', 'out')

    def run(self, ctx):
        size = ctx.read_pin('Size')

        thickness = 4
        if self.is_pin_linked('Thickness'):
            thickness =  ctx.read_pin('Thickness')
        
        step = 10
        if self.is_pin_linked('Step'):
            step =  ctx.read_pin('Step')

        if type(size) == tuple:
            ctx.write_pin('Grid', grid_img(size[0], size[1], step, thickness))
            
class ColorGrid2D(flow.Node):
    title = 'ColorGrid2D'
    category = 'Visualization'

    def __init__(self, other=None):
        super(ColorGrid2D, self).__init__(other)
        if other == None:
            self.add_pin('Grid', 'in')
            self.add_pin('ColorMap', 'in')
            self.add_pin('Result', 'out')

    def run(self, ctx):
        grid = ctx.read_pin('Grid')
        color = ctx.read_pin('ColorMap')

        if type(grid) == flow.Image and type(color) == flow.Image:
            size = grid.size()

            grid_data = grid.to_array()
            color_data = color.to_array()

            grid_data = grid_data.reshape(np.product(size), 1)
            grid_data = np.hstack((grid_data, grid_data, grid_data, grid_data))
            
            color_data = color_data.reshape(np.product(size), 4)
            color_data = color_data * grid_data

            color_data = color_data.reshape([size[0], size[1], 4])
            result = flow.Image(color_data, 'vec4f')
            ctx.write_pin('Result', result)

class DeformationColor(flow.Node):
    title = 'DeformationColor'
    category = 'Visualization'

    def __init__(self, other=None):
        super(DeformationColor, self).__init__(other)
        if other == None:
            self.add_pin('Deformation', 'in')
            self.add_pin('ZSlice', 'in')
            self.add_pin('ColorMap', 'out')

    def run(self, ctx):
        df = ctx.read_pin('Deformation')
        z = ctx.read_pin('ZSlice')
        if type(df) == flow.Image and type(z) == int:
            size = df.size()
            data = df.to_array()[z]

            data = data.reshape((np.product(size[:2]), 3))
            x, y, z = np.hsplit(data, 3)

            # Determine magnitude in world coordinates
            spacing = df.spacing()
            x = x * spacing[0]
            y = y * spacing[1]
            z = z * spacing[2]
            m = np.sqrt(x*x + y*y + z*z)
            
            # Normalize and color
            max = np.max(m)
            m = cm.jet(m / max)
            m = m.reshape((size[0], size[1], 4))

            result = flow.Image(m.astype('float32'), 'vec4f')
            result.set_origin(df.origin())
            result.set_spacing(df.spacing())
            ctx.write_pin('ColorMap', result)






def install_module():
    flow.install_template(CreateGrid2D())
    flow.install_template(ColorGrid2D())
    flow.install_template(DeformationColor())
