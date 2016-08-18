import flow
import numpy as np

class TransposeImage(flow.Node):
    title = 'Transpose'
    category = 'Image'
    
    def __init__(self, other=None):
        super(TransposeImage, self).__init__(other)
        if other == None:
            self.add_pin('In', 'in')
            self.add_pin('Axes', 'in')
            self.add_pin('Out', 'out')

    def run(self, ctx):
        img = ctx.read_pin('In')
        axes = ctx.read_pin('Axes')
        
        if type(axes) == str:
            # Expecting a string of format 'x, y, z'.
            # TODO: Should python-style tuples and lists ('(...)', '[...]') be supported?
            axes = axes.split(',')
            axes = map(int, axes)

        if type(img) == flow.Image:
            data = img.to_array()

            data = np.transpose(data, axes)

            result = flow.Image(data, img.pixel_type())
            ctx.write_pin('Out', result)

class ImageMatrix1x2(flow.Node):
    title = 'Matrix1x2'
    category = 'Image'
    
    def __init__(self, other=None):
        super(ImageMatrix1x2, self).__init__(other)
        if other == None:
            self.add_pin('A0', 'in')
            self.add_pin('A1', 'in')
            self.add_pin('Out', 'out')

    def run(self, ctx):
        a0 = ctx.read_pin('A0')
        a1 = ctx.read_pin('A1')
        
        if type(a0) == flow.Image and type(a1) == flow.Image:
            if a0.pixel_type() != a1.pixel_type():
                return

            da0 = a0.to_array()
            da1 = a1.to_array()
            
            data = np.append(da0, da1, 0)

            result = flow.Image(data, a0.pixel_type())
            ctx.write_pin('Out', result)





def install_module():
    flow.install_template(TransposeImage())
    flow.install_template(ImageMatrix1x2())


