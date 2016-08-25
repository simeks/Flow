import flow
import numpy as np
import inspect

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


class FunctionNode(flow.Node):
    def __init__(self, other=None, func=None, title=None, category=None):
        super(FunctionNode, self).__init__(other)
        if other == None:
            self.args = inspect.getargspec(func).args
            for a in self.args:
                self.add_pin(a, 'in')
            self.add_pin('Out', 'out')
            self.func = func
            if title != None:
                self.title = title
            if category != None:
                self.category = category
        else:
            self.func = other.func
            self.title = other.title
            self.category = other.category
            self.args = other.args
        self.class_name = self.func.__name__

    def run(self, ctx):
        if self.func == None:
            return

        args = []
        for a in self.args:
            p = ctx.read_pin(a)
            if p == None:
                print '[Warning] Pin \'%s\' not set.' % a # TODO: Better msg/warning/error
            args.append(p)

        result = self.func(*args)
        if type(result) == flow.Image and type(args[0]) == flow.Image:
            result.set_origin(args[0].origin())
            result.set_spacing(args[0].spacing())
        ctx.write_pin('Out', result)   

def log10(In):
    return flow.Image(np.log10(In.to_array()), In.pixel_type())

def less(In, S):
    return flow.Image(np.less(In.to_array(), S).astype('uint8'), 'uint8')

def greater(In, S):
    return flow.Image(np.greater(In.to_array(), S).astype('uint8'), 'uint8')

def fill_mask(In, Mask, Value):
    return flow.Image(np.ma.array(In.to_array(), mask=Mask.to_array()).filled(Value), In.pixel_type())


def install_module():
    flow.install_template(TransposeImage())
    flow.install_template(ImageMatrix1x2())
    flow.install_template(FunctionNode(func=log10, title='Log10', category='Image/Math'))
    flow.install_template(FunctionNode(func=less, title='Less', category='Image/Logic'))
    flow.install_template(FunctionNode(func=greater, title='Greater', category='Image/Logic'))
    flow.install_template(FunctionNode(func=fill_mask, title='FillMask', category='Image'))


