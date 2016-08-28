import flow
import numpy as np
import inspect

class FunctionNode(flow.Node):
    def __init__(self, other=None, func=None, args=None, title=None, category=None):
        super(FunctionNode, self).__init__(other)
        if other == None:
            if not hasattr(func, '__name__'):
                raise TypeError('Invalid function type, make sure you use the different @node_func decorators.')

            if args:
                self.args = args
            else:
                self.args = []

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
        self.class_name = self.func.__module__ + '.' + self.func.__name__

    def run(self, ctx):
        if self.func == None:
            return

        kwargs = {}
        for a in self.args:
            p = ctx.read_pin(a)
            if p == None:
                print '[Warning] Pin \'%s\' not set.' % a # TODO: Better msg/warning/error
            else:
                kwargs[a] = p

        result = self.func(**kwargs)

        # If our first argument and the result is an image, copy metadata
        in_img = kwargs[self.args[0]]
        if type(result) == flow.Image and type(in_img) == flow.Image:
            result.set_origin(in_img.origin())
            result.set_spacing(in_img.spacing())

        ctx.write_pin('Out', result)   


def node_func(title, category = ''):
    def dec(fn):
        flow.install_template(FunctionNode(func=fn, args=inspect.getargspec(fn).args, title=title, category=category))
        return fn
    return dec

def numpy_func(title, category = ''):
    def dec(fn):
        def dec2(**kwargs):
            kwargs2 = {}
            for k, v in kwargs.iteritems():
                if type(v) == flow.Image:
                    kwargs2[k] = v.to_array()
                else:
                    kwargs2[k] = v
            arr = fn(**kwargs2)
            if type(arr) == np.array or type(arr) == np.ndarray:
                return flow.Image(arr, str(arr.dtype))
            return arr

        dec2.__name__ = fn.__name__
        dec2.__module__ = fn.__module__
        args = inspect.getargspec(fn).args
        flow.install_template(FunctionNode(func=dec2, args=args, title=title, category=category))
        return fn
    return dec
