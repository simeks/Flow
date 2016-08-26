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


def node_func(title, category = ''):
    def dec(fn):
        flow.install_template(FunctionNode(func=fn, args=inspect.getargspec(fn).args, title=title, category=category))
        return fn
    return dec

def numpy_func(title, category = ''):
    def dec(fn):
        def dec2(*args):
            args2 = []
            for a in args:
                if type(a) == flow.Image:
                    args2.append(a.to_array())
                else:
                    args2.append(a)
            arr = fn(*args2)
            return flow.Image(arr, str(arr.dtype)) # TODO: Conversion dtype -> pixel_type

        dec2.__name__ = fn.__name__
        dec2.__module__ = fn.__module__
        args = inspect.getargspec(fn).args
        flow.install_template(FunctionNode(func=dec2, args=args, title=title, category=category))
        return fn
    return dec

#     class dec(object):
#     def __init__(self, title, category):
#         self.title = title
#         self.category = category
#         self.fn = fn
#         self.args = inspect.getargspec(fn).args
#         self.name = fn.__name__

#     def __call__(self, *args):
#         return self.fn(*args)

# class numpy_func(node_func):
#     def __init__(self, fn):
#         funcs.append(fn.__name__)
#         super(numpy_func, self).__init__(fn)

#     def __call__(self, *args):
#         args2 = []
#         for a in args:
#             if type(a) == flow.Image:
#                 args2.append(a.to_array())
#             else:
#                 args2.append(a)
#         arr = self.fn(*args2)
#         return flow.Image(arr, str(arr.dtype)) # TODO: Conversion dtype -> pixel_type


