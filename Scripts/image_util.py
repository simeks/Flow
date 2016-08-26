import flow
import numpy as np
from function_node import numpy_func, node_func

@node_func('Transpose', 'Image')
def TransposeImage(In, Axes):
    if type(Axes) == str:
        # Expecting a string of format 'x, y, z'.
        # TODO: Should python-style tuples and lists ('(...)', '[...]') be supported?
        Axes = Axes.split(',')
        Axes = map(int, Axes)

    data = np.transpose(In.to_array(), Axes)
    return flow.Image(data, In.pixel_type())

@node_func('Matrix1x2', 'Image')
def ImageMatrix1x2(A0, A1):
    da0 = A0.to_array()
    da1 = A1.to_array()
    
    data = np.append(da0, da1, 0)
    return flow.Image(data, A0.pixel_type())

@numpy_func('Log10', 'Image/Math')
def log10(In):
    return np.log10(In)

@numpy_func('Less', 'Image/Logic')
def less(In, S):
    return np.less(In, S).astype('uint8')

@numpy_func('Greater', 'Image/Logic')
def greater(In, S):
    return np.greater(In, S).astype('uint8')

@numpy_func('FillMask', 'Image')
def fill_mask(In, Mask, Value):
    return np.ma.array(In, mask=Mask).filled(Value)

# Multiplies either two images of the same size or an image with a scalar.
#def multiply(A, B):
#    if type(A) == flow.Image and type(B) == flow.Image:
#        return flow.Image(np.



