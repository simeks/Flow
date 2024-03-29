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

@node_func('Matrix3x1', 'Image')
def ImageMatrix3x1(A0, B0, C0):
    da0 = A0.to_array()
    db0 = B0.to_array()
    dc0 = C0.to_array()
    
    data = np.append(da0, db0, 1)
    data = np.append(data, dc0, 1)
    return flow.Image(data, A0.pixel_type())

@numpy_func('Add', 'Image/Math')
def add(A, B):
    return A + B

@numpy_func('Log10', 'Image/Math')
def log10(In):
    return np.log10(In)

@numpy_func('Less', 'Image/Logic')
def less(In, S):
    return np.less(In, S).astype('uint8')

@numpy_func('Greater', 'Image/Logic')
def greater(In, S):
    return np.greater(In, S).astype('uint8')

@numpy_func('GreaterEqual', 'Image/Logic')
def greater_equal(In, S):
    return np.greater_equal(In, S).astype('uint8')

@numpy_func('FillMask', 'Image')
def fill_mask(In, Mask, Value):
    return np.ma.array(In, mask=Mask).filled(Value)

@numpy_func('InvertMask', 'Image')
def invert_mask(In, Mask):
    return numpy.ma.masked_array(In, numpy.logical_not(Mask))

@numpy_func('FillMaskColor', 'Image')
def fill_mask_color(In, Mask, Value = (0, 255, 0, 255)):
    Mask = Mask.reshape((np.product(Mask.shape), 1))
    Mask = np.hstack((Mask, Mask, Mask, Mask))
    if type(Value) == str:
        Value = Value.split(',')
        Value = map(int, Value)

    return flow.Image(np.ma.array(In, mask=Mask).filled(Value), 'vec4u8')

# Multiplies either two images of the same size or an image with a scalar.
#def multiply(A, B):
#    if type(A) == flow.Image and type(B) == flow.Image:
#        return flow.Image(np.



