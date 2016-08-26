import flow
import numpy as np
from function_node import numpy_func, node_func

# Normalizes an image to the range [0, 255]
def normalize_image(img):
    min = img.min()
    max = img.max()

    return 255.0 * (img - min) / (max - min)

@numpy_func('ColorPair', 'Visualization')
def ColorPair(A, B):
        # Normalize the images [0, 255]
        an = normalize_image(A).astype('uint8')
        bn = normalize_image(B).astype('uint8')

        s = an.size
        r = an.reshape((s, 1))
        g = np.minimum(an, bn).reshape((s, 1))
        b = bn.reshape((s, 1))
        a = np.ones((s, 1)) * 255

        out = np.hstack((r, g, b, a))
        out = out.reshape(list(A.shape) + [4])
        
        return flow.Image(out.astype('uint8'), 'vec4u8')

