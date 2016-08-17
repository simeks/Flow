import flow
import numpy as np

# Normalizes an image to the range [0, 255]
def normalize_image(img):
    min = img.min()
    max = img.max()

    return 255.0 * (img - min) / (max - min)


class ColorPair(flow.Node):
    title = 'ColorPair'
    category = 'Visualization'
    
    def __init__(self, other=None):
        super(ColorPair, self).__init__(other)
        if other == None:
            self.add_pin('A', 'in')
            self.add_pin('B', 'in')
            self.add_pin('Out', 'out')

    def run(self, ctx):
        img_a = ctx.read_pin('A')
        img_b = ctx.read_pin('B')
        if type(img_a) == flow.Image and type(img_b) == flow.Image:
            a_data = img_a.to_array()
            b_data = img_b.to_array()

            # Normalize the images [0, 255]
            an = normalize_image(a_data).astype('uint8')
            bn = normalize_image(b_data).astype('uint8')

            s = an.size
            r = an.reshape((s, 1))
            g = np.minimum(an, bn).reshape((s, 1))
            b = bn.reshape((s, 1))
            a = np.ones((s, 1)) * 255

            out = np.hstack((r, g, b, a))
            print out.shape
            out = out.reshape(list(a_data.shape[::-1]) + [4])
            print out.shape
            
            result = flow.Image(out.astype('uint8'), 'vec4u8')
            result.set_origin(img_a.origin())
            result.set_spacing(img_a.spacing())
            ctx.write_pin('Out', result)


def install_module():
    flow.install_template(ColorPair())