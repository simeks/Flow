from scipy import ndimage
import flow
import numpy as np

def create_water_mask(wat_fixed_img, bone_mask):
    # Just create a mask and use the deformation as constraints
    # Erode water images
    wat_fixed_eroded_img = ndimage.binary_erosion(np.greater_equal(wat_fixed_img, 500), np.ones([7,3,7]))

    # Include bone mask
    if bone_mask:
    	wat_fixed_eroded_img = np.logical_or(wat_fixed_eroded_img, bone_mask)

    return wat_fixed_eroded_img.astype('uint8')



class WaterMask(flow.Node):
	title = 'WaterConstraintMask'
	category = 'Imiomics'

	def __init__(self, other=None):
		super(WaterMask, self).__init__(other)
		if other == None:
			self.add_pin('FixedWater', 'in')
			self.add_pin('BoneMask', 'in')
			self.add_pin('OutFile', 'in')
			self.add_pin('WaterMask', 'out')


	def read_fixed_water(self, ctx):
		fixed_water = ctx.read_pin('FixedWater')
		if type(fixed_water) == str:
			img = flow.Image()
			img.load_from_file(fixed_water)
			return img
		elif type(fixed_water) == flow.Image:
			return fixed_water
		else:
			return None

	def read_bone_mask(self, ctx):
		bone_mask = ctx.read_pin('BoneMask')
		if type(bone_mask) == str:
			img = flow.Image()
			img.load_from_file(bone_mask)
			return img
		elif type(bone_mask) == flow.Image:
			return bone_mask
		else:
			return None

	def run(self, ctx):
		if self.is_pin_linked('FixedWater'):
			fixed_water = self.read_fixed_water(ctx)
			bone_mask = self.read_bone_mask(ctx)
			if bone_mask:
				water_mask = create_water_mask(fixed_water.to_array(), bone_mask.to_array())
			else:
				water_mask = create_water_mask(fixed_water.to_array(), None)

			result = flow.Image(water_mask, 'uint8')
			result.set_origin(fixed_water.origin())
			result.set_spacing(fixed_water.spacing())

			if self.is_pin_linked('OutFile'):
				out_file = ctx.read_pin('OutFile')
				if type(out_file) == str:
					# Save to file
					result.save_to_file(out_file)

			if self.is_pin_linked('WaterMask'):
				ctx.write_pin('WaterMask', result)

		
def install_module():
	flow.install_template(WaterMask())
