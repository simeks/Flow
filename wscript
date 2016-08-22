APPNAME = 'Flow'
VERSION = '0.0.1'

CONFIGURATIONS = ['debug', 'release']
PLATFORMS = {
	'win32' : ['win64'],
	'linux' : ['linux_x64_gcc', 'linux_x64_clang']
}

SUBFOLDERS = ['Source']

top = '.'
out = 'build'

import os
from waflib.extras import msvs
from waflib import Utils
from waflib.TaskGen import extension, before_method, after_method, feature

def supported_platforms():
	p = Utils.unversioned_sys_platform()
	return PLATFORMS[p]

def platform_vs_to_waf(p):
	if platform == 'Win32':
		return 'win32'
	if platform == 'x64':
		return 'win64'

def configuration_vs_to_waf(p):
	if platform == 'Debug':
		return 'debug'
	if platform == 'Release':
		return 'release'

def configuration_waf_to_vs(p):
	if platform == 'debug':
		return 'Debug'
	if platform == 'release':
		return 'Release'


class vsnode_target(msvs.vsnode_target):
	def get_build_command(self, props):
		p = self.get_build_params(props)
		return "%s build_%s_%s %s" % (p[0], platform_vs_to_waf(props.platform), configuration_vs_to_waf(props.configuration), p[1])
		
class msvs_2013(msvs.msvs_generator):
	cmd = 'msvs2013'
	numver = '13.00'
	vsver = '2013'
	platform_toolset_ver = 'v120'
	def init(self):
		msvs.msvs_generator.init(self)
		self.vsnode_target = vsnode_target

def options(opt):
	opt.load('compiler_cxx')
	opt.load('python')
	opt.load('cuda')

def configure_msvc_x64_common(conf):
	flags = [
		'/WX', '/W4', '/MP',
		'/EHsc',
		'/wd4127', # C4127: conditional expression is constant.
		'/wd4251', # C4251: * needs to have dll-interface to be used by clients of class '*'.
		'/openmp',
		]

	conf.env['CFLAGS'] += flags
	conf.env['CXXFLAGS'] += flags

	conf.env['DEFINES'] += [
		'_WIN32', 
		'_WIN64', 
		'FLOW_PLATFORM_WINDOWS', 
		'FLOW_PLATFORM_WIN64', 
		'_UNICODE', 
		'UNICODE',
		'_CRT_SECURE_NO_WARNINGS',
		'_SCL_SECURE_NO_DEPRECATE',
	]
	conf.env['LINKFLAGS'] += [ '/MACHINE:X64' ]
	conf.env['LIBS'] += ["kernel32", "user32", "gdi32", "comdlg32", "advapi32", "Ws2_32", "psapi", "Rpcrt4", "Shell32", "Ole32"]
	
	conf.env['CUDAFLAGS'] += ['--use-local-env', '--cl-version 2013', '--machine 64', '--compile', '-Xcompiler "'+' '.join(flags)+'Xcudafe "--diag_suppress=field_without_dll_interface"']

def configure_msvc_x64_debug(conf):
	configure_msvc_x64_common(conf)
	flags = ['/MDd', '/Od']
	conf.env['CFLAGS'] += flags
	conf.env['CXXFLAGS'] += flags
	conf.env['DEFINES'] += ['_DEBUG', 'FLOW_BUILD_DEBUG']
	conf.env['LIBPATH_SIMPLEITK'] = os.path.join(conf.env['SIMPLEITK_BUILD'], 'SimpleITK-build', 'lib', 'Debug')
	conf.env['CUDAFLAGS'] += ["-G", "-g"]

def configure_msvc_x64_release(conf):
	configure_msvc_x64_common(conf)
	flags = ['/MD', '/O2']
	conf.env['CFLAGS'] += flags
	conf.env['CXXFLAGS'] += flags
	conf.env['DEFINES'] += ['NDEBUG', 'FLOW_BUILD_RELEASE']
	conf.env['LIBPATH_SIMPLEITK'] = os.path.join(conf.env['SIMPLEITK_BUILD'], 'SimpleITK-build', 'lib', 'Release')

@before_method('propagate_uselib_vars')
@feature('numpy')
def init_numpy(self):
	"""
	Add the NUMPY variable.
	"""
	self.uselib = self.to_list(getattr(self, 'uselib', []))
	if not 'NUMPY' in self.uselib:
		self.uselib.append('NUMPY')

@before_method('propagate_uselib_vars')
@feature('simpleitk')
def init_simpleitk(self):
	"""
	Add the SIMPLEITK variable.
	"""
	self.uselib = self.to_list(getattr(self, 'uselib', []))
	if not 'SIMPLEITK' in self.uselib:
		self.uselib.append('SIMPLEITK')


def configure(conf):
	conf.load('compiler_cxx')
	conf.load('python')
	conf.load('cuda')
	conf.check_python_version()
	conf.check_python_headers('pyembed')

	# Look for numpy
	numpy_inc_path = ''
	for p in [__import__('numpy').get_include(), os.path.join(conf.env['PYTHONDIR'], 'numpy', 'core', 'include')]:
		if os.path.isfile(os.path.join(p, 'numpy', 'arrayobject.h')):
			numpy_inc_path = p

	if numpy_inc_path == '':
		conf.fatal('Failed to determine include path for numpy.')

	conf.env['INCLUDES_NUMPY'] = [numpy_inc_path]
	conf.env['LIBPATH_NUMPY'] = conf.env['LIBPATH_PYEMBED']

	#TODO:
	
	simpleitk_path = ''
	for p in ['C:\\dev\\SimpleITK-0.9.1\\build_sharedlib', 'D:\\SimpleITK-0.9.1\\build_sharedlib']:
		if os.path.isdir(p):
			simpleitk_path = p

	if simpleitk_path == '':
		conf.fatal('Failed to determine path for SimpleITK.')

	conf.env['SIMPLEITK_BUILD'] = simpleitk_path
	conf.env['INCLUDES_SIMPLEITK'] = [os.path.join(conf.env['SIMPLEITK_BUILD'],'include','SimpleITK-0.9')]
	conf.env['LIB_SIMPLEITK'] = ['SimpleITKIO-0.9', 'SimpleITKCommon-0.9', 'SimpleITKExplicit-0.9']

	variant_configure = {
		'win64_debug': configure_msvc_x64_debug,
		'win64_release': configure_msvc_x64_release,
	}

	for p in supported_platforms():
		for c in CONFIGURATIONS:
			v = p + '_' + c
			conf.setenv(v, env=conf.env.derive().detach()) # Make sure to make a deep copy of base env
			variant_configure[v](conf)
			conf.setenv('')

	conf.recurse(SUBFOLDERS, mandatory=False)

def build(bld):
	if bld.cmd == 'msvs2013':
		print 'Generating MSVS files'
		bld.solution_name = APPAME + '.sln'
		bld.configurations = [configuration_waf_to_vs(c) for c in CONFIGURATIONS]
		bld.platforms = ['x64']		
		bld.projects_dir = bld.srcnode.make_node('.depproj')
		bld.projects_dir.mkdir()

		bld.env['PLATFORM'] = 'msvs2013'
		bld.env['CONFIGURATION'] = ''
	else:
		if not bld.variant:
			# A variant needs to be specified, the variant is of the form "<platform>_<configuration>"
			bld.fatal('No variant specified, read the comments in the wscript file!')

		bld.env['PLATFORM'] = bld.platform
		bld.env['CONFIGURATION'] = bld.configuration

		print 'Variant: %s' % bld.variant
	bld.recurse(SUBFOLDERS, mandatory=False)



def init(ctx):
	from waflib.Build import BuildContext, CleanContext, InstallContext, UninstallContext

	for p in supported_platforms():
		for c in CONFIGURATIONS:
			for x in (BuildContext, CleanContext, InstallContext, UninstallContext):
				name = x.__name__.replace('Context','').lower()
				class tmp(x):
					cmd = name + '_' + p + '_' + c
					variant = p + '_' + c
					platform = p
					configuration = c

