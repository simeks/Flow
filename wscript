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
from waflib.Configure import conf

QT5_MODULES = ['Qt5Core', 'Qt5Gui', 'Qt5Widgets']


def supported_platforms():
	p = Utils.unversioned_sys_platform()
	return PLATFORMS[p]

def platform_vs_to_waf(p):
	if p == 'Win32':
		return 'win32'
	if p == 'x64':
		return 'win64'

def configuration_vs_to_waf(p):
	if p == 'Debug':
		return 'debug'
	if p == 'Release':
		return 'release'

def configuration_waf_to_vs(p):
	if p == 'debug':
		return 'Debug'
	if p == 'release':
		return 'Release'


class vsnode_target(msvs.vsnode_target):
	def get_build_command(self, props):
		p = self.get_build_params(props)
		return "%s build_%s_%s %s" % (p[0], platform_vs_to_waf(props.platform), configuration_vs_to_waf(props.configuration), p[1])
	#def collect_source(self):
		# TODO: likely to be required
	def collect_properties(self):
		"""
		Visual studio projects are associated with platforms and configurations (for building especially)
		"""
		super(vsnode_target, self).collect_properties()
		for x in self.build_properties:
			variant = '%s_%s' % (platform_vs_to_waf(x.platform), configuration_vs_to_waf(x.configuration))
			v = self.ctx.all_envs[variant]

			x.outdir = os.path.join(self.ctx.path.abspath(), out, variant, 'Source')
			x.preprocessor_definitions = ''
			x.includes_search_path = ''

			try:
				tsk = self.tg.link_task
			except AttributeError:
				pass
			else:
				x.output_file = os.path.join(x.outdir, tsk.outputs[0].win32path().split(os.sep)[-1])
				x.preprocessor_definitions = ';'.join(v.DEFINES)
				x.includes_search_path = ';'.join(v.INCPATHS)	
		
class msvs_2013(msvs.msvs_generator):
	cmd = 'msvs2013'
	numver = '13.00'
	vsver = '2013'
	platform_toolset_ver = 'v120'
	def init(self):
		msvs.msvs_generator.init(self)
		self.vsnode_target = vsnode_target

def options(opt):
	opt.load('compiler_cxx python cuda qt5 msvs')
	opt.add_option('--simpleitk', dest='simpleitk_root', action='store', default=False, help='Path to SimpleITK.')

def configure_msvc_x64_common(conf):
	flags = [
		'/FS', '/WX', '/W4', '/MP',
		'/EHsc',
		'/wd4127', # C4127: conditional expression is constant.
		'/wd4251', # C4251: * needs to have dll-interface to be used by clients of class '*'.
		'/openmp',
		]

	v = conf.env
	v.CFLAGS += flags
	v.CXXFLAGS += flags

	v.DEFINES += [
		'_WIN32', 
		'_WIN64', 
		'FLOW_PLATFORM_WINDOWS', 
		'FLOW_PLATFORM_WIN64', 
		'_UNICODE', 
		'UNICODE',
		'_CRT_SECURE_NO_WARNINGS',
		'_SCL_SECURE_NO_DEPRECATE',
	]
	v.LINKFLAGS += [ '/MACHINE:X64' ]
	v.LIB += ["kernel32", "user32", "gdi32", "comdlg32", "advapi32", "Ws2_32", "psapi", "Rpcrt4", "Shell32", "Ole32"]
	
	v.CUDAFLAGS += ['--use-local-env', '--cl-version=2013', '--machine=64', '--compile', '-Xcudafe="--diag_suppress=field_without_dll_interface"']

def configure_msvc_x64_debug(conf):
	configure_msvc_x64_common(conf)
	flags = ['/MDd', '/Od']

	v = conf.env
	v.CFLAGS += flags
	v.CXXFLAGS += flags
	v.DEFINES += ['_DEBUG', 'FLOW_BUILD_DEBUG']
	v.CUDAFLAGS += ['-G', '-g', '-Xcompiler="'+' '.join(v.CXXFLAGS)+'"']

def configure_msvc_x64_release(conf):
	configure_msvc_x64_common(conf)
	flags = ['/MD', '/O2']

	v = conf.env
	v.CFLAGS += flags
	v.CXXFLAGS += flags
	v.DEFINES += ['NDEBUG', 'FLOW_BUILD_RELEASE']
	v.CUDAFLAGS += ['-Xcompiler="'+' '.join(v.CXXFLAGS)+'"']

def configure(conf):
	conf.load('compiler_cxx python cuda msvs')

	v = conf.env

	# Qt5
	conf.load('qt5')
	
	v.LIB_QT5 = []
	v.LIB_QT5_DEBUG = []
	v.LIBPATH_QT5 = []
	v.LIBPATH_QT5_DEBUG = []
	v.INCLUDES_QT5 = []
	v.INCLUDES_QT5_DEBUG = []
	v.DEFINES_QT5 = []
	v.DEFINES_QT5_DEBUG = []

	for m in QT5_MODULES:
		v.LIB_QT5 += v['LIB_%s' % m.upper()]
		v.LIB_QT5_DEBUG += v['LIB_%s_DEBUG' % m.upper()]
		v.INCLUDES_QT5 += v['INCLUDES_%s' % m.upper()]
		v.INCLUDES_QT5_DEBUG += v['INCLUDES_%s' % m.upper()]
		v.DEFINES_QT5 += v['DEFINES_%s' % m.upper()]
		v.DEFINES_QT5_DEBUG += v['DEFINES_%s_DEBUG' % m.upper()]
		v.LIBPATH_QT5 += v['LIBPATH_%s' % m.upper()]
		v.LIBPATH_QT5_DEBUG += v['LIBPATH_%s_DEBUG' % m.upper()]

	# Python
	conf.check_python_version()
	conf.check_python_headers('pyembed')

	# Numpy
	numpy_inc_path = ''
	for p in [__import__('numpy').get_include(), os.path.join(v.PYTHONDIR, 'numpy', 'core', 'include')]:
		if os.path.isfile(os.path.join(p, 'numpy', 'arrayobject.h')):
			numpy_inc_path = p

	if numpy_inc_path == '':
		conf.fatal('Failed to determine include path for numpy.')

	v.INCLUDES_NUMPY = [numpy_inc_path]
	v.LIBPATH_NUMPY = v.LIBPATH_PYEMBED

	# SimpleITK
	sitk_root = conf.root.find_node(conf.options.simpleitk_root)
	# TODO: Path
	for p in ['C:\\dev\\SimpleITK-0.9.1\\build_sharedlib', 'D:\\SimpleITK-0.9.1\\build_sharedlib']:
		if sitk_root != None:
			break
		sitk_root = conf.root.find_node(p)
	
	print 'SimpleITK: %s' % sitk_root.abspath()
	if sitk_root == None:
		conf.fatal('Failed to determine location of for SimpleITK.')

	# Release
	sitk_libpath = sitk_root.find_node('SimpleITK-build/lib/Release').abspath()
	sitk_includes = sitk_root.find_node('include/SimpleITK-0.9').abspath()
	conf.check_cxx(
		header_name='sitkCommon.h', 
		lib=['SimpleITKCommon-0.9', 'SimpleITKIO-0.9', 'SimpleITKExplicit-0.9'], 
		libpath=sitk_libpath, 
		includes=sitk_includes, 
		uselib_store='SIMPLEITK', 
		mandatory=True)

	# Debug
	sitk_libpath = sitk_root.find_node('SimpleITK-build/lib/Debug').abspath()
	conf.check_cxx(
		header_name='sitkCommon.h', 
		lib=['SimpleITKCommon-0.9', 'SimpleITKIO-0.9', 'SimpleITKExplicit-0.9'], 
		libpath=sitk_libpath, 
		includes=sitk_includes, 
		uselib_store='SIMPLEITK_DEBUG', 
		mandatory=True)

	# sqlite
	sqlite3_root = conf.path.find_node('External/sqlite').abspath()
	sqlite3_libpath = [sqlite3_root]
	sqlite3_includes = [sqlite3_root]
	conf.check_cxx(
		header_name='sqlite3.h', 
		lib='sqlite3', 
		libpath=sqlite3_libpath, 
		includes=sqlite3_includes, 
		uselib_store='SQLITE3',
		mandatory=True)

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
	v = bld.env
	if bld.cmd == 'msvs2013':
		print 'Generating MSVS files'
		bld.solution_name = APPNAME + '.sln'
		bld.configurations = [configuration_waf_to_vs(c) for c in CONFIGURATIONS]
		bld.platforms = ['x64']
		bld.projects_dir = bld.srcnode.make_node('.depproj')
		bld.projects_dir.mkdir()

		v.PLATFORM = 'msvs2013'
		v.CONFIGURATION = ''
	else:
		if not bld.variant:
			# A variant needs to be specified, the variant is of the form "<platform>_<configuration>"
			bld.fatal('No variant specified, read the comments in the wscript file!')

		v.PLATFORM = bld.platform
		v.CONFIGURATION = bld.configuration

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

