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

import os, shutil
from waflib.extras import msvs
from waflib import Utils, Task
from waflib.TaskGen import extension, before_method, after_method, feature
from waflib.Configure import conf
from waflib.Node import Node

QT5_MODULES = ['Qt5Core', 'Qt5Gui', 'Qt5Widgets', 'icui18n', 'icuuc', 'icudata']
SIMPLEITK_MODULES = [
	'SimpleITKCommon-0.9',
	'SimpleITKIO-0.9',
]
ITK_MODULES = [
	'ITKIOVTK-4.8',
	'ITKCommon-4.8',
	'ITKIOBioRad-4.8',
	'ITKIOBMP-4.8',
	'ITKIOGE-4.8',
	'ITKIOGIPL-4.8',
	'ITKIOHDF5-4.8',
	'ITKIOImageBase-4.8',
	'ITKIOIPL-4.8',
	'ITKIOJPEG-4.8',
	'ITKIOLSM-4.8',
	'ITKIOMesh-4.8',
	'ITKIOMeta-4.8',
	'ITKIONIFTI-4.8',
	'ITKIONRRD-4.8',
	'ITKIOPNG-4.8',
	'ITKIOSiemens-4.8',
	'ITKIOStimulate-4.8',
	'ITKIOTIFF-4.8',
]


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
	def get_clean_command(self, props):
		p = self.get_build_params(props)
		v = platform_vs_to_waf(props.platform) + '_' + configuration_vs_to_waf(props.configuration)
		return "%s clean_%s %s" % (p[0], v, p[1])
	def get_rebuild_command(self, props):
		p = self.get_build_params(props)
		v = platform_vs_to_waf(props.platform) + '_' + configuration_vs_to_waf(props.configuration)
		return "%s clean_%s build_%s %s" % (p[0], v, v, p[1])
	def get_filter_name(self, node):
		# This method is modified to conform to the changes in the dirs() method.
		lst = msvs.diff(node, self.tg.path)
		return '\\'.join(lst[1:]) or ''
	def dirs(self):
		"""
		Get the list of parent folders of the source files (header files included)
		for writing the filters
		"""

		# This version of the modified is a bit special as it avoids adding its own root dir to the list.
		# The reason for this is that it's just annoying having the projects structured as:
		# ProjectA
		# - ProjectA
		# -- FileA
		# -- ... 

		lst = []
		def add(x):
			if x.height() > self.tg.path.height() + 1 and x not in lst:
				lst.append(x)
				add(x.parent)
		for x in self.source:
			add(x.parent)
		return lst
	def collect_source(self):
		tg = self.tg
		source_files = tg.to_nodes(getattr(tg, 'source', []))

		source = []
		def add(x):
			if x.height() > self.tg.path.height():
				source.extend(x.ant_glob('(*.h|*.hpp|*.H|*.inl|*.c|*.cpp|*.cu)'))
				add(x.parent)

		source_paths = []
		for x in source_files:
			if x.parent not in source_paths:
				source_paths.append(x.parent)

		for x in source_paths:
			add(x)

		# remove duplicates
		self.source.extend(list(set(source)))
		self.source.sort(key=lambda x: x.win32path())
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
				x.includes_search_path = ';'.join(self.tg.env.INCPATHS)	

class vsnode_build_all(msvs.vsnode_build_all):
	"""
	Fake target used to emulate the behaviour of "make all" (starting one process by target is slow)
	This is the only alias enabled by default
	"""
	def __init__(self, ctx, node, name='build_all_projects'):
		msvs.vsnode_build_all.__init__(self, ctx, node, name)
		self.is_active = True

	def get_build_command(self, props):
		p = self.get_build_params(props)
		return "%s build_%s_%s %s" % (p[0], platform_vs_to_waf(props.platform), configuration_vs_to_waf(props.configuration), p[1])
	def get_clean_command(self, props):
		p = self.get_build_params(props)
		v = platform_vs_to_waf(props.platform) + '_' + configuration_vs_to_waf(props.configuration)
		return "%s clean_%s %s" % (p[0], v, p[1])
	def get_rebuild_command(self, props):
		p = self.get_build_params(props)
		v = platform_vs_to_waf(props.platform) + '_' + configuration_vs_to_waf(props.configuration)
		return "%s clean_%s build_%s %s" % (p[0], v, v, p[1])

class vsnode_install_all(msvs.vsnode_install_all):
	"""
	Fake target used to emulate the behaviour of "make install"
	"""
	def __init__(self, ctx, node, name='install_all_projects'):
		msvs.vsnode_install_all.__init__(self, ctx, node, name)

	def get_build_command(self, props):
		p = self.get_build_params(props)
		return "%s build_%s_%s %s" % (p[0], platform_vs_to_waf(props.platform), configuration_vs_to_waf(props.configuration), p[1])
	def get_clean_command(self, props):
		p = self.get_build_params(props)
		v = platform_vs_to_waf(props.platform) + '_' + configuration_vs_to_waf(props.configuration)
		return "%s clean_%s %s" % (p[0], v, p[1])
	def get_rebuild_command(self, props):
		p = self.get_build_params(props)
		v = platform_vs_to_waf(props.platform) + '_' + configuration_vs_to_waf(props.configuration)
		return "%s clean_%s build_%s %s" % (p[0], v, v, p[1])

class msvs_2013(msvs.msvs_generator):
	cmd = 'msvs2013'
	numver = '13.00'
	vsver = '2013'
	platform_toolset_ver = 'v120'
	def init(self):
		msvs.msvs_generator.init(self)
		self.vsnode_target = vsnode_target
		self.vsnode_build_all = vsnode_build_all
		self.vsnode_install_all = vsnode_install_all

class copy_file(Task.Task):
	color = 'PINK'
	def run(self):
		shutil.copyfile(self.inputs[0].abspath(), self.outputs[0].abspath())


@feature('copy_qt_bins')
@after_method('apply_link')
def copy_qt_bins(self):
	v = self.env 
	if v.CONFIGURATION != 'debug' and v.CONFIGURATION != 'release':
		return

	if v.PLATFORM == 'win64':
		qt_bin = self.bld.root.make_node(v.QT_HOST_BINS)
	else:
		qt_bin = self.bld.root.make_node(v.LIBPATH_QT5[0])

	print qt_bin
	for m in QT5_MODULES:
	 	pf = 'd' if v.CONFIGURATION == 'debug' else ''
		f = v.cxxshlib_PATTERN % (m+pf)
		output = self.bld.path.find_node(out).find_node(v.PLATFORM + '_' + v.CONFIGURATION).find_node('Source')
		files = qt_bin.ant_glob([f, f+'.*'])
		for e in files:
	 		self.create_task('copy_file', e, output.make_node(e.name))

@feature('copy_simpleitk_bins')
@after_method('apply_link')
def copy_sitk_bins(self):
	v = self.env 
	if v.CONFIGURATION != 'debug' and v.CONFIGURATION != 'release':
		return

	if v.PLATFORM == 'win64':
		sitk_bin = self.bld.root.make_node([v.SIMPLEITK_ROOT, 'SimpleITK-build', 'bin', v.CONFIGURATION])
	else:
		sitk_bin = self.bld.root.make_node([v.SIMPLEITK_ROOT, 'SimpleITK-build', 'lib'])

	for m in SIMPLEITK_MODULES:
		f = v.cxxshlib_PATTERN % m
		output = self.bld.path.find_node(out).find_node(v.PLATFORM + '_' + v.CONFIGURATION).find_node('Source')

		files = sitk_bin.ant_glob([f, f+'.*'])
		for e in files:
	 		self.create_task('copy_file', e, output.make_node(e.name))

	 # ITK TODO: For now we assume it is located there
	if v.PLATFORM == 'win64':
		sitk_bin = self.bld.root.make_node([v.SIMPLEITK_ROOT, 'ITK-build', 'bin', v.CONFIGURATION])
	else:
		sitk_bin = self.bld.root.make_node([v.SIMPLEITK_ROOT, 'ITK-build', 'lib'])

	for m in ITK_MODULES:
		f = v.cxxshlib_PATTERN % m
		output = self.bld.path.find_node(out).find_node(v.PLATFORM + '_' + v.CONFIGURATION).find_node('Source')

		files = sitk_bin.ant_glob([f, f+'.*'])
		for e in files:
	 		self.create_task('copy_file', e, output.make_node(e.name))


def options(opt):
	opt.load('compiler_cxx python cuda qt5 msvs')
	opt.add_option('--simpleitk', dest='simpleitk_root', action='store', help='Path to SimpleITK.')
	opt.add_option('--no-cuda', dest='no_cuda', action='store_true', default=False)

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
	flags = ['/MDd', '/Od', '/Zi']

	v = conf.env
	v.CFLAGS += flags
	v.CXXFLAGS += flags
	v.DEFINES += ['DEBUG', '_DEBUG', 'FLOW_BUILD_DEBUG']
	v.CUDAFLAGS += ['-G', '-g', '-Xcompiler="'+' '.join(v.CXXFLAGS)+'"']
	v.LINKFLAGS += ['/DEBUG']

def configure_msvc_x64_release(conf):
	configure_msvc_x64_common(conf)
	flags = ['/MD', '/O2']

	v = conf.env
	v.CFLAGS += flags
	v.CXXFLAGS += flags
	v.DEFINES += ['NDEBUG', 'FLOW_BUILD_RELEASE']
	v.CUDAFLAGS += ['-Xcompiler="'+' '.join(v.CXXFLAGS)+'"']


def configure_gcc_x64_common(conf):
	flags = [
		'-m64', '-Werror', '-Wall', '-std=c++11', '-fopenmp',
		'-Wno-unused-variable',
		'-Wno-switch',
		]

	v = conf.env
	v.CC = 'gcc'
	v.CXX = 'g++'
	v.CFLAGS += flags
	v.CXXFLAGS += flags
	v.LINKFLAGS += [ '-fopenmp' ]

	v.DEFINES += [
		'FLOW_PLATFORM_LINUX', 
		'_UNICODE', 
		'UNICODE',
	]
def configure_gcc_x64_release(conf):
	configure_gcc_x64_common(conf)
	flags = ['-O2']

	v = conf.env
	v.CFLAGS += flags
	v.CXXFLAGS += flags
	v.DEFINES += ['NDEBUG', 'FLOW_BUILD_RELEASE']

def configure_clang_x64_common(conf):
	flags = [
		'-m64', '-Werror', '-Wall', '-std=c++11', '-fopenmp',
		'-Wno-inconsistent-missing-override',
		'-Wno-switch',
		]

	v = conf.env
	v.CC = 'clang'
	v.CXX = 'clang++'
	v.CFLAGS += flags
	v.CXXFLAGS += flags
	v.LINKFLAGS += [ '-fopenmp' ]

	v.DEFINES += [
		'FLOW_PLATFORM_LINUX', 
		'_UNICODE', 
		'UNICODE',
	]
def configure_clang_x64_release(conf):
	configure_clang_x64_common(conf)
	flags = ['-O2']

	v = conf.env
	v.CFLAGS += flags
	v.CXXFLAGS += flags
	v.DEFINES += ['NDEBUG', 'FLOW_BUILD_RELEASE']


def configure(conf):
	conf.load('compiler_cxx python msvs')

	v = conf.env

	if conf.options.no_cuda != True:
		v.USE_CUDA = True
		conf.load('cuda')

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

	v.RPATH += ['.']
	v.RPATH += v.LIBPATH_QT5

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
	sitk_root = None
	if conf.options.simpleitk_root:
		sitk_root = conf.root.find_node(conf.options.simpleitk_root)
	# TODO: Path
	for p in ['C:\\dev\\SimpleITK-0.9.1\\build_sharedlib', 'D:\\SimpleITK-0.9.1\\build_sharedlib']:
		if sitk_root != None:
			break
		sitk_root = conf.root.find_node(p)
	
	if sitk_root == None:
		conf.fatal('Failed to determine location of SimpleITK.')

	print 'SimpleITK: %s' % sitk_root.abspath()
	v.SIMPLEITK_ROOT = sitk_root.abspath()

	if Utils.unversioned_sys_platform() == 'win32':
		# Release
		sitk_libpath = [sitk_root.find_node('SimpleITK-build/lib/Release').abspath()]
		sitk_libpath += [sitk_root.find_node('ITK-build/lib/Release').abspath()]
		sitk_includes = sitk_root.find_node('include/SimpleITK-0.9').abspath()
		conf.check_cxx(
			header_name='sitkCommon.h', 
			lib=SIMPLEITK_MODULES + ITK_MODULES, 
			libpath=sitk_libpath, 
			includes=sitk_includes, 
			uselib_store='SIMPLEITK', 
			mandatory=True)

		# Debug
		sitk_libpath = [sitk_root.find_node('SimpleITK-build/lib/Debug').abspath()]
		sitk_libpath += [sitk_root.find_node('ITK-build/lib/Debug').abspath()]
		conf.check_cxx(
			header_name='sitkCommon.h', 
			lib=SIMPLEITK_MODULES + ITK_MODULES, 
			libpath=sitk_libpath, 
			includes=sitk_includes, 
			uselib_store='SIMPLEITK_DEBUG', 
			mandatory=True)
	else:
		# Release
		sitk_libpath = [sitk_root.find_node('SimpleITK-build/lib/').abspath()]
		sitk_libpath += [sitk_root.find_node('ITK-build/lib/').abspath()]
		sitk_includes = sitk_root.find_node('include/SimpleITK-0.9').abspath()
		conf.check_cxx(
			header_name='sitkCommon.h', 
			lib=SIMPLEITK_MODULES + ITK_MODULES, 
			libpath=sitk_libpath, 
			includes=sitk_includes, 
			uselib_store='SIMPLEITK', 
			mandatory=True)


		# Debug TODO: Linux debug libs
		sitk_libpath = [sitk_root.find_node('SimpleITK-build/lib/').abspath()]
		sitk_libpath += [sitk_root.find_node('ITK-build/lib/').abspath()]
		conf.check_cxx(
			header_name='sitkCommon.h', 
			lib=SIMPLEITK_MODULES + ITK_MODULES, 
			libpath=sitk_libpath, 
			includes=sitk_includes, 
			uselib_store='SIMPLEITK_DEBUG', 
			mandatory=True)
		
		v.RPATH += [sitk_root.find_node('ITK-build/lib/').abspath()]
		v.RPATH += [sitk_root.find_node('SimpleITK-build/lib/').abspath()]

	# sqlite

	if Utils.unversioned_sys_platform() == 'win32':
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
	else:
		conf.check_cxx(
			header_name='sqlite3.h', 
			lib='sqlite3', 
			uselib_store='SQLITE3',
			mandatory=True)

	# Platform specific setup

	variant_configure = {
		'win64_debug': configure_msvc_x64_debug,
		'win64_release': configure_msvc_x64_release,
		'linux_x64_gcc_release': configure_gcc_x64_release,
		'linux_x64_clang_release': configure_clang_x64_release,
	}

	for p in supported_platforms():
		for c in CONFIGURATIONS:
			v = p + '_' + c
			conf.setenv(v, env=conf.env.derive().detach()) # Make sure to make a deep copy of base env
			if v in variant_configure:			
				variant_configure[v](conf)
			else:			
				print 'No configuration set for variant %s' % v
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

