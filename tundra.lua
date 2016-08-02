require "tundra.syntax.glob"
native = require "tundra.native"


local win32_config = {
	Env = {
		QT5 = native.getenv("QT5", "C:\\Qt\\5.6\\msvc2013_64"),
		QT5_INCLUDE = "$(QT5)/include/",
		QT5_LIBS = "$(QT5)/lib/",
		QT5_BIN = "$(QT5)/bin/", 
		SITK = native.getenv("SITK", "D:\\SimpleITK-0.9.1"),
		SITK_BUILD = "$(SITK)/build_sharedlib",
		SITK_LIBS = "$(SITK_BUILD)/SimpleITK-build/lib",
		PYTHON = "C:\\Anaconda2",
		NUMPY = "$(PYTHON)/Lib/site-packages/numpy/core",
		CUDA_PATH = native.getenv("CUDA_PATH", ""),
		CPPDEFS = { "FLOW_PLATFORM_WINDOWS", "FLOW_PLATFORM_WIN64", "_UNICODE", "UNICODE" },
		CXXOPTS = {
			"/WX", "/W4", "/EHsc", "/D_CRT_SECURE_NO_WARNINGS", "/D_SCL_SECURE_NO_DEPRECATE",
			"/wd4127", -- C4127: conditional expression is constant.
			"/wd4251", -- C4251: * needs to have dll-interface to be used by clients of class '*'.
			{ 
				--"/analyze",
				"/MDd", 
				"/Od"; 
				Config = "*-*-debug" 
			},
			{ "/MD", "/O2"; Config = {"*-*-release", "*-*-production"} },
			"/openmp"
		},
		GENERATE_PDB = {
			{ "0"; Config = "*-vs2013-release" },
			{ "1"; Config = { "*-vs2013-debug", "*-vs2013-production" } },
		}
	},
}

local macosx_config = {
	Env = {
		QT5 = native.getenv("QT5", "D:\\Qt\\5.6\\msvc2013_64"),
		QT5_INCLUDE = "$(QT5)/include/",
		QT5_LIBS = "$(QT5)/lib/",
		QT5_BIN = "$(QT5)/bin/",
		CPPDEFS = { "FLOW_PLATFORM_MACOSX" },
		CXXOPTS = {
			"-Wall", "-std=c++11", "-stdlib=libc++",
			{ "-O0", "-g"; Config = "*-*-debug" },
			{ "-O2",  "/DNDEBUG"; Config = {"*-*-release", "*-*-production"} },
		},
		COPTS = {
			"-Wall",
			{ "-O0", "-g"; Config = "*-*-debug" },
			{ "-O2"; Config = {"*-*-release", "*-*-production"} },
		},
		LD = { "-lc++", "-F$(QT5_LIBS)", },
	}
}

Build {
	Configs = {
		Config {
			Name = "macosx-gcc",
			
			DefaultOnHost = "macosx",
			Tools = { "clang-osx" },
			Inherit = macosx_config,
		},
		Config {
			Name = 'win64-vs2013',
			Tools = { { "msvc-vs2013"; TargetArch = "x64" }, },
			DefaultOnHost = "windows",
			Inherit = win32_config,
		},
	},

	Env = {
		CPPDEFS = {
			{ "DEBUG", "FLOW_BUILD_DEBUG"; Config = "*-*-debug" },
			{ "NDEBUG", "FLOW_BUILD_RELEASE"; Config = {"*-*-release", "*-*-production"} },
		},
	},

	IdeGenerationHints = {
		Msvc = {
			-- Remap config names to MSVC platform names (affects things like header scanning & debugging)
			PlatformMappings = {
				['win64-vs2013'] = 'x64',
			},
			-- Remap variant names to MSVC friendly names
			VariantMappings = {
				['release']    = 'Release',
				['debug']      = 'Debug',
			},
		},

		-- Override solutions to generate and what units to put where.
		MsvcSolutions = {
			['Flow.sln'] = {},          -- receives all the units due to empty set
		},
		
		BuildAllByDefault = true,
	},

	Passes = {
		GenerateSources = { Name="GenerateSources", BuildOrder = 1 },
	},
	
    Variants = { "debug", "release" },

	Units = { "units.lua", "units.tests.lua" }
}

