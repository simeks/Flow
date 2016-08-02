require "tundra.syntax.files"


Program {
	Name = "Test_Core",
	Depends = { "Core" },
	Env = {
		CPPPATH = { 
			"Source/Core",
			"Source",
			"$(SITK_BUILD)/include/SimpleITK-0.9",
			"$(PYTHON)/include",
			"$(NUMPY)/include",
		}, 
		CPPDEFS = { "FLOW_CORE_EXPORTS" },
		LIBPATH = {
			"$(PYTHON)/libs",
			{ "$(SITK_LIBS)/Release", "$(SITK_BUILD)/ITK-build/lib/Release"; Config = { "win64-*-release" } },
			{ "$(SITK_LIBS)/Debug", "$(SITK_BUILD)/ITK-build/lib/Debug"; Config = { "win64-*-debug" } },
		},
	},

	Sources = {
		FGlob {
			Dir = "Source/Core/Flow/test",
			Extensions = { ".c", ".cpp", ".cxx", ".h", ".inl" },
			Filters = {
				{ Pattern = "[/\\]windows[/\\]"; Config = { "win64-*" } },
				{ Pattern = "[/\\]macosx[/\\]"; Config = "mac*-*" }
			},
		},
		FGlob {
			Dir = "Source/Core/Platform/test",
			Extensions = { ".c", ".cpp", ".cxx", ".h", ".inl" },
			Filters = {
				{ Pattern = "[/\\]windows[/\\]"; Config = { "win64-*" } },
				{ Pattern = "[/\\]macosx[/\\]"; Config = "mac*-*" }
			},
		},
		FGlob {
			Dir = "Source/Core/Cuda/test",
			Extensions = { ".c", ".cpp", ".cxx", ".h", ".inl" },
			Filters = {
				{ Pattern = "[/\\]windows[/\\]"; Config = { "win64-*" } },
				{ Pattern = "[/\\]macosx[/\\]"; Config = "mac*-*" }
			},
		},
		"Source/Tools/Testing/Framework.h",
		"Source/Tools/Testing/Framework.cpp"
	},

	Libs = { 
		{ 
			"kernel32.lib", 
			"user32.lib", 
			"gdi32.lib", 
			"comdlg32.lib", 
			"advapi32.lib", 
			"ws2_32.lib", 
			Config = { "win32-*-*", "win64-*-*" } 
		}
	},
}

Default "Test_Core"
