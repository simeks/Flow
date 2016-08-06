require "tundra.syntax.files"

DefRule {
	Name = "GenerateMoc",
	Pass = "GenerateSources",
	Command = "$(QT5_BIN)/moc $(<) -o $(@)",
	
	Blueprint = {
		Input = { Required = true, Type = "string" },
		Output = { Required = true, Type = "string" },
	},

	Setup = function (env, data)
		return {
			InputFiles    = { data.Input },
			OutputFiles   = { "$(OBJECTDIR)/_generated/" .. data.Output },
		}
	end,
}

DefRule {
	Name = "GenerateQRC",
	Pass = "GenerateSources",
	Command = "$(QT5_BIN)/rcc $(<) -o $(@) -name application",

	Blueprint = {
		Input = { Required = true, Type = "string" },
		Output = { Required = true, Type = "string" },
	},

	Setup = function (env, data)
		return {
			InputFiles    = { data.Input },
			OutputFiles   = { "$(OBJECTDIR)/_generated/" .. data.Output },
		}
	end,
}

DefRule {
	Name = "GenerateUI",
	Pass = "GenerateSources",
	Command = "$(QT5_BIN)/uic $(<) -o $(@)",

	Blueprint = {
		Input = { Required = true, Type = "string" },
		Output = { Required = true, Type = "string" },
	},

	Setup = function (env, data)
		return {
			InputFiles    = { data.Input },
			OutputFiles   = { "$(OBJECTDIR)/_generated/" .. data.Output },
		}
	end,
}



local function GenerateMocSources(sources)
	local result = {}
	for _, src in ipairs(tundra.util.flatten(sources)) do
		result[#result + 1] = GenerateMoc { Input = src, Output = tundra.path.get_filename_base(src) .. "_moc.cpp" }
	end
	return result
end


local function GenerateUISources(sources)
	local result = {}
	for _, src in ipairs(tundra.util.flatten(sources)) do
		result[#result + 1] = GenerateUI { Input = src, Output = "ui_" .. tundra.path.get_filename_base(src) .. ".h" }
	end
	return result
end

SharedLibrary {
	Name = "Core",
	Env = {
		CPPPATH = { 
			"Source/Core",
			"Source",
			"$(SITK_BUILD)/include/SimpleITK-0.9",
			"$(PYTHON)/include",
			"$(NUMPY)/include",
			"$(CUDA_PATH)/include",
		}, 
		CPPDEFS = { "FLOW_CORE_EXPORTS" },
		LIBPATH = {
			"$(PYTHON)/libs",
			{ "$(SITK_LIBS)/Release", "$(SITK_BUILD)/ITK-build/lib/Release"; Config = { "win64-*-release" } },
			{ "$(SITK_LIBS)/Debug", "$(SITK_BUILD)/ITK-build/lib/Debug"; Config = { "win64-*-debug" } },
			"$(CUDA_PATH)/lib/x64",
		},
	},

	Sources = {
		FGlob {
			Dir = "Source/Core",
			Extensions = { ".c", ".cpp", ".cxx", ".h", ".inl", ".cu" },
			Filters = {
				{ Pattern = "[/\\]windows[/\\]"; Config = { "win64-*" } },
				{ Pattern = "[/\\]macosx[/\\]"; Config = "mac*-*" },
				{ Pattern = "[/\\]test[/\\]"; Config = {}; },
			},
		},
	},
	
	Libs = { 
		{ 	
			"kernel32.lib", "user32.lib", "gdi32.lib", "comdlg32.lib", "advapi32.lib", "Ws2_32.lib", "psapi.lib", "Rpcrt4.lib", "Shell32.lib", "Ole32.lib";
			Config = { "win64-*-*" } 
		},
		{
			"SimpleITKIO-0.9.lib",
			"SimpleITKCommon-0.9.lib",	
			"SimpleITKExplicit-0.9.lib",
			"cudart.lib"
		},
	},
}

Program {
	Name = "FlowLab",
	Sources = {
		FGlob {
			Dir = "Source/FlowLab",
			Extensions = { ".c", ".cpp", ".h", ".inl" },
			Filters = {
				{ Pattern = "[/\\]windows[/\\]"; Config = { "win64-*" } },
				{ Pattern = "[/\\]macosx[/\\]"; Config = "mac*-*" },
			},
		},
		GenerateMoc {
			Input = "Source/FlowLab/MainWindow.h",
			Output = "MainWindow_moc.cpp"
		},
		GenerateMoc {
			Input = "Source/FlowLab/Flow/QtFlowDiagramScene.h",
			Output = "QtFlowDiagramScene_moc.cpp"
		},
		GenerateMoc {
			Input = "Source/FlowLab/Flow/QtFlowDiagramView.h",
			Output = "QtFlowDiagramView_moc.cpp"
		},
		GenerateMoc {
			Input = "Source/FlowLab/Flow/QtFlowNode.h",
			Output = "QtFlowNode_moc.cpp"
		},
		GenerateMoc {
			Input = "Source/FlowLab/Flow/QtFlowPin.h",
			Output = "QtFlowPin_moc.cpp"
		},
		GenerateMoc {
			Input = "Source/FlowLab/Flow/QtFlowConnection.h",
			Output = "QtFlowConnection_moc.cpp"
		},
		GenerateMoc {
			Input = "Source/FlowLab/Flow/QtNodePropertyWidget.h",
			Output = "QtNodePropertyWidget_moc.cpp"
		},
		GenerateMoc {
			Input = "Source/FlowLab/ConsoleWidget.h",
			Output = "ConsoleWidget_moc.cpp"
		},
		GenerateUISources {
			Glob { 
				Dir = "Source/FlowLab", 
				Extensions = { ".ui" } 
			}, 
		},
		GenerateQRC {
			Input = "Source/FlowLab/resources.qrc",
			Output = "resources.cpp"
		}
	},
	Env = {
		CPPPATH = { 
			"$(QT5_INCLUDE)/QtWidgets",
			"$(QT5_INCLUDE)/QtGui",
			"$(QT5_INCLUDE)/QtCore", 
			"$(QT5_INCLUDE)",
			"$(OBJECTDIR)/_generated",
			"Source",
		},
		LIBPATH = {
			"$(QT5_LIBS)",
		},
		PROGOPTS = {
			{ "/SUBSYSTEM:WINDOWS"; Config = { "win64-*-release" } },
			{ "/SUBSYSTEM:CONSOLE"; Config = { "win64-*-debug" } },
		},
	},
	Libs = { 
		{ 	
			"kernel32.lib", "user32.lib", "gdi32.lib", "comdlg32.lib", "advapi32.lib", "Ws2_32.lib", "psapi.lib", "Rpcrt4.lib", "Shell32.lib";
			Config = { "win64-*-*" } 
		},
		{
			"Qt5Cored.lib", "Qt5Widgetsd.lib", "Qt5Guid.lib"; 
			Config = { "win64-*-debug" } 
		},
		{
			"Qt5Core.lib", "Qt5Widgets.lib", "Qt5Gui.lib"; 
			Config = { "win64-*-release", "win64-*-production" } 
		},
	},

	Frameworks = { "Cocoa", "QtCore", "QtWidgets", "QtGui", "OpenGL", "AGL"  },

	Depends = { "Core" },
}

Program {
	Name = "FlowCLI",
	Sources = {
		FGlob {
			Dir = "Source/FlowCLI",
			Extensions = { ".c", ".cpp", ".h", ".inl" },
			Filters = {
				{ Pattern = "[/\\]windows[/\\]"; Config = { "win64-*" } },
				{ Pattern = "[/\\]macosx[/\\]"; Config = "mac*-*" },
			},
		},
	},
	Env = {
		CPPPATH = { 
			"Source",
		},
	},
	Libs = { 
		{ 	
			"kernel32.lib", "user32.lib", "gdi32.lib", "comdlg32.lib", "advapi32.lib", "Ws2_32.lib", "psapi.lib", "Rpcrt4.lib", "Shell32.lib";
			Config = { "win64-*-*" } 
		},
	},


	Depends = { "Core" },
}

SharedLibrary {
	Name = "Registration",
	Target = "$(OBJECTDIR)/Plugins/Plugin_Registration.dll",
	Env = {
		CPPPATH = { 
			"Source/Plugins/Registration",
			"Source",
			"External/gco-v3.0/",
			"$(CUDA_PATH)/include",
		},
		LIBPATH = {
			"$(CUDA_PATH)/lib/x64",
		},
	},

	Sources = {
		FGlob {
			Dir = "Source/Plugins/Registration",
			Extensions = { ".c", ".cpp", ".h", ".inl", ".cu" },
			Filters = {
				{ Pattern = "[/\\]windows[/\\]"; Config = { "win64-*" } },
				{ Pattern = "[/\\]macosx[/\\]"; Config = "mac*-*" },
			},
		},
	},
	
	Libs = { 
		"cudart.lib"
	},
	Depends = { "Core" },
}

SharedLibrary {
	Name = "Imiomics",
	Target = "$(OBJECTDIR)/Plugins/Plugin_Imiomics.dll",

	Env = {
		CPPPATH = { 
			"Source/Plugins/Imiomics",
			"Source",
			"External/sqlite",
		},
		LIBPATH = {
			"External/sqlite",
		},
		CXXOPTS = { "/openmp" }
	},
	
	Sources = {
		FGlob {
			Dir = "Source/Plugins/Imiomics",
			Extensions = { ".c", ".cpp", ".h", ".inl" },
			Filters = {
				{ Pattern = "[/\\]windows[/\\]"; Config = { "win64-*" } },
				{ Pattern = "[/\\]macosx[/\\]"; Config = "mac*-*" },
			},
		},
	},
	Libs = { "sqlite3.lib" },
	Depends = { "Core" },
}



Default "FlowLab"
