#include "Common.h"

#include "Core.h"
#include "Flow/FlowType.h"
#include "Flow/FlowSystem.h"
#include "Modules/ModuleManager.h"
#include "Platform/FilePath.h"
#include "Platform/Process.h"
#include "Python/ScriptManager.h"

Core* Core::s_instance = nullptr;

Core& Core::get()
{
    assert(s_instance);
    return *s_instance;
}
void Core::create()
{
    if (!s_instance)
    {
        s_instance = new Core();
    }
}
void Core::destroy()
{
    if (s_instance)
    {
        delete s_instance;
        s_instance = nullptr;
    }
}

Core::Core()
{
}
Core::~Core()
{
    FlowSystem::destroy();
    ScriptManager::destroy();
    ModuleManager::destroy();

    flow_types::clear_types();
}
void Core::initialize(int, char**)
{
    console::print("Initializing Core\n");

    FlowSystem::create();
    ModuleManager::create();

    FilePath plugin_path(process::base_dir());
    plugin_path += "Plugins";
    ModuleManager::get().add_module_directory(plugin_path.get());
    
    console::print("Loading plugins:\n");
    std::vector<std::string> plugins;
    ModuleManager::get().find_modules("Plugins/*", plugins);
    for (auto& p : plugins)
    {
        console::print("* %s\n", p.c_str());
        ModuleManager::get().load_module(p);
    }

    ScriptManager::create();
    ScriptManager::get().load_modules();

    FlowSystem::get().initialize();
}
