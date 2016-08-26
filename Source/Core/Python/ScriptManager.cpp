#include "Common.h"
#include "PythonWrapper.h"

#include "ScriptManager.h"
#include "PyFlowModule.h"

#include "Platform/FilePath.h"
#include "Platform/FileUtil.h"
#include "Platform/Process.h"
#include "PythonUtil.h"

#include "NumPy.h"

ScriptManager* ScriptManager::s_instance = nullptr;

ScriptManager& ScriptManager::get()
{
    assert(s_instance);
    return *s_instance;
}
void ScriptManager::create()
{
    if (!s_instance)
    {
        s_instance = new ScriptManager();
    }
}
void ScriptManager::destroy()
{
    if (s_instance)
    {
        delete s_instance;
        s_instance = nullptr;
    }
}
ScriptManager::ScriptManager()
{
    Py_Initialize();
    numpy::initialize();

    add_directory("Scripts");

    py_flow_module::init_module();
}
ScriptManager::~ScriptManager()
{
    Py_Finalize();
}
void ScriptManager::load_modules()
{
    std::vector<std::string> files;
    file_util::find_files("Scripts/*.py", files);
    for (auto& f : files)
    {
        FilePath path(f);
        path.trim_extension();

        PyObject* m = PyImport_ImportModule(path.c_str());
        if (m)
        {
            if (PyObject_HasAttrString(m, "install_module"))
            {
                PyObject* func = PyObject_GetAttrString(m, "install_module");
                if (func)
                {
                    if (!PyObject_CallObject(func, nullptr))
                    {
                        PyErr_Print();
                    }
                    else
                    {
                        console::print("Python: Module '%s' installed\n", path.c_str());
                    }
                }
            }
        }
        else
        {
            PyErr_Print();
        }
    }
}
void ScriptManager::add_directory(const std::string& path)
{
    PyObject* sys_path = PySys_GetObject("path");
    PyList_Append(sys_path, PyString_FromString(path.c_str()));
}

