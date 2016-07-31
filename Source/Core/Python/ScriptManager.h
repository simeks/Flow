#ifndef __CORE_PYTHON_SCRIPT_MANAGER_H__
#define __CORE_PYTHON_SCRIPT_MANAGER_H__

#include "API.h"

class CORE_API ScriptManager
{
public:
    ScriptManager();
    ~ScriptManager();

    void load_modules();
    void add_directory(const std::string& path);

    static ScriptManager& get();
    static void create();
    static void destroy();

private:
    static ScriptManager* s_instance;

};


#endif // __CORE_PYTHON_SCRIPT_MANAGER_H__
