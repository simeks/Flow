#ifndef __CORE_H__
#define __CORE_H__

#include "API.h"

class FlowSystem;
class Module;
class CORE_API Core
{
public:
    Core();
    ~Core();

    void initialize(int, char**);

    static Core& get();
    static void create();
    static void destroy();

private:
    static Core* s_instance;
};

#endif // __CORE_H__
