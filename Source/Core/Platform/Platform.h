// Copyright 2008-2014 Simon Ekström


#ifndef _PLATFORM_H
#define _PLATFORM_H

#ifdef FLOW_PLATFORM_WINDOWS
#include "windows/Platform_win.h"
#elif FLOW_PLATFORM_MACOSX
#include "macosx/Platform_macosx.h"
#elif FLOW_PLATFORM_LINUX
#include "linux/Platform_linux.h"
#endif

#include <Core/API.h>

namespace platform
{
    CORE_API void set_utf8_output();
}


#endif // _PLATFORM_H
