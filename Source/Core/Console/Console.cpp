// Copyright 2008-2014 Simon Ekström

#include "Common.h"

#include "Console.h"
#include "Log.h"

#include "Thread/Thread.h"

// Define to write log output to OutputDebugString
#ifdef FLOW_BUILD_DEBUG
#ifdef FLOW_PLATFORM_WINDOWS
#define OUTPUTDEBUGSTRING(msg) OutputDebugStringA(msg); \
    printf(msg);
#elif FLOW_PLATFORM_MACOSX
#define OUTPUTDEBUGSTRING(msg) printf("%s", msg)
#endif
#else
#define OUTPUTDEBUGSTRING(msg) ((void)0)
#endif

//-------------------------------------------------------------------------------
namespace
{
    Log g_main_log;
    CriticalSection g_log_lock;

    console::OutputCallback g_callback = 0;
    void* g_callback_data = 0;
}

//-------------------------------------------------------------------------------
void console::set_output_file(const char* file_name)
{
    ScopedLock<CriticalSection> scoped_lock(g_log_lock);
    if (g_main_log.is_open())
        g_main_log.close();
    g_main_log.open(file_name);
}
//-------------------------------------------------------------------------------
void console::print(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    print_v(fmt, args);
    va_end(args);
}
void console::print_v(const char* fmt, va_list arg)
{
    ScopedLock<CriticalSection> scoped_lock(g_log_lock);

    char msg[MAX_MSG_SIZE];
    int size = vsnprintf(msg, MAX_MSG_SIZE - 2, fmt, arg);

    // Run callback before appending '\n'
    if (g_callback)
    {
        g_callback(g_callback_data, 0, msg);
    }

    if (size < 0) // Was string truncated?
    {
        msg[sizeof(msg) - 1] = '\0';
    }

    if (g_main_log.is_open())
    {
        g_main_log.write(msg);
    }

    OUTPUTDEBUGSTRING(msg);
}
void console::warning(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    warning_v(fmt, args);
    va_end(args);
}
void console::warning_v(const char* fmt, va_list arg)
{
    ScopedLock<CriticalSection> scoped_lock(g_log_lock);
    
    char msg[MAX_MSG_SIZE];
    msg[0] = '\0';
    size_t len = 0;

    strcpy(msg, "[Warning] ");
    len = strlen(msg);

    int size = vsnprintf(msg + len, MAX_MSG_SIZE - len - 2, fmt, arg);

    // Run callback before appending '\n'
    if (g_callback)
    {
        g_callback(g_callback_data, Msg_Warning, msg + len);
    }


    if (size < 0) // Was string truncated?
    {
        msg[MAX_MSG_SIZE - 2] = '\n'; msg[MAX_MSG_SIZE - 1] = '\0';
    }
    else
    {
        strcat(msg, "\n");
    }
    if (g_main_log.is_open())
    {
        g_main_log.write(msg);
    }

    OUTPUTDEBUGSTRING(msg);
}
void console::error(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    error_v(fmt, args);
    va_end(args);
}
void console::error_v(const char* fmt, va_list arg)
{
    {
        ScopedLock<CriticalSection> scoped_lock(g_log_lock);

        char msg[MAX_MSG_SIZE];
        size_t len = 0;

        strcpy(msg, "[Error] ");
        len = strlen(msg);

        int size = vsnprintf(msg + len, MAX_MSG_SIZE - len - 2, fmt, arg);

        // Run callback before appending '\n'
        if (g_callback)
        {
            g_callback(g_callback_data, Msg_Error, msg + len);
        }
        if (size < 0) // Was string truncated?
        {
            msg[MAX_MSG_SIZE - 2] = '\n'; msg[MAX_MSG_SIZE - 1] = '\0';
        }
        else
        {
            strcat(msg, "\n");
        }
        if (g_main_log.is_open())
        {
            g_main_log.write(msg);
        }


        OUTPUTDEBUGSTRING(msg);
    }

}

void console::fatal_error(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    fatal_error_v(fmt, args);
    //va_end(args); // Fatal error terminates application so we won't get here
}
void console::fatal_error_v(const char* fmt, va_list arg)
{
    {
        ScopedLock<CriticalSection> scoped_lock(g_log_lock);

        char msg[MAX_MSG_SIZE];
        size_t len = 0;

        strcpy(msg, "[Error] ");
        len = strlen(msg);

        int size = vsnprintf(msg + len, MAX_MSG_SIZE - len - 2, fmt, arg);

        // Run callback before appending '\n'
        if (g_callback)
        {
            g_callback(g_callback_data, Msg_FatalError, msg + len);
        }
        if (size < 0) // Was string truncated?
        {
            msg[MAX_MSG_SIZE - 2] = '\n'; msg[MAX_MSG_SIZE - 1] = '\0';
        }
        else
        {
            strcat(msg, "\n");
        }
        if (g_main_log.is_open())
        {
            g_main_log.write(msg);
        }


        OUTPUTDEBUGSTRING(msg);
    }
    exit(1);
}
void console::assert_message(const char* fmt, ...)
{
    ScopedLock<CriticalSection> scoped_lock(g_log_lock);

    char tmp[MAX_MSG_SIZE];
    size_t len = 0;

    strcpy(tmp, "[Assertion] ");
    len = strlen(tmp);

    va_list args;
    va_start(args, fmt);
    vsnprintf(tmp + len, MAX_MSG_SIZE - len - 2, fmt, args);
    va_end(args);

    tmp[MAX_MSG_SIZE - 2] = '\n'; tmp[MAX_MSG_SIZE - 1] = '\0';

    // Run callback before appending '\n'
    if (g_callback)
    {
        g_callback(g_callback_data, Msg_Error|Msg_Assert, tmp + 12);
    }

    strcat(tmp, "\n");
    if (g_main_log.is_open())
    {
        g_main_log.write(tmp);
    }


    OUTPUTDEBUGSTRING(tmp);

}

void console::set_callback(OutputCallback callback, void* data)
{
    g_callback = callback;
    g_callback_data = data;
}



