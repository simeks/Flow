// Copyright 2008-2014 Simon Ekström

#include "Common.h"

#include "Log.h"

#include <time.h>


//-------------------------------------------------------------------------------
Log::Log() : _file_handle(NULL)
{
    strcat(_file_name, "");
}
Log::~Log()
{
    close();
}
void Log::open(const char* file_name)
{
    strcpy_s(_file_name, file_name);
    if (strlen(_file_name) == 0)
        return;

    _file_handle = fopen(_file_name, "w");
}
void Log::close()
{
    if (_file_handle)
    {
        fclose(_file_handle);
        _file_handle = nullptr;
    }
}
bool Log::is_open() const
{
    return (_file_handle != 0);
}
//-------------------------------------------------------------------------------
void Log::write(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    write_v(fmt, args);
    va_end(args);
}
void Log::write_v(const char* fmt, va_list arg)
{
    char msg[MAX_MSG_SIZE];
    char timestamp[32];

    if (vsnprintf(msg, MAX_MSG_SIZE - 2, fmt, arg) < 0)
    {
        msg[MAX_MSG_SIZE - 2] = '\n'; msg[MAX_MSG_SIZE - 1] = '\0';
    }

    if (_file_handle)
    {
        // Add timestamp
        tm* pt = nullptr;
        time_t cur_time;
        time(&cur_time);
        pt = localtime(&cur_time);
        snprintf(timestamp, 32, "%02d:%02d:%02d: ", pt->tm_hour, pt->tm_min, pt->tm_sec);

        fprintf(_file_handle, "%s%s", timestamp, msg);
        fflush(_file_handle);
    }
}
//-------------------------------------------------------------------------------
