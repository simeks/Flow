// Copyright 2008-2014 Simon Ekström

#ifndef __LOG_H__
#define __LOG_H__

#define MAX_MSG_SIZE 4096
#define MAX_LOG_NAME_LEN 1024

/// @brief Representation of a log file
class Log
{
    char _file_name[MAX_LOG_NAME_LEN];
    FILE* _file_handle;

public:
    Log();
    ~Log();

    void open(const char* file_name);
    void close();

    bool is_open() const;

    //-------------------------------------------------------------------------------
    void write(const char* fmt, ...);
    void write_v(const char* fmt, va_list arg);
    //-------------------------------------------------------------------------------

};


#endif // __LOG_H__

