// Copyright 2008-2014 Simon Ekström

#ifndef __CORE_CONSOLE_H__
#define __CORE_CONSOLE_H__

#define FATAL_ERROR(fmt, ...) console::fatal_error("[Error] %s:%d: "##fmt, __FILE__, __LINE__, __VA_ARGS__)

namespace console
{
    enum Flags
    {
        Msg_Warning = 1,
        Msg_Error = 2,
        Msg_FatalError = 4,
        Msg_Assert = 8
    };

    /// @brief Initializes the console system
    ///	@param file_name Specifies name for the log file
    CORE_API void initialize(const char* file_name);
    CORE_API void shutdown();

    ///	@param file_name Specifies name for the log file
    CORE_API void set_output_file(const char* file_name);

    CORE_API void print(const char* fmt, ...);
    CORE_API void print_v(const char* fmt, va_list arg);

    CORE_API void warning(const char* fmt, ...);
    CORE_API void warning_v(const char* fmt, va_list arg);

    CORE_API void error(const char* fmt, ...);
    CORE_API void error_v(const char* fmt, va_list arg);

    /// Fatal error causing the application to terminate
    CORE_API void fatal_error(const char* fmt, ...);
    CORE_API void fatal_error_v(const char* fmt, va_list arg);

    CORE_API void assert_message(const char* fmt, ...);

    typedef void(*OutputCallback)(void*, uint32_t flags, const char* msg);
    CORE_API void set_callback(OutputCallback callback, void* data);

    //-------------------------------------------------------------------------------

} // namespace logging





#endif // __CORE_CONSOLE_H__