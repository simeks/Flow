// Copyright 2008-2014 Simon Ekström

#ifndef _TIMER_H
#define _TIMER_H


namespace timer
{
    CORE_API void initialize();

    /// Tick count at the initalization of this timer.
    CORE_API uint64_t start_tick_count();

    /// Returns the applications current tick count.
    CORE_API uint64_t tick_count();

    /// Returns elapsed seconds since timer initialization.
    CORE_API double seconds();

    /// Returns the number of seconds per timer tick.
    CORE_API double seconds_per_tick();

};




#endif // _TIMER_H
