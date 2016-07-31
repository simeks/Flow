#ifndef __CORE_PYTHON_H__
#define __CORE_PYTHON_H__

#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

#include "Common.h"

#endif // __CORE_PYTHON_H__
