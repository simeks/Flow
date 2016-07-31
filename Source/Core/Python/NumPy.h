#ifndef __CORE_PYTHON_NUMPY_H__
#define __CORE_PYTHON_NUMPY_H__

#include "Flow/FlowImage.h"

namespace numpy
{
    void initialize();

    PyObject* create_array(FlowImage* img);
    bool read_array(FlowImage* img, PyObject* arr, int type_hint = image::PixelType_Unknown);
}

#endif // __CORE_PYTHON_NUMPY_H__
