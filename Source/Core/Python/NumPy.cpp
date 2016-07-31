#include "Common.h"

#include "NumPy.h"

#include "PythonWrapper.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


void numpy::initialize()
{
    import_array();
}
PyObject* numpy::create_array(FlowImage* img)
{
    assert(img->pixel_type() != image::PixelType_Unknown);
    if (img->pixel_type() == image::PixelType_Unknown)
    {
        PyErr_SetString(PyExc_AttributeError, "Cannot create numpy array of unknown type.");
        return nullptr;
    }

    PyObject* ret = nullptr;
    if (img->pixel_type() == image::PixelType_Vec3f ||
        img->pixel_type() == image::PixelType_Vec3d)
    {
        int np_type = -1;
        switch (img->pixel_type())
        {
        case image::PixelType_Vec3f:
            np_type = NPY_FLOAT32;
            break;
        case image::PixelType_Vec3d:
            np_type = NPY_FLOAT64;
            break;
        }

        int nd = img->ndims();
        npy_intp dims[4];

        int i = 0;
        for (; i < nd; ++i)
        {
            dims[nd - 1 - i] = img->size()[i];
        }
        dims[i] = 3;

        PyArrayObject* obj = (PyArrayObject*)PyArray_SimpleNew(nd+1, dims, np_type);
        uint8_t* dest = (uint8_t*)PyArray_DATA(obj);

        Image img_data = img->image();
        assert(int(img_data.pixel_count() * img_data.pixel_size()) == PyArray_NBYTES(obj));
        img_data.copy_to(dest);

        ret = (PyObject*)obj;
    }
    else if (img->pixel_type() == image::PixelType_Vec4u8 || img->pixel_type() == image::PixelType_Vec4f)
    {
        int np_type = NPY_UINT8;

        int nd = img->ndims();
        npy_intp dims[4];

        int i = 0;
        for (; i < nd; ++i)
        {
            dims[nd - 1 - i] = img->size()[i];
        }
        dims[i] = 4;

        PyArrayObject* obj = (PyArrayObject*)PyArray_SimpleNew(nd+1, dims, np_type);
        uint8_t* dest = (uint8_t*)PyArray_DATA(obj);

        Image img_data = img->image();
        assert(int(img_data.pixel_count() * img_data.pixel_size()) == PyArray_NBYTES(obj));
        img_data.copy_to(dest);

        ret = (PyObject*)obj;
    }
    else
    {
        int np_type = -1;
        switch (img->pixel_type())
        {
        case image::PixelType_UInt8:
            np_type = NPY_UINT8;
            break;
        case image::PixelType_UInt16:
            np_type = NPY_UINT16;
            break;
        case image::PixelType_UInt32:
            np_type = NPY_UINT32;
            break;
        case image::PixelType_Float32:
            np_type = NPY_FLOAT32;
            break;
        case image::PixelType_Float64:
            np_type = NPY_FLOAT64;
            break;
        }

        int nd = img->ndims();
        npy_intp dims[4];

        int i = 0;
        for (; i < nd; ++i)
        {
            dims[nd - 1 - i] = img->size()[i];
        }

        PyArrayObject* obj = (PyArrayObject*)PyArray_SimpleNew(nd, dims, np_type);
        uint8_t* dest = (uint8_t*)PyArray_DATA(obj);
        
        Image img_data = img->image();
        assert(int(img_data.pixel_count() * img_data.pixel_size()) == PyArray_NBYTES(obj));
        img_data.copy_to(dest);

        ret = (PyObject*)obj;
    }
    return ret;
}
bool numpy::read_array(FlowImage* img, PyObject* arr, int type_hint)
{
    if (!arr || !PyArray_Check(arr))
        return false;

    PyArrayObject* arr_object = (PyArrayObject*)arr;
    img->release_image();

    if (type_hint == image::PixelType_Vec3d || type_hint == image::PixelType_Vec3f)
    {
        int ndims = PyArray_NDIM(arr_object);
        if (ndims > 4)
        {
            PyErr_SetString(PyExc_AttributeError, "Unsupported number of dimensions.");
            return false;
        }

        npy_intp* dims = PyArray_DIMS(arr_object);

        std::vector<uint32_t> size;
        for (int i = 0; i < ndims - 1; ++i)
        {
            size.push_back((uint32_t)dims[i]);
        }

        Image img_data(size, type_hint, (const uint8_t*)PyArray_DATA(arr_object));
        img->set_image(img_data);
    }
    else if (type_hint == image::PixelType_Vec4u8 || type_hint == image::PixelType_Vec4f)
    {
        int ndims = PyArray_NDIM(arr_object);
        if (ndims > 4)
        {
            PyErr_SetString(PyExc_AttributeError, "Unsupported number of dimensions.");
            return false;
        }

        npy_intp* dims = PyArray_DIMS(arr_object);

        std::vector<uint32_t> size;
        for (int i = 0; i < ndims - 1; ++i)
        {
            size.push_back((uint32_t)dims[i]);
        }

        Image img_data(size, type_hint, (const uint8_t*)PyArray_DATA(arr_object));
        img->set_image(img_data);
    }
    else if (type_hint != image::PixelType_Unknown)
    {
        int ndims = PyArray_NDIM(arr_object);

        npy_intp* dims = PyArray_DIMS(arr_object);

        std::vector<uint32_t> size;
        for (int i = ndims - 1; i >= 0; --i)
        {
            size.push_back((uint32_t)dims[i]);
        }

        Image img_data(size, type_hint, (const uint8_t*)PyArray_DATA(arr_object));
        img->set_image(img_data);
    }
    else
    {
        PyErr_SetString(PyExc_AttributeError, "Cannot convert numpy array to specified format.");
        return false;
    }
    return true;
}