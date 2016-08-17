#include "Common.h"

#include "FlowImage.h"

#include "Python/NumPy.h"
#include "Python/PyFlowModule.h"
#include "Python/PyFlowObject.h"
#include "Python/PythonWrapper.h"

#include "Platform/FilePath.h"
#include "RVF/RVF.h"
#include "Image/ITK.h"

static PyObject* py_FlowImage_allocate_fn(PyObject* self, PyObject* args)
{
    FlowImage* object = object_cast<FlowImage>(py_flow_object::owner(self));
    if (object)
    {
        PyObject* pixel_type = nullptr;
        PyObject* size_tuple = nullptr;
        if (PyArg_ParseTuple(args, "SO:allocate", &pixel_type, &size_tuple))
        {
            std::vector<uint32_t> size;
            int ndims = (int)PyTuple_Size(size_tuple);
            for (int i = 0; i < ndims; ++i)
            {
                PyObject* s = PyTuple_GetItem(size_tuple, i);
                if (PyInt_Check(s))
                {
                    size.push_back(PyInt_AsLong(s));
                }
                else
                {
                    PyErr_SetString(PyExc_ValueError, "Invalid value in size tuple.");
                    return nullptr;
                }
            }

            char* pixel_type_str = PyString_AsString(pixel_type);
            for (int i = 0; pixel_type_str[i]; ++i) pixel_type_str[i] = (char)tolower(pixel_type_str[i]);

            int pixel_t = image::string_to_pixel_type(pixel_type_str);
            if(pixel_t == image::PixelType_Unknown)
            {
                PyErr_SetString(PyExc_AttributeError, "Pixel type not recognized.");
                return nullptr;
            }

            object->allocate(pixel_t, size);
        }
        else
        {
            return nullptr;
        }
    }

    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* py_FlowImage_ndims_fn(PyObject* self)
{
    FlowImage* object = object_cast<FlowImage>(py_flow_object::owner(self));
    if (object)
    {
        return PyInt_FromLong(object->ndims());
    }
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject* py_FlowImage_size_fn(PyObject* self)
{
    FlowImage* object = object_cast<FlowImage>(py_flow_object::owner(self));
    if (object)
    {
        PyObject* size_tuple = PyTuple_New(3);
        for (int i = 0; i < 3; ++i)
            PyTuple_SetItem(size_tuple, i, PyInt_FromLong(object->size()[i]));

        return size_tuple;
    }
    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* py_FlowImage_origin_fn(PyObject* self)
{
    FlowImage* object = object_cast<FlowImage>(py_flow_object::owner(self));
    if (object)
    {
        PyObject* tuple = PyTuple_New(3);
        for (int i = 0; i < 3; ++i)
            PyTuple_SetItem(tuple, i, PyFloat_FromDouble(object->origin()[i]));

        return tuple;
    }
    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* py_FlowImage_spacing_fn(PyObject* self)
{
    FlowImage* object = object_cast<FlowImage>(py_flow_object::owner(self));
    if (object)
    {
        PyObject* tuple = PyTuple_New(3);
        for (int i = 0; i < 3; ++i)
            PyTuple_SetItem(tuple, i, PyFloat_FromDouble(object->spacing()[i]));

        return tuple;
    }
    Py_INCREF(Py_None);
    return Py_None;
}


static PyObject* py_FlowImage_set_origin_fn(PyObject* self, PyObject* args)
{
    FlowImage* object = object_cast<FlowImage>(py_flow_object::owner(self));
    if (object)
    {
        PyObject* origin_tuple = nullptr;
        if (PyArg_ParseTuple(args, "O:set_origin", &origin_tuple))
        {
            if (PyTuple_Check(origin_tuple))
            {
                Vec3d origin;
                int ndims = (int)PyTuple_Size(origin_tuple);
                for (int i = 0; i < ndims; ++i)
                {
                    PyObject* s = PyTuple_GetItem(origin_tuple, i);
                    if (PyFloat_Check(s))
                    {
                        origin[i] = PyFloat_AsDouble(s);
                    }
                    else if (PyInt_Check(s))
                    {
                        origin[i] = PyInt_AsLong(s);
                    }
                    else
                    {
                        PyErr_SetString(PyExc_ValueError, "Invalid value in tuple.");
                        return nullptr;
                    }
                }
                object->set_spacing(origin);
            }
            else
            {
                PyErr_SetString(PyExc_TypeError, "Expected tuple.");
            }
        }
        else
        {
            return nullptr;
        }
    }

    Py_RETURN_NONE;
}
static PyObject* py_FlowImage_set_spacing_fn(PyObject* self, PyObject* args)
{
    FlowImage* object = object_cast<FlowImage>(py_flow_object::owner(self));
    if (object)
    {
        PyObject* spacing_tuple = nullptr;
        if (PyArg_ParseTuple(args, "O:set_spacing", &spacing_tuple))
        {
            if (PyTuple_Check(spacing_tuple))
            {
                Vec3d spacing;
                int ndims = (int)PyTuple_Size(spacing_tuple);
                for (int i = 0; i < ndims; ++i)
                {
                    PyObject* s = PyTuple_GetItem(spacing_tuple, i);
                    if (PyFloat_Check(s))
                    {
                        spacing[i] = PyFloat_AsDouble(s);
                    }
                    else if (PyInt_Check(s))
                    {
                        spacing[i] = PyInt_AsLong(s);
                    }
                    else
                    {
                        PyErr_SetString(PyExc_ValueError, "Invalid value in tuple.");
                        return nullptr;
                    }
                }
                object->set_spacing(spacing);
            }
            else
            {
                PyErr_SetString(PyExc_TypeError, "Expected tuple.");
            }
        }
        else
        {
            return nullptr;
        }
    }

    Py_RETURN_NONE;
}
static PyObject* py_FlowImage_pixel_type_fn(PyObject* self)
{
    FlowImage* object = object_cast<FlowImage>(py_flow_object::owner(self));
    if (object)
    {
        PyObject* str = nullptr;

        switch (object->pixel_type())
        {
        case image::PixelType_UInt8:
            str = PyString_FromString("uint8");
            break;
        case image::PixelType_UInt16:
            str = PyString_FromString("uint16");
            break;
        case image::PixelType_UInt32:
            str = PyString_FromString("uint32");
            break;
        case image::PixelType_Float32:
            str = PyString_FromString("float32");
            break;
        case image::PixelType_Float64:
            str = PyString_FromString("float64");
            break;
        case image::PixelType_Vec3f:
            str = PyString_FromString("vec3f");
            break;
        case image::PixelType_Vec4u8:
            str = PyString_FromString("vec4u8");
            break;
        case image::PixelType_Vec4f:
            str = PyString_FromString("vec4f");
            break;
        case image::PixelType_Vec4d:
            str = PyString_FromString("vec4d");
            break;
        default:
            str = PyString_FromString("unknown");
            break;
        }

        if (str != nullptr)
            return str;
    }
    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* py_FlowImage_to_array_fn(PyObject* self)
{
    FlowImage* object = object_cast<FlowImage>(py_flow_object::owner(self));
    if (object)
    {
        return numpy::create_array(object);
    }
    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* py_FlowImage_from_array_fn(PyObject* self, PyObject* args)
{
    FlowImage* object = object_cast<FlowImage>(py_flow_object::owner(self));
    if (object)
    {
        PyObject* arr = nullptr;
        PyObject* hint_str = nullptr;
        if (PyArg_ParseTuple(args, "OS:from_array", &arr, &hint_str))
        {
            int hint = image::PixelType_Unknown;
            if (hint_str)
            {
                hint = image::string_to_pixel_type(PyString_AsString(hint_str));
            }

            if (!numpy::read_array(object, arr, hint))
            {
                return nullptr;
            }
        }
        else
        {
            return nullptr;
        }
    }
    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* py_FlowImage_load_fn(PyObject* self, PyObject* args)
{
    FlowImage* object = object_cast<FlowImage>(py_flow_object::owner(self));
    if (object)
    {
        const char* file = nullptr;
        if (PyArg_ParseTuple(args, "s:load_from_file", &file))
        {
            if (file)
            {
                FilePath path(file);

                if (path.extension() == "rvf")
                {
                    Image img = rvf::load_rvf(path.get());
                    if (img.valid())
                    {
                        object->set_image(img);
                    }
                    else
                    {
                        PyErr_SetString(PyExc_IOError, "Failed to load RVF file.");
                        return nullptr;
                    }
                }
                else
                {
                    // Try ITK
                    Image img = image::load_image(path.get());
                    if (!img.valid())
                    {
                        PyErr_SetString(PyExc_IOError, "Failed to load image file.");
                        return nullptr;
                    }
                    object->set_image(img);
                }
            }
        }
        else
        {
            return nullptr;
        }
    }
    Py_INCREF(Py_None);
    return Py_None;
}
static PyObject* py_FlowImage_save_fn(PyObject* self, PyObject* args)
{
    FlowImage* object = object_cast<FlowImage>(py_flow_object::owner(self));
    if (object)
    {
        const char* file = nullptr;
        if (PyArg_ParseTuple(args, "s:save_to_file", &file))
        {
            if (file)
            {
                FilePath path(file);

                if (path.extension() == "rvf")
                {
                    // RVF file
                    if (object->pixel_type() != image::PixelType_Vec3d)
                    {
                        PyErr_SetString(PyExc_AttributeError, "RVF images must be of ImageVec3d format.");
                        return nullptr;
                    }

                    if (!rvf::save_rvf(path.get(), object->image()))
                    {
                        PyErr_SetString(PyExc_IOError, "Failed to save RVF file.");
                        return nullptr;
                    }
                }
                else
                {
                    // Try ITK
                    if (!image::save_image(path.get(), object->image()))
                    {
                        PyErr_SetString(PyExc_IOError, "Failed to save image file.");
                        return nullptr;
                    }
                }
            }
        }
        else
        {
            return nullptr;
        }
    }
    Py_INCREF(Py_None);
    return Py_None;
}



static PyMethodDef py_FlowImage_methods[] = {
    { "allocate", py_FlowImage_allocate_fn, METH_VARARGS, "Allocates an image." },
    { "ndims", (PyCFunction)py_FlowImage_ndims_fn, METH_NOARGS, "Returns the number of dimensions." },
    { "size", (PyCFunction)py_FlowImage_size_fn, METH_NOARGS, "Returns the size of the image." },
    { "origin", (PyCFunction)py_FlowImage_origin_fn, METH_NOARGS, "Returns the origin of the image." },
    { "spacing", (PyCFunction)py_FlowImage_spacing_fn, METH_NOARGS, "Returns the spacing of the image." },
    { "set_origin", (PyCFunction)py_FlowImage_set_origin_fn, METH_VARARGS, "Sets the origin of the image." },
    { "set_spacing", (PyCFunction)py_FlowImage_set_spacing_fn, METH_VARARGS, "Sets the spacing of the image." },
    { "pixel_type", (PyCFunction)py_FlowImage_pixel_type_fn, METH_NOARGS, "Returns the pixel type of the image." },
    { "to_array", (PyCFunction)py_FlowImage_to_array_fn, METH_NOARGS, "Returns the image data array." },
    { "from_array", py_FlowImage_from_array_fn, METH_VARARGS, "Sets the image data from a numpy array." },
    { "load_from_file", py_FlowImage_load_fn, METH_VARARGS, "Loads an image from the specified path." },
    { "save_to_file", py_FlowImage_save_fn, METH_VARARGS, "Saves the image to the specified path." },
    { NULL }  /* Sentinel */
};

IMPLEMENT_SCRIPT_OBJECT(FlowImage, "Image", "Image", py_FlowImage_methods);

FlowImage::FlowImage()
{
}
FlowImage::FlowImage(const Image& img) :
    _image(img)
{
}
FlowImage::~FlowImage()
{
}
void FlowImage::allocate(int type, const std::vector<uint32_t>& size)
{
    assert(type != image::PixelType_Unknown);
    _image.create(size, type);
}
void FlowImage::set_image(const Image& img)
{
    _image = img;
}
void FlowImage::release_image()
{
    _image.release();
}
void FlowImage::set_origin(const Vec3d& origin)
{
    _image.set_origin(origin);
}
void FlowImage::set_spacing(const Vec3d& spacing)
{
    _image.set_spacing(spacing);
}

const Vec3i& FlowImage::size() const
{
    assert(_image.valid());
    return _image.size();
}
const Vec3d& FlowImage::origin() const
{
    return _image.origin();
}
const Vec3d& FlowImage::spacing() const
{
    return _image.spacing();
}
int FlowImage::ndims() const
{
    assert(_image.valid());
    return _image.ndims();
}
int FlowImage::pixel_type() const
{
    return _image.pixel_type();
}
Image& FlowImage::image()
{
    return _image;
}
const Image& FlowImage::image() const
{
    return _image;
}

FlowImage::FlowImage(const FlowImage& other)
{
    //if (other._image.valid())
    //{
    //    // TODO: Clone or just keep copy?
    //    _image = other._image.clone();
    //}
    _image = other._image;
}
FlowImage& FlowImage::operator=(const FlowImage& other)
{
    //if (other._image.valid())
    //{
    //    _image = other._image.clone();
    //}
    _image = other._image;
    return *this;
}
int FlowImage::script_object_init(PyObject*, PyObject* args, PyObject* /*kwds*/)
{
    PyObject* arr = nullptr;
    PyObject* hint_str = nullptr;
    if (PyArg_ParseTuple(args, "|OS:__init__", &arr, &hint_str))
    {
        if (arr)
        {
            int hint = image::PixelType_Unknown;
            if (hint_str)
            {
                hint = image::string_to_pixel_type(PyString_AsString(hint_str));
            }
            else
            {
                PyErr_SetString(PyExc_AttributeError, "Pixel type expected.");
                return -1;
            }

            if (!numpy::read_array(this, arr, hint))
            {
                return -1;
            }
        }
    }
    else
    {
        return -1;
    }
    return 0;
}