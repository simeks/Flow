#include "Common.h"

#include "Convert.h"
#include "Image.h"

ImageData::ImageData(size_t s) : size(s)
{
    if (size)
        data = new uint8_t[size]();
}
ImageData::~ImageData()
{
    if (data)
        delete[] data;
    size = 0;
}

Image::Image() : 
    _data(nullptr),
    _data_ptr(nullptr),
    _ndims(0),
    _spacing(1, 1, 1),
    _pixel_type(image::PixelType_Unknown)
{
}
Image::Image(const std::vector<uint32_t>& size, int type, const uint8_t* data) :
    _data(nullptr),
    _data_ptr(nullptr),
    _ndims(0),
    _spacing(1, 1, 1),
    _pixel_type(image::PixelType_Unknown)
{
    create(size, type, data);
}
Image::Image(int ndims, const Vec3i& size, int type, const uint8_t* data) :
    _data(nullptr),
    _data_ptr(nullptr),
    _ndims(0),
    _spacing(1, 1, 1),
    _pixel_type(image::PixelType_Unknown)
{
    create(ndims, size, type, data);
}
Image::~Image()
{
}

void Image::create(const std::vector<uint32_t>& size, int type, const uint8_t* data)
{
    assert(size.size() <= 3);
    assert(type != image::PixelType_Unknown);

    release();

    int ndims = (int)size.size();
    Vec3i vsize;
    int i = 0;
    for (; i < size.size(); ++i)
        vsize[i] = size[i];
    for (; i < 3; ++i)
        vsize[i] = 1;

    if (_ndims != ndims || _size != vsize || _pixel_type != type || !_data)
    {
        _ndims = ndims;
        _size = vsize;

        _pixel_type = type;
        size_t num_bytes = _size.x * _size.y * _size.z * pixel_size();
        _data = std::make_shared<ImageData>(num_bytes);
        _data_ptr = _data->data;

        size_t elem_size = image::pixel_size(type);
        size_t total = elem_size;
        for (int i = 0; i < _ndims; ++i)
        {
            _step[i] = total;
            total *= _size[i];
        }
    }
    if (data)
    {
        size_t num_bytes = _size.x * _size.y * _size.z * pixel_size();
        memcpy(_data_ptr, data, num_bytes);
    }
}
void Image::create(int ndims, const Vec3i& size, int type, const uint8_t* data)
{
    assert(ndims <= 3);
    assert(type != image::PixelType_Unknown);

    if (_ndims != ndims || _size != size || _pixel_type != type || !_data)
    {
        release();

        _ndims = ndims;
        _size = size;
        _pixel_type = type;
        size_t num_bytes = _size.x * _size.y * _size.z * pixel_size();
        _data = std::make_shared<ImageData>(num_bytes);
        _data_ptr = _data->data;

        size_t elem_size = image::pixel_size(type);
        size_t total = elem_size;
        for (int i = 0; i < ndims; ++i)
        {
            _step[i] = total;
            total *= _size[i];
        }
    }

    if (data)
    {
        size_t num_bytes = _size.x * _size.y * _size.z * pixel_size();
        memcpy(_data_ptr, data, num_bytes);
    }
}
void Image::release()
{
    _pixel_type = image::PixelType_Unknown;
    _data_ptr = nullptr;
    _data.reset();
}

void Image::set_origin(const Vec3d& origin)
{
    _origin = origin;
}
void Image::set_spacing(const Vec3d& spacing)
{
    _spacing = spacing;
}

const Vec3i& Image::size() const
{
    return _size;
}
const Vec3d& Image::origin() const
{
    return _origin;
}
const Vec3d& Image::spacing() const
{
    return _spacing;
}
int Image::ndims() const
{
    return _ndims;
}

int Image::pixel_type() const
{
    return _pixel_type;
}
size_t Image::pixel_size() const
{
    return image::pixel_size(_pixel_type);
}
size_t Image::pixel_count() const
{
    return _size[0] * _size[1] * _size[2];
}

Image Image::clone() const
{
    Image img;
    img.create(_ndims, _size, _pixel_type);
    img.set_spacing(_spacing);
    img.set_origin(_origin);

    size_t num_bytes = _size.x * _size.y * _size.z * pixel_size();
    // TODO: Stepping
    memcpy(img._data_ptr, _data_ptr, num_bytes);
    
    return img;
}

bool Image::valid() const
{
    return (_data_ptr != nullptr);
}

const uint8_t* Image::ptr() const
{
    assert(valid());
    return _data_ptr;
}
uint8_t* Image::ptr()
{
    assert(valid());
    return _data_ptr;
}
const size_t* Image::step() const
{
    assert(valid());
    return _step;
}
void Image::copy_to(uint8_t* dest) const
{
    assert(valid());
    size_t num_bytes = _size.x * _size.y * _size.z * pixel_size();
    // TODO: Stepping
    memcpy(dest, _data_ptr, num_bytes);
}
Image Image::convert_to(int type) const
{
    if (type == pixel_type())
    {
        return clone();
    }

    return image::convert_image(*this, type);
}
Image Image::convert_to(int type, double scale, double shift) const
{
    return image::convert_image(*this, type, scale, shift);
}


Image::Image(const Image& other) :
_data(other._data),
_data_ptr(other._data_ptr),
_ndims(other._ndims),
_size(other._size),
_origin(other._origin),
_spacing(other._spacing),
_pixel_type(other._pixel_type)
{
    _step[0] = other._step[0];
    _step[1] = other._step[1];
    _step[2] = other._step[2];
}
Image& Image::operator=(const Image& other)
{
    _data = other._data;
    _data_ptr = other._data_ptr;
    _ndims = other._ndims;
    _size = other._size;
    _origin = other._origin;
    _spacing = other._spacing;
    _pixel_type = other._pixel_type;

    _step[0] = other._step[0];
    _step[1] = other._step[1];
    _step[2] = other._step[2];

    return *this;
}

void image::find_min_max(const Image& img, double& min, double& max)
{
    min = std::numeric_limits<double>::max();
    max = -std::numeric_limits<double>::max();
    if (img.pixel_type() == image::PixelType_Float32)
    {
        ImageFloat32 m(img);

        // TODO: step
        for (size_t i = 0; i < m.pixel_count(); ++i)
        {
            min = std::min<double>(m[i], min);
            max = std::max<double>(m[i], max);
        }
    }
    else if (img.pixel_type() == image::PixelType_Float64)
    {
        ImageFloat64 m(img);

        // TODO: step
        for (size_t i = 0; i < m.pixel_count(); ++i)
        {
            min = std::min<double>(m[i], min);
            max = std::max<double>(m[i], max);
        }
    }
    else
    {
        FATAL_ERROR("Only float and double supported!");
    }
}
