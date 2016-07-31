#ifndef __CORE_FLOW_IMAGE_H__
#define __CORE_FLOW_IMAGE_H__

#include "FlowObject.h"

#include <Core/Image/Image.h>

class CORE_API FlowImage : public FlowObject
{
    DECLARE_SCRIPT_OBJECT(FlowImage, FlowObject);
public:
    FlowImage();
    FlowImage(const Image& img);
    virtual ~FlowImage();

    void allocate(int type, const std::vector<uint32_t>& size);
    void set_image(const Image& img);
    void release_image();

    void set_origin(const Vec3d&);
    void set_spacing(const Vec3d&);

    const Vec3i& size() const;
    const Vec3d& origin() const;
    const Vec3d& spacing() const;
    int ndims() const;

    int pixel_type() const;

    Image& image();
    const Image& image() const;

    FlowImage(const FlowImage&);
    FlowImage& operator=(const FlowImage&);

private:
    int script_object_init(PyObject* /*self*/, PyObject* /*args*/, PyObject* /*kwds*/) OVERRIDE;

    Image _image;
};


#endif // __CORE_FLOW_IMAGE_H__
