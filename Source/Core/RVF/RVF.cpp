#include "Common.h"

#include "Flow/FlowImage.h"
#include "RVF.h"

#include <fstream>


Image rvf::load_rvf(const std::string& file)
{
    std::ifstream f;
    f.open(file, std::fstream::in | std::fstream::binary);

    if (!f.is_open())
        return Image();

    std::string line;
    std::getline(f, line);
    std::stringstream ss(line);

    uint32_t w, h, d;
    ss >> w >> h >> d;

    std::getline(f, line);
    ss = std::stringstream(line);

    double spacing_x, spacing_y, spacing_z;
    ss >> spacing_x >> spacing_y >> spacing_z;

    Image img;

    img.create(3, Vec3i(w, h, d), image::PixelType_Vec3d);
    img.set_spacing(Vec3d(spacing_x, spacing_y, spacing_z));

    uint32_t num_elems = w*h*d;
    f.read((char*)img.ptr(), num_elems * sizeof(Vec3d));

    f.close();

    return img;
}
bool rvf::save_rvf(const std::string& file, const ImageVec3d& img)
{
    assert(img.pixel_type() == image::PixelType_Vec3d);

    uint32_t w = img.size().x, h = img.size().y, d = img.size().z;
    uint32_t num_elems = w*h*d;
    
    Vec3d spacing = img.spacing();

    std::ofstream f;
    f.open(file, std::fstream::out | std::fstream::binary);
    if (!f.is_open())
        return false;

    f << w << " " << h << " " << d << std::endl;
    f << spacing.x << " " << spacing.y << " " << spacing.z << std::endl;
    f.write((const char*)img.ptr(), num_elems*sizeof(Vec3d));
    f.close();

    return true;
}

