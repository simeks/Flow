#include <Core/Common.h>
#include <Core/Image/Image.h>
#include <Core/Image/Resample.h>
#include <Core/Image/ITK.h>

#include "Optimizer.h"
#include "RegistrationEngine.h"

template<typename TImage>
static TImage normalize_image(const TImage& image)
{
    double min, max;
    image::find_min_max(image, min, max);

    TImage result = image.clone();

    double d = max - min;
#pragma omp parallel for
    for (int64_t i = 0; i < (int64_t)result.pixel_count(); ++i)
    {
        result[i] = TImage::TPixelType((result[i] - min) / d);
    }
    return result;
}

static ImageUInt8 downsample_mask(const ImageUInt8& mask, double scale)
{
    assert(scale > 0.0 && scale < 1.0);
    double inv_scale = 1.0 / scale;

    std::vector<Vec3i> neighbors;

    neighbors.push_back(Vec3i(0, 0, 0));
    neighbors.push_back(Vec3i(1, 0, 0));
    neighbors.push_back(Vec3i(0, 1, 0));
    neighbors.push_back(Vec3i(0, 0, 1));
    neighbors.push_back(Vec3i(1, 0, 1));
    neighbors.push_back(Vec3i(1, 1, 0));
    neighbors.push_back(Vec3i(0, 1, 1));
    neighbors.push_back(Vec3i(1, 1, 1));

    Vec3i old_dims = mask.size();
    Vec3i new_dims(1, 1, 1);
    for (int i = 0; i < mask.ndims(); ++i)
    {
        new_dims[i] = (int)ceil(old_dims[i] * scale);
    }

    ImageUInt8 result(mask.ndims(), new_dims, (uint8_t)0);
    result.set_spacing(mask.spacing() * inv_scale);
    result.set_origin(mask.origin());

#pragma omp parallel for
    for (int z = 0; z < new_dims.z; ++z)
    {
        for (int y = 0; y < new_dims.y; ++y)
        {
            for (int x = 0; x < new_dims.x; ++x)
            {
                Vec3i v0 = inv_scale * Vec3i(x, y, z);
                uint8_t maxval = 0;
                for (int n = 0; n < neighbors.size(); n++)
                {
                    uint8_t tmpval = mask.at(v0 + neighbors[n], image::Border_Constant);
                    if (tmpval > maxval)
                        maxval = tmpval;
                }
                result(x, y, z) = maxval;
            }
        }
    }

    return result;
}

static ImageVec3d downsample_constraint_values(const ImageVec3d& values, const ImageUInt8& mask, double scale)
{
    assert(scale > 0.0 && scale < 1.0);
    double inv_scale = 1.0 / scale;

    std::vector<Vec3i> neighbors;

    neighbors.push_back(Vec3i(0, 0, 0));
    neighbors.push_back(Vec3i(1, 0, 0));
    neighbors.push_back(Vec3i(0, 1, 0));
    neighbors.push_back(Vec3i(0, 0, 1));
    neighbors.push_back(Vec3i(1, 0, 1));
    neighbors.push_back(Vec3i(1, 1, 0));
    neighbors.push_back(Vec3i(0, 1, 1));
    neighbors.push_back(Vec3i(1, 1, 1));

    Vec3i old_dims = values.size();
    Vec3i new_dims(1, 1, 1);
    for (int i = 0; i < values.ndims(); ++i)
    {
        new_dims[i] = (int)ceil(old_dims[i] * scale);
    }

    ImageVec3d result(values.ndims(), new_dims, Vec3d(0, 0, 0));
    result.set_spacing(values.spacing() * inv_scale);
    result.set_origin(values.origin());

#pragma omp parallel for
    for (int z = 0; z < new_dims.z; ++z)
    {
        for (int y = 0; y < new_dims.y; ++y)
        {
            for (int x = 0; x < new_dims.x; ++x)
            {
                Vec3i p = Vec3i(x, y, z);
                int nmask = 0;
                Vec3d val = Vec3d(0, 0, 0);
                for (auto& n : neighbors)
                {
                    if (mask.at(2 * p + n, image::Border_Constant) > 0)
                    {
                        nmask++;
                        val += values.at(2 * p + n, image::Border_Replicate);
                    }
                }

                if (nmask > 0)
                {
                    result(x, y, z) = 0.5*val / double(nmask);
                }
            }
        }
    }
    return result;
}


RegistrationEngine::RegistrationEngine(Optimizer* optimizer) :
    _optimizer(optimizer), 
    _pyramid_levels(6),
    _pyramid_max_level(0),
    _image_pair_count(0),
    _normalize_images(true)
{
}
RegistrationEngine::~RegistrationEngine()
{
}

ImageVec3d RegistrationEngine::execute(const Params& params)
{
    _fixed_pyramid.clear();
    _moving_pyramid.clear();
    _deformation_pyramid.clear();
    _residual_pyramid.clear();
    _constraint_mask_pyramid.clear();
    _constraint_pyramid.clear();

    _image_pair_count = (int)params.fixed.size();
    if (_image_pair_count == 0)
        FATAL_ERROR("Invalid number of image pairs!");

    _fixed_pyramid.resize(_pyramid_levels);
    _moving_pyramid.resize(_pyramid_levels);
    _deformation_pyramid.resize(_pyramid_levels);
    _residual_pyramid.resize(_pyramid_levels-1);
    _constraint_mask_pyramid.resize(_pyramid_levels);
    _constraint_pyramid.resize(_pyramid_levels);

    for (int i = 0; i < _image_pair_count; ++i)
    {
        int pixel_type = params.fixed[i].pixel_type();

        if (params.fixed[i].pixel_type() != params.moving[i].pixel_type() ||
            params.fixed[i].size() != params.moving[i].size())
            FATAL_ERROR("Image pairs needs to have the same size and pixel type!");

        _fixed_pyramid[0].resize(_image_pair_count);
        _moving_pyramid[0].resize(_image_pair_count);

        if (_normalize_images && pixel_type == image::PixelType_Float32)
        {
            console::print("Normalizes image pair...\n");
            _fixed_pyramid[0][i] = normalize_image<ImageFloat32>(params.fixed[i]);
            _moving_pyramid[0][i] = normalize_image<ImageFloat32>(params.moving[i]);
        }
        else if (_normalize_images && pixel_type == image::PixelType_Float64)
        {
            console::print("Normalizes image pair...\n");
            _fixed_pyramid[0][i] = normalize_image<ImageFloat64>(params.fixed[i]);
            _moving_pyramid[0][i] = normalize_image<ImageFloat64>(params.moving[i]);
        }
        else
        {
            // We don't normalize non-float images
            _fixed_pyramid[0][i] = params.fixed[i];
            _moving_pyramid[0][i] = params.moving[i];
        }
    }

    if (params.starting_guess)
    {
        if (params.starting_guess.pixel_type() != image::PixelType_Vec3d)
            FATAL_ERROR("Starting guess needs to be of Vec3d type!");

        if (params.starting_guess.size() != _fixed_pyramid[0][0].size())
            FATAL_ERROR("Starting guess needs to be of same size as the image pair.");

        _deformation_pyramid[0] = params.starting_guess;
    }

    if (params.constraint_mask)
    {
        if (params.constraint_mask.pixel_type() != image::PixelType_UInt8)
            FATAL_ERROR("Constraint mask needs to be of UInt8 type!");

        if (params.constraint_mask.size() != _fixed_pyramid[0][0].size())
            FATAL_ERROR("Constraint mask needs to be of same size as the image pair.");

        _constraint_mask_pyramid[0] = params.constraint_mask;
    }
    if (params.constraint_values)
    {
        if (params.constraint_values.pixel_type() != image::PixelType_Vec3d)
            FATAL_ERROR("Constraint value image needs to be of Vec3d type!");

        if (params.constraint_values.size() != _fixed_pyramid[0][0].size())
            FATAL_ERROR("Constraint value image needs to be of same size as the image pair.");

        _constraint_pyramid[0] = params.constraint_values;
    }

    build_pyramid();

    for (int l = _pyramid_levels - 1; l >= 0; l--) 
    {
        if (l >= _pyramid_max_level) 
        {
            _optimizer->execute(&_fixed_pyramid[l][0], &_moving_pyramid[l][0], _image_pair_count, _constraint_mask_pyramid[l], _constraint_pyramid[l], _deformation_pyramid[l]);
        }

        if (l > 0) 
        {
            image::upsample_vectorfield(_deformation_pyramid[l], 2.0, _residual_pyramid[l - 1], _deformation_pyramid[l - 1]);
        }
    }
    return _deformation_pyramid[0];
}

void RegistrationEngine::build_pyramid()
{
    console::print("Building image pyramid... ");

    if (!_constraint_mask_pyramid[0].valid())
        _constraint_mask_pyramid[0] = ImageUInt8(_fixed_pyramid[0][0].ndims(), _fixed_pyramid[0][0].size(), uint8_t(0));

    for (int l = 0; l < _pyramid_levels - 1; ++l) 
    {
        _fixed_pyramid[l+1].resize(_image_pair_count);
        _moving_pyramid[l+1].resize(_image_pair_count);

        for (int i = 0; i < _image_pair_count; ++i)
        {
            double s = unit_sigma(_fixed_pyramid[l][i]);
            _fixed_pyramid[l + 1][i] = image::downsample_image_gaussian(_fixed_pyramid[l][i], 0.5, s);

            s = unit_sigma(_moving_pyramid[l][i]);
            _moving_pyramid[l + 1][i] = image::downsample_image_gaussian(_moving_pyramid[l][i], 0.5, s);
        }

        if (_constraint_mask_pyramid[l])
            _constraint_mask_pyramid[l + 1] = downsample_mask(_constraint_mask_pyramid[l], 0.5);

        if (_constraint_pyramid[l])
            _constraint_pyramid[l + 1] = downsample_constraint_values(_constraint_pyramid[l], _constraint_mask_pyramid[l], 0.5);
    }

    if (!_deformation_pyramid[0].valid()) // Has no starting guess
        _deformation_pyramid[0] = ImageVec3d(_fixed_pyramid[0][0].ndims(), _fixed_pyramid[0][0].size(), Vec3d(0, 0, 0));

    for (int l = 0; l < _pyramid_levels-1; l++)
    {
        _deformation_pyramid[l + 1] = image::downsample_vectorfield(_deformation_pyramid[l], 0.5, _residual_pyramid[l]);
    }

    console::print("Done.\n");
}
double RegistrationEngine::unit_sigma(const Image& img)
{
    Vec3d spacing = img.spacing();
    double min_spacing = spacing.x;
    min_spacing = std::min(min_spacing, spacing.y);
    min_spacing = std::min(min_spacing, spacing.z);
    return min_spacing;
}
