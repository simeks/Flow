#include <Core/Common.h>

#include <Core/Flow/FlowContext.h>
#include <Core/Flow/FlowImage.h>
#include <Core/Image/Image.h>

#include "BlockedGraphCutOptimizer.h"
#include "RegistrationEngine.h"
#include "RegistrationNode.h"

IMPLEMENT_OBJECT(RegistrationNode, "RegistrationNode");

template<typename TImage>
static ImageVec3d exec_registration(const RegistrationEngine::Params& params)
{
    BlockedGraphCutOptimizer<TImage> optimizer;
    RegistrationEngine engine(&optimizer);
    return engine.execute(params);
}

RegistrationNode::RegistrationNode()
{
    add_pin("Fixed0", FlowPin::In);
    add_pin("Moving0", FlowPin::In);
    add_pin("Deformation", FlowPin::Out);

    add_pin("Fixed1", FlowPin::In);
    add_pin("Moving1", FlowPin::In);

    add_pin("Fixed2", FlowPin::In);
    add_pin("Moving2", FlowPin::In);

    add_pin("ConstraintMask", FlowPin::In);
    add_pin("ConstraintValues", FlowPin::In);

    add_pin("StartingGuess", FlowPin::In);

}
RegistrationNode::~RegistrationNode()
{
}
void RegistrationNode::run(FlowContext& context)
{
    RegistrationEngine::Params params;
    int pixel_type = image::PixelType_Unknown;

    for (int i = 0; i < 3; ++i)
    {
        std::stringstream ss; 
        
        ss << "Fixed" << i;
        FlowImage* fixed = context.read_pin<FlowImage>(ss.str());
        ss.str("");

        ss << "Moving" << i;
        FlowImage* moving = context.read_pin<FlowImage>(ss.str());

        if (!fixed || !moving)
            continue;

        Image fixed_img = fixed->image();
        Image moving_img = moving->image();

        if (!fixed_img || !moving_img)
            FATAL_ERROR("Not valid images!");

        if (fixed_img.pixel_type() != moving_img.pixel_type() ||
            fixed_img.size() != moving_img.size())
            FATAL_ERROR("Both images needs to be of same type and size.");

        switch (fixed_img.pixel_type())
        {
        case image::PixelType_UInt8:
        case image::PixelType_UInt16:
        case image::PixelType_UInt32:
        case image::PixelType_Float32:
            if (pixel_type == image::PixelType_Unknown)
                pixel_type = image::PixelType_Float32;

            // Convert if necessary
            params.fixed.push_back(ImageFloat32(fixed_img)); 
            params.moving.push_back(ImageFloat32(moving_img));

            break;
        case image::PixelType_Float64:
            if (pixel_type == image::PixelType_Unknown)
                pixel_type = image::PixelType_Float64;

            // Convert if necessary
            params.fixed.push_back(ImageFloat64(fixed_img));
            params.moving.push_back(ImageFloat64(moving_img));

            break;
        case image::PixelType_Vec4u8:
        case image::PixelType_Vec4f:
            if (pixel_type == image::PixelType_Unknown)
                pixel_type = image::PixelType_Vec4f;

            // Convert if necessary
            params.fixed.push_back(ImageColorf(fixed_img));
            params.moving.push_back(ImageColorf(moving_img));

            break;
        case image::PixelType_Vec3f:
        case image::PixelType_Vec3d:
        default:
            FATAL_ERROR("RegistrationNode: Could not register image pair.");
        }
    }

    if (params.fixed.empty())
        FATAL_ERROR("Expects at least one valid image pair.");

    FlowImage* constraint_mask = context.read_pin<FlowImage>("ConstraintMask");
    if (constraint_mask)
    {
        Image mask_img = constraint_mask->image();
        if (mask_img.pixel_type() == image::PixelType_Float32 ||
            mask_img.pixel_type() == image::PixelType_Float64)
        {
            mask_img = mask_img.convert_to(image::PixelType_UInt8, 255.0, 0.0);
        }
        else if (mask_img.pixel_type() != image::PixelType_UInt8)
        {
            FATAL_ERROR("Constraint mask needs to be of UInt8 type.");
        }

        if (mask_img.size() != params.fixed.back().size())
            FATAL_ERROR("Constraint mask needs to be of same size as the source and reference images.");

        params.constraint_mask = mask_img;
    }

    FlowImage* constraint_values = context.read_pin<FlowImage>("ConstraintValues");
    if (constraint_values)
    {
        Image values_img = constraint_values->image();
        if (values_img.pixel_type() != image::PixelType_Vec3d)
            FATAL_ERROR("Constraint value image needs to be of Vec3d type.");

        if (values_img.size() != params.fixed.back().size())
            FATAL_ERROR("Constraint value image needs to be of same size as the source and reference images.");

        params.constraint_values = values_img;
    }

    FlowImage* starting_guess = context.read_pin<FlowImage>("StartingGuess");
    if (starting_guess)
    {
        Image guess_img = starting_guess->image();
        if (guess_img.pixel_type() != image::PixelType_Vec3d)
            FATAL_ERROR("Starting guess needs to be of Vec3d type.");

        if (guess_img.size() != params.fixed.back().size())
            FATAL_ERROR("Starting guess needs to be of same size as the source and reference images.");

        params.starting_guess = guess_img;
    }

    Image def;
    switch (pixel_type)
    {
    case image::PixelType_Float32:
        def = exec_registration<ImageFloat32>(params);
        break;
    case image::PixelType_Float64:
        def = exec_registration<ImageFloat64>(params);
        break;
    case image::PixelType_Vec4f:
        def = exec_registration<ImageColorf>(params);
        break;
    default:
        FATAL_ERROR("RegistrationNode: Could not register image pair.");
    }

    if (def)
    {
        FlowImage* result = new FlowImage(def);
        context.write_pin("Deformation", result);
    }
}
const char* RegistrationNode::title() const
{
    return "Registration";
}
const char* RegistrationNode::category() const
{
    return "Registration";
}
