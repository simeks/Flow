#include <Core/Common.h>
#include <Core/Flow/FlowContext.h>
#include <Core/Flow/FlowImage.h>
#include <Core/Flow/FlowString.h>
#include <Core/Image/ITK.h>
#include <Core/RVF/RVF.h>

#include "SubjectDatabase.h"
#include "ConstraintNode.h"

IMPLEMENT_OBJECT(ConstraintNode, "ConstraintNode");

ConstraintNode::ConstraintNode(SubjectDatabase* db) :
    _subject_db(db)
{
    add_pin("FixedId", FlowPin::In);
    add_pin("MovingId", FlowPin::In);
    add_pin("Mask", FlowPin::Out);
    add_pin("Value", FlowPin::Out);
}
ConstraintNode::~ConstraintNode()
{
}
void ConstraintNode::run(FlowContext& context)
{
    FlowString* fixed_id = context.read_pin<FlowString>("FixedId");
    FlowString* moving_id = context.read_pin<FlowString>("MovingId");
    if (fixed_id && moving_id)
    {
        std::string db_file = context.env_var("subject_db_file");
        SubjectDatabaseScope db(*_subject_db, db_file);
        if (!db->is_open())
            FATAL_ERROR("Failed to open subject database '%s'.", db_file.c_str());

        SubjectConstraintPtr constraint = db->get_constraint(fixed_id->get(), moving_id->get());
        if (constraint)
        {
            FlowPinPtr mask_pin = pin("Mask");
            if (mask_pin->is_linked())
            {
                Image img = image::load_image(constraint->data["ConstraintMask"]);
                if (img.valid())
                    context.write_pin(mask_pin, new FlowImage(img));
                else
                    console::error("Failed to load constraint mask '%s'.", constraint->data["ConstraintMask"].c_str());
            }
            FlowPinPtr value_pin = pin("Value");
            if (value_pin->is_linked())
            {
                Image img = rvf::load_rvf(constraint->data["ConstraintValue_RVF"]);
                if (img.valid())
                    context.write_pin(value_pin, new FlowImage(img));
                else
                    console::error("Failed to load constraint values '%s'.", constraint->data["ConstraintValue_RVF"].c_str());
            }
        }
    }
}
const char* ConstraintNode::title() const
{
    return "Constraint";
}
const char* ConstraintNode::category() const
{
    return "Imiomics";
}


IMPLEMENT_OBJECT(ConstraintPathsNode, "ConstraintPathsNode");

ConstraintPathsNode::ConstraintPathsNode(SubjectDatabase* db) :
    _subject_db(db)
{
    add_pin("FixedId", FlowPin::In);
    add_pin("MovingId", FlowPin::In);
    add_pin("Mask", FlowPin::Out);
    add_pin("Value", FlowPin::Out);
}
ConstraintPathsNode::~ConstraintPathsNode()
{
}
void ConstraintPathsNode::run(FlowContext& context)
{
    FlowString* fixed_id = context.read_pin<FlowString>("FixedId");
    FlowString* moving_id = context.read_pin<FlowString>("MovingId");
    if (fixed_id && moving_id)
    {
        SubjectConstraintPtr constraint = _subject_db->get_constraint(fixed_id->get(), moving_id->get());
        if (constraint)
        {
            FlowPinPtr mask_pin = pin("Mask");
            if (mask_pin->is_linked())
            {
                context.write_pin(mask_pin, new FlowString(constraint->data["ConstraintMask"]));
            }
            FlowPinPtr value_pin = pin("Value");
            if (value_pin->is_linked())
            {
                context.write_pin(value_pin, new FlowString(constraint->data["ConstraintValue_RVF"]));
            }
        }
    }
}
const char* ConstraintPathsNode::title() const
{
    return "ConstraintPath";
}
const char* ConstraintPathsNode::category() const
{
    return "Imiomics";
}
