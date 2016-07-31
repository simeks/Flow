#include <Core/Common.h>
#include <Core/Flow/FlowContext.h>
#include <Core/Flow/FlowImage.h>
#include <Core/Flow/FlowString.h>
#include <Core/Image/ITK.h>

#include "SubjectDatabase.h"
#include "SubjectNode.h"

IMPLEMENT_OBJECT(SubjectNode, "SubjectNode");

SubjectNode::SubjectNode(SubjectDatabase* db) :
    _subject_db(db)
{
    add_pin("Id", FlowPin::In);
    add_pin("FatImage", FlowPin::Out);
    add_pin("WaterImage", FlowPin::Out);
    add_pin("BodyMask", FlowPin::Out);
    add_pin("BodySFCM", FlowPin::Out);
}
SubjectNode::~SubjectNode()
{
}
void SubjectNode::run(FlowContext& context)
{
    FlowString* id = context.read_pin<FlowString>("Id");
    if (id)
    {
        assert(_subject_db);

        std::string db_file = context.env_var("subject_db_file");
        SubjectDatabaseScope db(*_subject_db, db_file);
        if (!db->is_open())
            FATAL_ERROR("Failed to open subject database '%s'.", db_file.c_str());

        SubjectPtr subject = db->get_subject(id->get());
        assert(subject->study == Subject::Study_POEM);
        if (subject)
        {
            FlowPinPtr fat_img_pin = pin("FatImage");
            if (fat_img_pin->is_linked())
            {
                Image img = image::load_image(subject->data["FilteredFat"]);
                if (img.valid())
                    context.write_pin(fat_img_pin, new FlowImage(img));
                else
                    console::error("Failed to load FilteredFat '%s'.", subject->data["FilteredFat"].c_str());
            }
            FlowPinPtr water_img_pin = pin("WaterImage");
            if (water_img_pin->is_linked())
            {
                Image img = image::load_image(subject->data["FilteredWat"]);
                if (img.valid())
                    context.write_pin(water_img_pin, new FlowImage(img));
                else
                    console::error("Failed to load FilteredWat '%s'.", subject->data["FilteredWat"].c_str());
            }
            FlowPinPtr body_mask_pin = pin("BodyMask");
            if (body_mask_pin->is_linked())
            {
                Image img = image::load_image(subject->data["BodyMask"]);
                if (img.valid())
                    context.write_pin(body_mask_pin, new FlowImage(img));
                else
                    console::error("Failed to load BodyMask '%s'.", subject->data["BodyMask"].c_str());
            }
            FlowPinPtr body_sfcm_pin = pin("BodySFCM");
            if (body_sfcm_pin->is_linked())
            {
                Image img = image::load_image(subject->data["BodySFCM"]);
                if (img.valid())
                    context.write_pin(body_sfcm_pin, new FlowImage(img));
                else
                    console::error("Failed to load BodySFCM '%s'.", subject->data["BodySFCM"].c_str());
            }
        }
    }
}
const char* SubjectNode::title() const
{
    return "Subject";
}
const char* SubjectNode::category() const
{
    return "Imiomics";
}

IMPLEMENT_OBJECT(SubjectPathsNode, "SubjectPathsNode");

SubjectPathsNode::SubjectPathsNode(SubjectDatabase* db) :
    _subject_db(db)
{
    add_pin("Id", FlowPin::In);
    add_pin("FatImage", FlowPin::Out);
    add_pin("WaterImage", FlowPin::Out);
    add_pin("BodyMask", FlowPin::Out);
    add_pin("BodySFCM", FlowPin::Out);
}
SubjectPathsNode::~SubjectPathsNode()
{
}
void SubjectPathsNode::run(FlowContext& context)
{
    FlowString* id = context.read_pin<FlowString>("Id");
    if (id)
    {
        SubjectPtr subject = _subject_db->get_subject(id->get());
        if (subject)
        {
            assert(subject->study == Subject::Study_POEM);
            FlowPinPtr fat_img_pin = pin("FatImage");
            if (fat_img_pin->is_linked())
            {
                context.write_pin(fat_img_pin, new FlowString(subject->data["FilteredFat"]));
            }
            FlowPinPtr water_img_pin = pin("WaterImage");
            if (water_img_pin->is_linked())
            {
                context.write_pin(water_img_pin, new FlowString(subject->data["FilteredWat"]));
            }
            FlowPinPtr body_mask_pin = pin("BodyMask");
            if (body_mask_pin->is_linked())
            {
                context.write_pin(body_mask_pin, new FlowString(subject->data["BodyMask"]));
            }
            FlowPinPtr body_sfcm_pin = pin("BodySFCM");
            if (body_sfcm_pin->is_linked())
            {
                context.write_pin(body_sfcm_pin, new FlowString(subject->data["BodySFCM"]));
            }
        }
    }
}
const char* SubjectPathsNode::title() const
{
    return "SubjectPath";
}
const char* SubjectPathsNode::category() const
{
    return "Imiomics";
}
