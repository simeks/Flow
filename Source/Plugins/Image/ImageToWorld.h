#ifndef __IMAGE_TO_WORLD_H__
#define __IMAGE_TO_WORLD_H__

#include <Core/Flow/FlowNode.h>

/// @brief Node for resampling an image from image space to world space.
class ImageSliceToWorldNode : public FlowNode
{
    DECLARE_OBJECT(ImageSliceToWorldNode, FlowNode);
public:
    ImageSliceToWorldNode();
    virtual ~ImageSliceToWorldNode();

    virtual void run(FlowContext& context) OVERRIDE;

    const char* title() const OVERRIDE;
    const char* category() const OVERRIDE;
};

#endif // __IMAGE_TO_WORLD_H__
