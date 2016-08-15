#include <Core/Common.h>
#include <Core/Flow/FlowSystem.h>

#include "ImageModule.h"
#include "ImageToWorld.h"

IMPLEMENT_MODULE(ImageModule);

ImageModule::ImageModule()
{
}
ImageModule::~ImageModule()
{
}

void ImageModule::install()
{
    install_nodes();
}
void ImageModule::uninstall()
{
}
void ImageModule::install_nodes()
{
    FlowSystem::get().install_template(new ImageSliceToWorldNode());
}
