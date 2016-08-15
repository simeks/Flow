#ifndef __IMAGE_MODULE_H__
#define __IMAGE_MODULE_H__

#include <Core/Modules/ModuleInterface.h>

class ImageModule : public ModuleInterface
{
public:
    ImageModule();
    ~ImageModule();

    void install() OVERRIDE;
    void uninstall() OVERRIDE;

private:
    void install_nodes();

};

#endif // __IMAGE_MODULE_H__
