#ifndef __IMIOMICS_MODULE_H__
#define __IMIOMICS_MODULE_H__

#include <Core/Modules/ModuleInterface.h>

#include "SubjectDatabase.h"

class ImiomicsModule : public ModuleInterface
{
public:
    ImiomicsModule();
    ~ImiomicsModule();

    void install() OVERRIDE;
    void uninstall() OVERRIDE;

private:
    void install_nodes();

    SubjectDatabase _database;
};

#endif // __IMIOMICS_MODULE_H__
