#include <Core/Common.h>
#include <Core/Flow/FlowSystem.h>

#include "ImiomicsModule.h"
#include "SubjectNode.h"
#include "ConstraintNode.h"

IMPLEMENT_MODULE(ImiomicsModule);

ImiomicsModule::ImiomicsModule()
{
}
ImiomicsModule::~ImiomicsModule()
{
}

void ImiomicsModule::install()
{
    install_nodes();
}
void ImiomicsModule::uninstall()
{
}
void ImiomicsModule::install_nodes()
{
    FlowSystem::get().install_template(new SubjectNode(&_database));
    FlowSystem::get().install_template(new SubjectPathsNode(&_database));
    FlowSystem::get().install_template(new ConstraintNode(&_database));
    FlowSystem::get().install_template(new ConstraintPathsNode(&_database));
}
