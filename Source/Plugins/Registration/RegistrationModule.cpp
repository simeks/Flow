#include <Core/Common.h>

#include <Core/Core.h>
#include <Core/Flow/FlowSystem.h>

#include "RegistrationModule.h"
#include "RegistrationNode.h"
#include "TransformNode.h"

IMPLEMENT_MODULE(RegistrationModule);

RegistrationModule::RegistrationModule()
{
}
RegistrationModule::~RegistrationModule()
{
}

void RegistrationModule::install()
{
    FlowSystem::get().install_template(new RegistrationNode());
    FlowSystem::get().install_template(new TransformNode());
}
void RegistrationModule::uninstall()
{
}
