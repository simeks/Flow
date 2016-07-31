#include "Common.h"

#include "DebugNodes.h"
#include "FlowContext.h"
#include "FlowNode.h"
#include "FlowPrimitives.h"
#include "FlowString.h"
#include "FlowSystem.h"

class ConsoleOutputNode : public FlowNode
{
    DECLARE_OBJECT(ConsoleOutputNode, FlowNode);
public:
    ConsoleOutputNode()
    {
        add_pin("In", FlowPin::In);
    }

    void run(FlowContext& context) OVERRIDE
    {
        FlowObject* obj = context.read_pin("In");
        if (obj)
        {
            console::print("Class: %s\n", obj->get_class()->name());
            console::print("Value: ");
            if (obj->is_a<FlowString>())
            {
                console::print(object_cast<FlowString>(obj)->get().c_str());
            }
            else if (obj->is_a<NumericObject>())
            {
                NumericObject* num = object_cast<NumericObject>(obj);
                if (num)
                {
                    if (num->is_integer())
                        console::print("%d", num->as_int());
                    else if (num->is_float())
                        console::print("%f", num->as_float());
                }
            }
            else
            {
                console::print("<Unknown>");
            }
            console::print("\n");
        }
    }
    const char* title() const OVERRIDE
    {
        return "ConsoleOutput";
    }
    const char* category() const OVERRIDE
    {
        return "Debug";
    }
};
IMPLEMENT_OBJECT(ConsoleOutputNode, "ConsoleOutputNode");

void flow_debug_nodes::install()
{
    FlowSystem::get().install_template(new ConsoleOutputNode());
}
