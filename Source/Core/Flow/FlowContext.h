#ifndef __CORE_FLOW_CONTEXT_H__
#define __CORE_FLOW_CONTEXT_H__

#include "FlowGraph.h"
#include "FlowPin.h"

class CORE_API FlowContext
{
public:
    FlowContext();
    ~FlowContext();

    FlowObject* read_pin(int id) const;
    FlowObject* read_pin(const std::string& pin_name) const;
    FlowObject* read_pin(FlowPinPtr pin) const;

    /// Tries to read the specified pin and convert the result to a 32 bit integer.
    int64_t read_int(const std::string& pin_name);

    /// Tries to read the specified pin and convert the result to a double-precision floating-point value.
    double read_float(const std::string& pin_name);

    /// Tries to read the specified pin and convert the result to a string.
    const std::string& read_string(const std::string& pin_name);

    template<typename T>
    T* read_pin(const std::string& pin_name);

    void write_pin(int id, FlowObject* data);
    void write_pin(const std::string& pin_name, FlowObject* data);
    void write_pin(FlowPinPtr pin, FlowObject* data);

    /// Triggers an execution pin
    void exec_pin(const std::string& pin_name);
    void exec_pin(FlowPinPtr pin);

    /// Returns the environment variable with the specified name. 
    /// Returns empty string ("") if no such variable exists.
    std::string env_var(const std::string& key) const;

    void allocate_context(FlowGraphPtr graph);

    template<typename ObjectType>
    ObjectType* allocate_object();

    void run_node(FlowNode* node);

    void run();
    void clean_up();

    FlowGraph* graph;

    std::map<FlowPin*, FlowObject*> pin_data;

    std::vector<FlowObject*> objects;

    std::map<std::string, std::string> env_vars;

    FlowNode* active_node;
    PyObject* script_object;

    std::deque<FlowNode*> node_queue;
};

template<typename ObjectType>
ObjectType* FlowContext::allocate_object()
{
    ObjectType* obj = new ObjectType();
    objects.push_back(obj);
    return obj;
}

template<typename T>
T* FlowContext::read_pin(const std::string& pin_name)
{
    return object_cast<T>(read_pin(pin_name));
}


#endif // __CORE_FLOW_CONTEXT_H__
