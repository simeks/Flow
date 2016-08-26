#include <Core/Common.h>

#include <Core/Core.h>
#include <Core/Flow/FlowCompiler.h>
#include <Core/Flow/FlowContext.h>
#include <Core/Flow/FlowPrimitives.h>
#include <Core/Flow/FlowString.h>
#include <Core/Flow/FlowSystem.h>
#include <Core/Flow/TerminalNode.h>
#include <Core/Json/Json.h>
#include <Core/Json/JsonObject.h>
#include <Core/Platform/Timer.h>

class ArgParser
{
public:
    ArgParser(int argc, char** argv)
    {
        _executable = argv[0];

        std::vector<std::string> tokens;
        for (int i = 1; i < argc; ++i)
            tokens.push_back(argv[i]);

        while (!tokens.empty())
        {
            const std::string& token = tokens.back();
            
            if (token[0] == '-')
            {
                int b = 1;
                if (token[1] == '-')
                {
                    b = 2;
                }

                std::string line = token.substr(b);
                size_t p = line.find('=');
                if (p != std::string::npos)
                {
                    std::string key = line.substr(0, p);
                    std::string value = line.substr(p + 1);
                    _values[key] = value;
                }
                else
                {
                    _values[line] = "";
                }
            }
            else
            {
                _tokens.push_back(token);
            }
            tokens.pop_back();
        }
    }

    bool is_set(const std::string& key) const
    {
        return _values.find(key) != _values.end();
    }
    const std::string& value(const std::string& key) const
    {
        assert(is_set(key));
        return _values.at(key);
    }
    const std::map<std::string, std::string>& values() const
    {
        return _values;
    }

    const std::string& token(int i) const
    {
        return _tokens[i];
    }
    int num_tokens() const
    {
        return (int)_tokens.size();
    }
    const std::string& executable() const
    {
        return _executable;
    }

private:
    std::string _executable;
    std::map<std::string, std::string> _values;
    std::vector<std::string> _tokens;

};

class CommandLineApp
{
public:
    CommandLineApp(int argc, char** argv) : _args(argc, argv)
    {
        Core::create();
        Core::get().initialize(argc, argv);
    }
    ~CommandLineApp()
    {
        _graph.reset();
        Core::destroy();
    }
    int run();
    void print_usage() const;
private:
    bool load_graph(const std::string& file);
    void set_env_vars(FlowContext& context);

    ArgParser _args;
    FlowGraphPtr _graph;
};

int CommandLineApp::run()
{
    if (_args.num_tokens() < 1)
    {
        print_usage();
        return 1;
    }

    _graph = FlowSystem::get().create_graph();
    if (!load_graph(_args.token(0)))
    {
        console::print("Failed to load graph '%s'.\n", _args.token(0).c_str());
        return 1;
    }

    FlowContext context;
    FlowCompiler compiler;
    compiler.compile(context, _graph);

    double t_start = timer::seconds();

    set_env_vars(context);
    context.run();
    context.clean_up();

    console::print("Completed, Elapsed time: %f", timer::seconds() - t_start);

    return 0;
}
void CommandLineApp::print_usage() const
{
    console::print("Usage: %s <graph file> [--key=value, ...]\n", _args.executable().c_str());
}
bool CommandLineApp::load_graph(const std::string& file)
{
    _graph->clear();

    JsonObject root;

    JsonReader reader;
    if (!reader.read_file(file, root))
    {
        console::error("Failed to read graph file: %s.", reader.error_message().c_str());
        return false;
    }

    const JsonObject& nodes = root["nodes"];
    for (auto node_obj : nodes.as_array())
    {
        FlowNodePtr node = nullptr;

        std::string node_class = node_obj["node_class"].as_string();

        FlowNode* tpl = FlowSystem::get().node_template(node_class);
        if (!tpl)
            FATAL_ERROR("Failed to find node: '%s'.", node_class.c_str());
        node.reset((FlowNode*)tpl->clone());

        Guid node_id = guid::from_string(node_obj["id"].as_string());
        node->set_node_id(node_id);

        if (node->get_class() == TerminalNode::static_class())
        {
            TerminalNode* terminal_node = object_cast<TerminalNode>(node.get());
            terminal_node->set_var_name(node_obj["terminal_var_name"].as_string());

            std::string value = node_obj["terminal_value"].as_string();
            if (terminal_node->value()->is_a(FlowString::static_class()))
            {
                object_cast<FlowString>(terminal_node->value())->set(value);
            }
            else if (terminal_node->value()->is_a(NumericObject::static_class()))
            {
                NumericObject* num = object_cast<NumericObject>(terminal_node->value());
                if (value == "")
                {
                    num->set_int(0);
                }
                else
                {
                    if (num->is_integer())
                    {
                        num->set_int(std::stoi(value));
                    }
                    else
                    {
                        num->set_float(std::stof(value));
                    }
                }
            }
        }

        _graph->add_node(node);
    }

    const JsonObject& links = root["links"];
    for (auto link : links.as_array())
    {
        Guid out_id = guid::from_string(link["out_node"].as_string());
        Guid in_id = guid::from_string(link["in_node"].as_string());

        FlowNodePtr out_node = _graph->node(out_id);
        FlowNodePtr in_node = _graph->node(in_id);

        FlowPinPtr out_pin = out_node->pins().at(link["out_pin"].as_int());
        FlowPinPtr in_pin = in_node->pins().at(link["in_pin"].as_int());

        out_pin->link_to(in_pin.get());
        in_pin->link_to(out_pin.get());
    }

    return true;
}
void CommandLineApp::set_env_vars(FlowContext& context)
{
    for (auto& v : _args.values())
    {
        context.env_vars[v.first] = v.second;
    }
}

#ifndef DEBUG
void output_callback(void*, uint32_t , const char* msg)
{
    printf(msg);
}
#endif

int main(int argc, char** argv)
{
    timer::initialize();
    memory::initialize();
    int res = 0;
    {
#ifndef DEBUG
        console::set_callback(output_callback, nullptr);
#endif

        CommandLineApp app(argc, argv);
        res = app.run();
    }
    memory::shutdown();
    return res;
}
