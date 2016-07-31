#ifndef __CORE_FLOW_PRIMITIVES_H___
#define __CORE_FLOW_PRIMITIVES_H___

#include "FlowObject.h"

class CORE_API NumericObject : public FlowObject
{
    DECLARE_OBJECT(NumericObject, FlowObject);
public:
    NumericObject();
    virtual ~NumericObject();

    virtual bool is_integer() const { return false; }
    virtual bool is_float() const { return false; }

    virtual int64_t as_int() const { return 0; }
    virtual double as_float() const { return 0; }

    virtual void set_int(int64_t) {}
    virtual void set_float(double) {}

private:
    virtual PyObject* py_create_object();
};

template<typename NumType>
class NumericObjectHelper : public NumericObject
{
    static_assert(std::is_arithmetic<NumType>::value, "Requires a numeric type.");

public:
    typedef NumType ValueType;

    NumericObjectHelper();
    NumericObjectHelper(ValueType value);
    virtual ~NumericObjectHelper();

    bool is_integer() const { return std::is_integral<ValueType>::value; }
    bool is_float() const { return std::is_floating_point<ValueType>::value; }

    int64_t as_int() const { return (int64_t)_value; }
    double as_float() const { return (double)_value; }

    void set_int(int64_t v) { _value = (ValueType)v; }
    void set_float(double v) { _value = (ValueType)v; }

    ValueType get() const { return _value; }
    void set(ValueType v) { _value = v; }

private:
    ValueType _value;
};

template<typename Type>
NumericObjectHelper<Type>::NumericObjectHelper() : _value(0)
{
}

template<typename Type>
NumericObjectHelper<Type>::~NumericObjectHelper()
{
}
template<typename Type>
NumericObjectHelper<Type>::NumericObjectHelper(ValueType value) : _value(value)
{
}

#define DECLARE_NUMERIC_TYPE(Name, CppType) \
class CORE_API Name : public NumericObjectHelper<CppType> \
{ \
    DECLARE_OBJECT(Name, NumericObject); \
public: \
    Name(); \
    Name(CppType v); \
    ~Name(); \
};

#define IMPLEMENT_NUMERIC_TYPE(Name, CppType, NameStr) \
    IMPLEMENT_OBJECT(Name, NameStr) \
    Name::Name() \
    { \
    } \
    Name::Name(CppType v) : NumericObjectHelper(v) \
    { \
    } \
        Name::~Name() \
    { \
    }

DECLARE_NUMERIC_TYPE(FlowInt, int32_t);
DECLARE_NUMERIC_TYPE(FlowUInt32, uint32_t);
DECLARE_NUMERIC_TYPE(FlowInt64, int64_t);
DECLARE_NUMERIC_TYPE(FlowUInt64, uint64_t);
DECLARE_NUMERIC_TYPE(FlowFloat, float);
DECLARE_NUMERIC_TYPE(FlowFloat64, double);



#endif // __CORE_FLOW_PRIMITIVES_H___
