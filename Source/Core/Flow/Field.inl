
template<typename T>
TField<T>::TField()
{
}
template<typename T>
TField<T>::TField(const char* name, size_t offset, uint32_t flags) : Field(name, offset, flags)
{
}

template<typename T>
T* TField<T>::value_ptr(const FlowObject* object) const
{
    return (T*)(((uint8_t*)object) + _offset);
}
template<typename T>
const T& TField<T>::value(const FlowObject* object) const
{
    return *value_ptr(object);
}
template<typename T>
void TField<T>::set_value(const FlowObject* object, const T& value) const
{
    *value_ptr(object) = value;
}
