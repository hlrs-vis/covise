#include "uaVariantPtr.h"

using namespace opencover::opcua;

UA_Variant_ptr::UA_Variant_ptr() = default;


UA_Variant_ptr::UA_Variant_ptr(UA_Variant* v)
: m_ptr(UA_Variant_new(), CustomDeleter())
{
    UA_Variant_copy(v, m_ptr.get());
}

UA_Variant *UA_Variant_ptr::operator->()
{
    return m_ptr.get();
}

const UA_Variant *UA_Variant_ptr::operator->() const
{
    return m_ptr.get();
}

UA_Variant *UA_Variant_ptr::get()
{
    return m_ptr.get();
}

const UA_Variant *UA_Variant_ptr::get() const 
{
    return m_ptr.get();
}

void UA_Variant_ptr::CustomDeleter::operator()(UA_Variant* v)
{
    UA_Variant_delete(v);
}
