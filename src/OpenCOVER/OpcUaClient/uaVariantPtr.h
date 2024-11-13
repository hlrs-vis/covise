#ifndef COVER_OPCUA_VARIANT_PTR_H
#define COVER_OPCUA_VARIANT_PTR_H
#include "export.h"
#include <open62541/types.h>
#include <memory>
namespace opencover{namespace opcua{

class OPCUACLIENTEXPORT UA_Variant_ptr{
public:
    UA_Variant_ptr();
    UA_Variant_ptr(UA_Variant*);
    UA_Variant *operator->();
    const UA_Variant *operator->() const;
    UA_Variant *get();
    const UA_Variant *get() const;
    UA_DateTime timestamp = 0;
private:
    std::shared_ptr<UA_Variant> m_ptr;
    struct OPCUACLIENTEXPORT CustomDeleter {
        void operator()(UA_Variant* v);
    };
};


}}

#endif // COVER_OPCUA_VARIANT_PTR_H
