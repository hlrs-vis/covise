#include "VrmlNodeTemplate.h"
#include "VrmlField.h"
#include "VrmlNode.h"
#include "VrmlScene.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <variant>

#include <cassert>

namespace vrml{

    //keep in line with VrmlField enum
    typedef std::tuple<
        VrmlSFBool,
        VrmlSFColor,
        VrmlSFColorRGBA,
        VrmlSFDouble,
        VrmlSFFloat,
        VrmlSFInt,
        VrmlSFRotation,
        VrmlSFTime,
        VrmlSFVec2d,
        VrmlSFVec3d,
        VrmlSFVec2f,
        VrmlSFVec3f,
        VrmlSFImage,
        VrmlSFString,
        VrmlMFBool,
        VrmlMFColor,
        VrmlMFColorRGBA,
        VrmlMFDouble,
        VrmlMFFloat,
        VrmlMFInt,
        VrmlMFRotation,
        VrmlMFString,
        VrmlMFTime,
        VrmlMFVec2d,
        VrmlMFVec3d,
        VrmlMFVec2f,
        VrmlMFVec3f,
        VrmlSFNode,
        VrmlMFNode,
        VrmlSFMatrix,
        VrmlField
    > VrmlTypesTuple;


template <typename VrmlType, typename Tuple, std::size_t Index = 0>
struct TypeToEnumHelper {
    static VrmlField::VrmlFieldType value() {
        if constexpr (Index < std::tuple_size_v<Tuple>) {
            if constexpr (std::is_same_v<VrmlType, std::tuple_element_t<Index, Tuple>>) {
                return static_cast<VrmlField::VrmlFieldType>(Index + 1);
            } else {
                return TypeToEnumHelper<VrmlType, Tuple, Index + 1>::value();
            }
        } else {
            return VrmlField::VrmlFieldType::NO_FIELD;
        }
    }
};

// Function to map VrmlType to enumeration value
template<typename VrmlType>
VrmlField::VrmlFieldType toEnumType(const VrmlType *t) {
    return TypeToEnumHelper<VrmlType, VrmlTypesTuple>::value();
}


// Helper to extract types from tuple and create a variant with pointers to those types
template<typename Tuple, std::size_t... Indices>
auto tuple_to_variant_ptr_impl(std::index_sequence<Indices...>) {
    return std::variant<std::add_pointer_t<std::tuple_element_t<Indices, Tuple>>...>{};
}

template<typename Tuple>
using tuple_to_variant_ptr = decltype(tuple_to_variant_ptr_impl<Tuple>(std::make_index_sequence<std::tuple_size_v<Tuple>>{}));



class VrmlNodeUpdateRegistry
{

public:
    VrmlNodeUpdateRegistry(VrmlNodeTemplate *nodeChild)
    : m_nodeChild(nodeChild)
    {}


private:
    //use pointers to avoid memory overhead for arrays
    
    using VrmlTypesVariant = tuple_to_variant_ptr<VrmlTypesTuple>;

    template<typename VrmlType>
    struct VrmlTypeStruct{
        VrmlType *value;
        bool initialized = false;
        VrmlNodeTemplate::FieldUpdateCallback<VrmlType> updateCb;
        // const VrmlType defaultValue;
    };


    template<typename Tuple, std::size_t... Indices>
    static std::variant<VrmlTypeStruct<std::tuple_element_t<Indices, Tuple>>...> tuple_to_VrmlTypeStruct_impl(std::index_sequence<Indices...>) {
        return std::variant<VrmlTypeStruct<std::tuple_element_t<Indices, Tuple>>...>{};
    }


    template<typename Tuple>
    using tuple_to_VrmlTypeStruct = decltype(tuple_to_VrmlTypeStruct_impl<Tuple>(std::make_index_sequence<std::tuple_size_v<Tuple>>{}));

    using VrmlTypeStructs = tuple_to_VrmlTypeStruct<VrmlTypesTuple>;
    
    std::map<std::string, VrmlTypeStructs> m_fields;
    VrmlNode *m_nodeChild;

public:
    const VrmlField *getField(const char *fieldName) const
    {
        auto it = m_fields.find(fieldName);
        if(it == m_fields.end()){
            return m_nodeChild->getField(fieldName);
        }
        const VrmlTypeStructs& field = it->second;
        
        return std::visit([](auto&& arg){
            return static_cast<const VrmlField*>(arg.value);
        }, field);
        return nullptr;
    }
    
    void setField(const char *fieldName, const VrmlField &fieldValue) {
        auto it = m_fields.find(fieldName);
        if(it == m_fields.end()){
            m_nodeChild->setField(fieldName, fieldValue);
            return;
        }
        auto& field = it->second;
        std::visit([fieldName, &fieldValue, this](auto&& arg){
            auto val = dynamic_cast<const std::remove_pointer_t<decltype(arg.value)>*>(&fieldValue);
            if(arg.value) //events do not have a field
            {
                if(!val){
                    System::the->error("Invalid VrmlType (%s) for %s field.\n",
                        fieldValue.fieldTypeName(), fieldName);
                    return;
                }
                *arg.value = *val;
            }
            arg.initialized = true;
            if(arg.updateCb){
                arg.updateCb(val);
            }

        }, field);
    }
    
    template<typename VrmlType>
    void registerField(const std::string& name, VrmlType *field, const VrmlNodeTemplate::FieldUpdateCallback<VrmlType> &updateCb = VrmlNodeTemplate::FieldUpdateCallback<VrmlType>{}){
        m_fields[name] = VrmlTypeStruct<VrmlType>{ field, false, updateCb};
    }

    bool initialized(const std::string& name){
        auto it = m_fields.find(name);
        if(it == m_fields.end()){
            return false;
        }
        std::visit([](auto&& arg){
            return arg.initialized;
        }, it->second);
        return false;
    }

    bool allInitialized(){
        bool retval = true;
        for(auto& [name, field] : m_fields){
            std::visit([&retval](auto&& arg){
                if(!arg.initialized){
                    retval = false;
                }
            }, field);
        }
        return retval;
    }

    template<typename VrmlType>
    VrmlType* copy(const VrmlType* other){
        return new VrmlType(*other);
    }

    // use new pointers with this inital values of other
    VrmlNodeUpdateRegistry(const VrmlNodeUpdateRegistry& other)
    : m_fields(other.m_fields)
    {
        for(auto& [name, field] : m_fields){
            std::visit([this, &field](auto&& arg){
                arg.value = copy(arg.value);
            }, field);
        }
    }

    template<typename VrmlType>
    void deleter(VrmlType* t){
        delete t;
    }
    std::ostream &printFields(std::ostream &os, int indent)
    {
        for(auto& [name, field] : m_fields){
            os << std::string(indent, ' ') << name << " : ";
            std::visit([&os](auto&& arg){
                os << *arg.value;
            }, field);
            os << std::endl;
        }
        return os;
    }
};

template<>
VrmlField* VrmlNodeUpdateRegistry::copy(const VrmlField* other){
    assert(!("can not copy abstract VrmlField"));
    return nullptr;
}

std::map<std::string, VrmlNodeTemplate::Constructors> VrmlNodeTemplate::m_constructors;

VrmlNodeTemplate::VrmlNodeTemplate(VrmlScene *scene, const std::string &name)
: VrmlNode(scene)
, m_constructor(m_constructors.find(name)) 
, m_impl(std::make_unique<VrmlNodeUpdateRegistry>(this)) {
    assert(m_constructor != m_constructors.end());
}

// use new pointers with this inital values of other
VrmlNodeTemplate::VrmlNodeTemplate(const VrmlNodeTemplate& other)
: VrmlNode(other)
, m_impl(std::make_unique<VrmlNodeUpdateRegistry>(*other.m_impl))
, m_constructor(other.m_constructor) {
    assert(m_constructor != m_constructors.end());
}

VrmlNodeTemplate::~VrmlNodeTemplate() = default;

vrml::VrmlNode *VrmlNodeTemplate::cloneMe() const
{
    return m_constructor->second.clone(this); 
}

vrml::VrmlNodeType *VrmlNodeTemplate::nodeType() const
{
    return m_constructor->second.creator; 
}

bool VrmlNodeTemplate::fieldInitialized(const std::string& name) const
{
    return m_impl->initialized(name);
}

bool VrmlNodeTemplate::allFieldsInitialized() const
{
    return m_impl->allInitialized();
}

void VrmlNodeTemplate::setField(const char *fieldName, const VrmlField &fieldValue) 
{
    m_impl->setField(fieldName, fieldValue);
}

std::ostream &VrmlNodeTemplate::printFields(std::ostream &os, int indent)
{
    return m_impl->printFields(os, indent);
}

const VrmlField *VrmlNodeTemplate::getField(const char *fieldName) const
{
    return m_impl->getField(fieldName);
}

void VrmlNodeTemplate::setFieldByName(const char *fieldName, const VrmlField &fieldValue)
{
    m_impl->setField(fieldName, fieldValue);
}


template<typename VrmlType>
void VrmlNodeTemplate::registerField(VrmlNodeTemplate *node, const std::string& name, VrmlType *field, const FieldUpdateCallback<VrmlType> &updateCb)
{
    return node->m_impl->registerField<VrmlType>(name, field, updateCb);
}

template <typename VrmlType>
void addField(VrmlNodeType *t, const std::string &name, VrmlType &field) {
    t->addField(name.c_str(), toEnumType<std::remove_reference_t<VrmlType>>());
}

template <typename VrmlType>
void addExposedField(VrmlNodeType *t, const std::string &name, VrmlType &field) {
    t->addExposedField(name.c_str(), toEnumType<std::remove_reference_t<VrmlType>>());
}

template <typename VrmlType>
void VrmlNodeTemplate::initFieldsHelperImpl(VrmlNodeTemplate *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::Private> &field)
{
    if (node) 
        registerField(node, field.name, field.value, field.updateCb);
    if (t) 
        addField(t, field.name, *field.value);
}

template <typename VrmlType>
void VrmlNodeTemplate::initFieldsHelperImpl(VrmlNodeTemplate *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::Exposed> &field)
{
    if (node) 
        registerField(node, field.name, field.value, field.updateCb);
    if (t) 
        addExposedField(t, field.name, *field.value);
}

template <typename VrmlType>
void VrmlNodeTemplate::initFieldsHelperImpl(VrmlNodeTemplate *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::EventIn> &field)
{
    if (node) 
        registerField(node, field.name, field.value, field.updateCb);
    if (t) 
        t->addEventIn(field.name.c_str(), toEnumType<VrmlType>());
}

template <typename VrmlType>
void VrmlNodeTemplate::initFieldsHelperImpl(VrmlNodeTemplate *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::EventOut> &field)
{
    if (node) 
        registerField(node, field.name, field.value, field.updateCb);
    if (t) 
        t->addEventOut(field.name.c_str(), toEnumType<VrmlType>());
}

#define VRMLNODECHILD2_TEMPLATE_IMPL(VrmlType) \
template void VRMLEXPORT VrmlNodeTemplate::registerField<VrmlType>(VrmlNodeTemplate *node, const std::string& name, VrmlType *field, const FieldUpdateCallback<VrmlType> &updateCb);
FOR_ALL_VRML_TYPES(VRMLNODECHILD2_TEMPLATE_IMPL)

#define TO_VRML_FIELD_TYPES_IMPL(VrmlType) \
template VrmlField::VrmlFieldType VRMLEXPORT toEnumType(const VrmlType *t);
FOR_ALL_VRML_TYPES(TO_VRML_FIELD_TYPES_IMPL)

#define INIT_FIELDS_HELPER_IMPL(VrmlType) \
template void VRMLEXPORT VrmlNodeTemplate::initFieldsHelperImpl(VrmlNodeTemplate *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::Private> &field); 
FOR_ALL_VRML_TYPES(INIT_FIELDS_HELPER_IMPL)

#define INIT_EXPOSED_FIELDS_HELPER_IMPL(VrmlType) \
template void VRMLEXPORT VrmlNodeTemplate::initFieldsHelperImpl(VrmlNodeTemplate *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::Exposed> &field); 
FOR_ALL_VRML_TYPES(INIT_EXPOSED_FIELDS_HELPER_IMPL)

#define INIT_EVENT_IN_HELPER_IMPL(VrmlType) \
template void VRMLEXPORT VrmlNodeTemplate::initFieldsHelperImpl(VrmlNodeTemplate *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::EventIn> &field); 
FOR_ALL_VRML_TYPES(INIT_EVENT_IN_HELPER_IMPL)

#define INIT_EVENT_OUT_HELPER_IMPL(VrmlType) \
template void VRMLEXPORT VrmlNodeTemplate::initFieldsHelperImpl(VrmlNodeTemplate *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::EventOut> &field); 
FOR_ALL_VRML_TYPES(INIT_EVENT_OUT_HELPER_IMPL)


} // vrml

