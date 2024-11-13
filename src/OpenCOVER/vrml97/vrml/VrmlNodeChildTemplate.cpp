#include "VrmlNodeChildTemplate.h"
#include "VrmlField.h"
#include "VrmlNode.h"
#include "VrmlScene.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <variant>



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
        VrmlSFMatrix
    > VrmlTypesTuple;


template <typename T, typename Tuple, std::size_t Index = 0>
struct TypeToEnumHelper {
    static VrmlField::VrmlFieldType value() {
        if constexpr (Index < std::tuple_size_v<Tuple>) {
            if constexpr (std::is_same_v<T, std::tuple_element_t<Index, Tuple>>) {
                return static_cast<VrmlField::VrmlFieldType>(Index + 1);
            } else {
                return TypeToEnumHelper<T, Tuple, Index + 1>::value();
            }
        } else {
            return VrmlField::VrmlFieldType::NO_FIELD;
        }
    }
};

// Function to map type to enumeration value
template<typename T>
VrmlField::VrmlFieldType toEnumType() {
    return TypeToEnumHelper<T, VrmlTypesTuple>::value();
}


// Helper to extract types from tuple and create a variant with pointers to those types
template<typename Tuple, std::size_t... Indices>
auto tuple_to_variant_ptr_impl(std::index_sequence<Indices...>) {
    return std::variant<std::add_pointer_t<std::tuple_element_t<Indices, Tuple>>...>{};
}

template<typename Tuple>
using tuple_to_variant_ptr = decltype(tuple_to_variant_ptr_impl<Tuple>(std::make_index_sequence<std::tuple_size_v<Tuple>>{}));


class VrmlNodeChildTemplateImpl
{

public:
    VrmlNodeChildTemplateImpl(VrmlNodeChildTemplate *nodeChild)
    : m_nodeChild(nodeChild)
    {}


private:
    //use pointers to avoid memory overhead for arrays
    
    using VrmlTypesVariant = tuple_to_variant_ptr<VrmlTypesTuple>;

    struct VrmlTypeStruct{
        VrmlTypesVariant type;
        bool initialized = false;
        std::function<void()> updateCb;
    };
    std::map<std::string, VrmlTypeStruct> m_fields;
    VrmlNode *m_nodeChild;
    template<typename T>
    const T* getField(const VrmlField &fieldValue, const T *t){
        const T* val = dynamic_cast<const T*>(&fieldValue);
        if(!val)
            std::cerr << "Field type mismatch" << std::endl;
        return val;
    }
    

public:
    void setField(const char *fieldName, const VrmlField &fieldValue) {
        auto it = m_fields.find(fieldName);
        if(it == m_fields.end()){
            m_nodeChild->setField(fieldName, fieldValue);
            return;
        }
        auto& field = it->second;
        std::visit([&fieldValue, this](auto&& arg){
            *arg = *getField(fieldValue, arg);
        }, field.type);
        field.initialized = true;
        if(field.updateCb){
            field.updateCb();
        }
    }
    template<typename T>
    T* registerField(const std::string& name, const std::function<void()> &updateCb =  std::function<void()>{}){
        auto val = new T;
        m_fields[name] = VrmlTypeStruct{ val, false, updateCb};
        return val;
    }


    bool initialized(const std::string& name){
        auto it = m_fields.find(name);
        if(it == m_fields.end()){
            return false;
        }
        return it->second.initialized;
    }

    bool allInitialized(){
        for(auto& [name, field] : m_fields){
            if(!field.initialized){
                return false;
            }
        }
        return true;
    }

    template<typename T>
    T* copy(const T* other){
        return new T(*other);
    }

    // use new pointers with this inital values of other
    VrmlNodeChildTemplateImpl(const VrmlNodeChildTemplateImpl& other)
    : m_fields(other.m_fields)
    {
        for(auto& [name, field] : m_fields){
            std::visit([this, &field](auto&& arg){
                arg = copy(arg);
            }, field.type);
        }
    }

    template<typename T>
    void deleter(T* t){
        delete t;
    }

    // class must be copyable -> don't use unique_ptr and delete manually
    ~VrmlNodeChildTemplateImpl() {
        for(auto& [name, field] : m_fields){
            std::visit([this](auto&& arg){
                deleter(arg);
            }, field.type);
        }
};
};


VrmlNodeChildTemplate::VrmlNodeChildTemplate(VrmlScene *scene)
: VrmlNode(scene)
, m_impl(std::make_unique<VrmlNodeChildTemplateImpl>(this)) {}

// use new pointers with this inital values of other
VrmlNodeChildTemplate::VrmlNodeChildTemplate(const VrmlNodeChildTemplate& other)
: VrmlNode(other)
, m_impl(std::make_unique<VrmlNodeChildTemplateImpl>(*other.m_impl)) {}

VrmlNodeChildTemplate::~VrmlNodeChildTemplate() = default;

bool VrmlNodeChildTemplate::initialized(const std::string& name) const
{
    return m_impl->initialized(name);
}

bool VrmlNodeChildTemplate::allInitialized() const
{
    return m_impl->allInitialized();
}

void VrmlNodeChildTemplate::setField(const char *fieldName, const VrmlField &fieldValue) 
{
    m_impl->setField(fieldName, fieldValue);
}

template<typename T>
T* VrmlNodeChildTemplate::registerField(const std::string& name, const std::function<void()> &updateCb)
{
    return m_impl->registerField<T>(name, updateCb);
}


#define VRMLNODECHILD2_TEMPLATE_IMPL(type) \
template type VRMLEXPORT *VrmlNodeChildTemplate::registerField<type>(const std::string&, const std::function<void()>&);

VRMLNODECHILD2_TEMPLATE_IMPL(VrmlSFBool)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlSFColor)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlSFColorRGBA)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlSFDouble)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlSFFloat)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlSFInt)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlSFRotation)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlSFTime)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlSFVec2d)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlSFVec3d)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlSFVec2f)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlSFVec3f)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlSFImage)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlSFString)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlMFBool)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlMFColor)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlMFColorRGBA)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlMFDouble)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlMFFloat)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlMFInt)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlMFRotation)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlMFString)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlMFTime)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlMFVec2d)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlMFVec3d)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlMFVec2f)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlMFVec3f)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlSFNode)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlMFNode)
VRMLNODECHILD2_TEMPLATE_IMPL(VrmlSFMatrix)

} // vrml

