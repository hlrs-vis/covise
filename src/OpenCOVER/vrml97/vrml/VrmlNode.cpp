#include "coEventQueue.h"
#include "System.h"
#include "VrmlField.h"
#include "VrmlNamespace.h"
#include "VrmlNodeScript.h"
#include "VrmlScene.h"

#include <cassert>
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
    VrmlNodeUpdateRegistry(VrmlNode *nodeChild)
    : m_nodeChild(nodeChild)
    {
        
    }


private:
    struct VrmlTypeStructBase
    {
        VrmlTypeStructBase(const std::string &name)
        : name(name)
        {
        }
        virtual ~VrmlTypeStructBase() = default;
        virtual void setField(const VrmlField &fieldValue) = 0;
        virtual const VrmlField *getField() const = 0;
        virtual void print(std::ostream &os) const = 0;
        virtual std::unique_ptr<VrmlTypeStructBase> copy() const = 0;
        bool initialized = false;
        const std::string name;
    };
    template<typename VrmlType>
    struct VrmlTypeStruct : public VrmlTypeStructBase{
        VrmlTypeStruct(const std::string &name, VrmlType *value, const VrmlNode::FieldUpdateCallback<VrmlType> &updateCb)
        : VrmlTypeStructBase(name)
        , value(value)
        , updateCb(updateCb)
        {
        }
        void setField(const VrmlField &fieldValue) override {
            auto val = dynamic_cast<const VrmlType*>(&fieldValue);
            if(!val){
                System::the->error("Invalid VrmlType (%s) for %s field.\n",
                    fieldValue.fieldTypeName(), name);
                return;
            }
            *value = *val;
            initialized = true;
            if(updateCb){
                updateCb(val);
            }
        }

        const VrmlField *getField() const override {
            return value;
        }

        void print(std::ostream &os) const override {
            os << *value;
        }

        std::unique_ptr<VrmlTypeStructBase> copy() const override {
            return std::make_unique<VrmlTypeStruct<VrmlType>>(*this);
        }
        //use pointers to avoid memory overhead for arrays
        VrmlType *value;
        VrmlNode::FieldUpdateCallback<VrmlType> updateCb;
        // const VrmlType defaultValue;
    };

    std::vector<std::unique_ptr<VrmlTypeStructBase>> m_fields;
    VrmlNode *m_nodeChild;
    std::function<void(const char *, const VrmlField &)> m_setFieldFunc;

public:
    const VrmlField *getField(const char *fieldName) const
    {
        auto it = std::find_if(m_fields.begin(), m_fields.end(), [fieldName](const auto& f){
            return f->name == fieldName;
        });
        if(it == m_fields.end()){
            return nullptr;
        }
        return it->get()->getField();
    }
    
    inline void setField(const char *fieldName, const VrmlField &fieldValue) {
        auto it = std::find_if(m_fields.begin(), m_fields.end(), [fieldName](const auto& f){
            return f->name == fieldName;
        });
        if(it == m_fields.end()){
            return;
        }
        (*it)->setField(fieldValue);
    }
    
    template<typename VrmlType>
    void registerField(const std::string& name, VrmlType *field, const VrmlNode::FieldUpdateCallback<VrmlType> &updateCb = VrmlNode::FieldUpdateCallback<VrmlType>{}){
        m_fields.push_back(std::make_unique<VrmlTypeStruct<VrmlType>>(name, field, updateCb));
    }

    bool initialized(const std::string& name){
        auto it = std::find_if(m_fields.begin(), m_fields.end(), [&name](const auto& f){
            return f->name == name;
        });
        if(it == m_fields.end()){
            return false;
        }
        return it->get()->initialized;
    }

    bool allInitialized(const std::vector<std::string>& exceptions){
        bool retval = true;
        for(auto& field : m_fields){
            if(std::find(exceptions.begin(), exceptions.end(), field->name) != exceptions.end()){
                continue;
            }
            if(!field.get()->initialized){
                retval = false;
            }
        }
        return retval;
    }

    VrmlNodeUpdateRegistry(const VrmlNodeUpdateRegistry& other)
    {
        for(auto& field : other.m_fields){
            m_fields.push_back(field->copy());
        }
    }

    std::ostream &printFields(std::ostream &os, int indent) const
    {
        for(auto& field : m_fields){
            os << std::string(indent, ' ') << field->name << " : ";
            field->print(os);
            os << std::endl;
        }
        return os;
    }
};


//VrmlNode
//--------------------------------------------------------------------------------------------------

std::map<std::string, VrmlNode::Constructors> VrmlNode::m_constructors;

std::ostream &operator<<(std::ostream &os, const VrmlNode &f)
{
    return f.print(os, 0);
}

void VrmlNode::initFields(VrmlNode *node, VrmlNodeType *t)
{
    initFieldsHelper(node, t,
                     exposedField("metadata", node->d_metadata));
}

VrmlNode::VrmlNode(VrmlScene *scene, const std::string &name)
: d_scene(scene)
, m_constructor(m_constructors.find(name)) 
, m_impl(std::make_unique<VrmlNodeUpdateRegistry>(this)) {
    assert(m_constructor != m_constructors.end());
}

// use new pointers with this inital values of other
VrmlNode::VrmlNode(const VrmlNode& other)
: d_modified(true)
, d_scene(other.d_scene)
, m_impl(std::make_unique<VrmlNodeUpdateRegistry>(*other.m_impl))
, m_constructor(other.m_constructor) {
    assert(m_constructor != m_constructors.end());
}

VrmlNode::~VrmlNode()
{
    d_refCount = -1;
    // Remove this node from the EventQueue Cache

    if (d_scene)
        d_scene->getIncomingSensorEventQueue()->removeNodeFromCache(this);

    // Remove the node's name (if any) from the map...
    if (!d_name.empty())
    {
        if (d_scene && d_scene->scope())
            d_scene->scope()->removeNodeName(this);
        d_name.clear();
    }
    //  if(d_myNamespace)
    //     d_myNamespace->removeNodeName(this);
    // Remove all routes from this node
    Route *r = d_routes;
    while (r)
    {
        Route *next = r->next();
        if (!d_isDeletedInline)
            delete r;
        r = next;
    }

    r = d_incomingRoutes;
    while (r)
    {
        Route *next = r->nextI();
        if (!d_isDeletedInline)
            delete r;
        r = next;
    }    
}

VrmlNodeList VrmlNode::nodeStack;

VrmlNode *VrmlNode::clone(VrmlNamespace *ns)
{
    if (isFlagSet())
    {
        VrmlNode *node = ns->findNode(name());
        if (node)
            return node;
    }

    VrmlNode *n = this->cloneMe();
    if (n)
    {
        if (*name())
            n->setName(name(), ns);
        setFlag();
        n->cloneChildren(ns);
    }
    return n;
}

vrml::VrmlNode *VrmlNode::cloneMe() const
{
    return m_constructor->second.clone(this); 
}

void VrmlNode::cloneChildren(VrmlNamespace *ns)
{
    if (d_metadata.get())
        d_metadata.set(d_metadata.get()->clone(ns));
}

// Copy the routes to nodes in the given namespace.

void VrmlNode::copyRoutes(VrmlNamespace *ns)
{
    const char *fromName = name();
    VrmlNode *fromNode = fromName ? ns->findNode(fromName) : 0;

    if (fromNode)
        for (Route *r = d_routes; r; r = r->next())
        {
            const char *toName = r->toNode()->name();
            VrmlNode *toNode = toName ? ns->findNode(toName) : 0;
            if (toNode)
                fromNode->addRoute(r->fromEventOut(), toNode, r->toEventIn());
        }
}

VrmlNode *VrmlNode::reference()
{
    ++d_refCount;
    return this;
}

void VrmlNode::dereference()
{
    if (--d_refCount == 0)
        delete this;
}

vrml::VrmlNodeType *VrmlNode::nodeType() const
{
    return m_constructor->second.creator; 
}

bool VrmlNode::fieldInitialized(const std::string& name) const
{
    return m_impl->initialized(name);
}

bool VrmlNode::fieldsInitialized(const std::vector<std::string>& names) const
{
    for(const auto &name : names){
        if(!m_impl->initialized(name)){
            return false;
        }
    }
    return true;
}

bool VrmlNode::allFieldsInitialized() const
{
    return m_impl->allInitialized({"metadata"});
}

void VrmlNode::setField(const char *fieldName, const VrmlField &fieldValue) 
{
    m_impl->setField(fieldName, fieldValue);
}

std::ostream &VrmlNode::printFields(std::ostream &os, int indent) const
{
    return m_impl->printFields(os, indent);
}

const VrmlField *VrmlNode::getField(const char *fieldName) const
{
    return m_impl->getField(fieldName);
}

// Set the name of the node. Some one else (the parser) needs
// to tell the scene about the name for use in USE/ROUTEs.

void VrmlNode::setName(const char *nodeName, VrmlNamespace *ns)
{
    if (nodeName && *nodeName)
    {
        d_name = nodeName;
        if (ns)
        {
            ns->addNodeName(this);
            d_myNamespace = ns;
        }
    }
    else
    {
        d_name.clear();
    }
}

VrmlNamespace *VrmlNode::getNamespace() const
{
    return d_myNamespace;
}

// Add to scene

void VrmlNode::addToScene(VrmlScene *scene, const char * /* relativeUrl */)
{
    d_scene = scene;
}

std::ostream &VrmlNode::print(std::ostream &os, int indent) const
{
    const char *nm = name();
    for (int i = 0; i < indent; ++i)
        os << ' ';

    if (nm && *nm)
        os << "DEF " << nm << " ";

    os << nodeType()->getName() << " { ";

    printFields(os, indent + INDENT_INCREMENT);

    os << " }";

    return os;
}

std::ostream &VrmlNode::printField(std::ostream &os,
                                   int indent,
                                   const char *name,
                                   const VrmlField &f)
{
    os << std::endl;
    for (int i = 0; i < indent; ++i)
        os << ' ';
    os << name << ' ' << f;
    return os;
}

// Dirty bit - indicates node needs to be revisited for rendering.

void VrmlNode::setModified()
{
    d_modified = true;
    if (d_scene)
        d_scene->setModified();
    if (d_metadata.get())
        d_metadata.get()->setModified();
    forceTraversal();
}

void VrmlNode::clearModified()
{
    d_modified = false;
}

bool VrmlNode::isModified() const
{
    return d_modified;
}

void VrmlNode::forceTraversal(bool once, int increment)
{
    //fprintf(stderr, "name=%s, once=%d, inc=%d\n", name(), int(once), increment);
    if (once)
    {
        if (d_traverseAtFrame < 0 || d_traverseAtFrame == System::the->frame())
            return;
        if (d_traverseAtFrame >= 0)
            d_traverseAtFrame = System::the->frame();
    }
    else
    {
        if (d_traverseAtFrame >= 0)
            d_traverseAtFrame = -increment;
        else
            d_traverseAtFrame -= increment;
    }

    for (ParentList::iterator it = parentList.begin();
         it != parentList.end();
         it++)
    {
        (*it)->forceTraversal(once, increment);
    }
}

void VrmlNode::decreaseTraversalForce(int num)
{
    if (d_traverseAtFrame >= 0)
        return;

    if (num == -1)
        num = -d_traverseAtFrame;

    d_traverseAtFrame += num;

    for (ParentList::iterator it = parentList.begin();
         it != parentList.end();
         it++)
    {
        (*it)->decreaseTraversalForce(num);
    }
}

int VrmlNode::getTraversalForce()
{
    if (d_traverseAtFrame < 0)
        return -d_traverseAtFrame;
    else
        return 0;
}

bool VrmlNode::haveToRender()
{
    return true;
    return (d_traverseAtFrame < 0 || d_traverseAtFrame == System::the->frame());
}

void VrmlNode::clearFlags()
{
    d_flag = false;
}

// Add a route from an eventOut of this node to an eventIn of another node.

Route *VrmlNode::addRoute(const char *fromEventOut,
                          VrmlNode *toNode,
                          const char *toEventIn)
{
#ifdef DEBUG
    fprintf(stderr, "%s::%s 0x%x addRoute %s\n",
            nodeType()->getName(), name(),
            (unsigned)this, fromEventOut);
#endif

    // Check to make sure fromEventOut and toEventIn are valid names...

    // Is this route already here?
    Route *r;
    for (r = d_routes; r; r = r->next())
    {
        if (toNode == r->toNode() && strcmp(fromEventOut, r->fromEventOut()) == 0 && strcmp(toEventIn, r->toEventIn()) == 0)
            return r; // Ignore duplicate routes
    }

    // Add route
    r = new Route(fromEventOut, toNode, toEventIn, this);
    if (d_routes)
    {
        r->setNext(d_routes);
        d_routes->setPrev(r);
    }
    d_routes = r;
    return r;
}

// Remove a route entry if it is in one of the lists.

void VrmlNode::removeRoute(Route *ir)
{
    Route *r;
    for (r = d_incomingRoutes; r; r = r->nextI())
    {
        if (r == ir)
        {
            if (r->prevI())
                r->prevI()->setNextI(r->nextI());
            else if (d_incomingRoutes == r)
                d_incomingRoutes = r->nextI();
            if (r->nextI())
                r->nextI()->setPrevI(r->prevI());
            break;
        }
    }

    for (r = d_routes; r; r = r->next())
    {
        if (r == ir)
        {
            if (r->prev())
                r->prev()->setNext(r->next());
            else if (d_routes == r)
                d_routes = r->next();
            if (r->next())
                r->next()->setPrev(r->prev());
            break;
        }
    }
}

void VrmlNode::addRouteI(Route *newr)
{
    Route *r;
    for (r = d_incomingRoutes; r; r = r->nextI())
    {
        if (r == newr)
            return; // Ignore duplicate routes
    }

    // Add route
    if (d_incomingRoutes)
    {
        newr->setNextI(d_incomingRoutes);
        d_incomingRoutes->setPrevI(newr);
    }
    d_incomingRoutes = newr;
}

// Remove a route from an eventOut of this node to an eventIn of another node.
void VrmlNode::deleteRoute(const char *fromEventOut,
                           VrmlNode *toNode,
                           const char *toEventIn)
{
    Route *r;
    for (r = d_routes; r; r = r->next())
    {
        if (toNode == r->toNode() && strcmp(fromEventOut, r->fromEventOut()) == 0 && strcmp(toEventIn, r->toEventIn()) == 0)
        {
            if (r->prev())
                r->prev()->setNext(r->next());
            else if (d_routes == r)
                d_routes = r->next();
            if (r->next())
                r->next()->setPrev(r->prev());
            delete r;
            break;
        }
    }
}

void VrmlNode::repairRoutes()
{
    Route *r;
    Route *routeToDelete = NULL;
    for (r = d_incomingRoutes; r; r = r->nextI())
    {
        VrmlNode *newFromNode = r->newFromNode();
        if (newFromNode != NULL)
        {
            routeToDelete = r;
            r->newFromRoute(newFromNode);
        }
    }
    if (routeToDelete != NULL)
        removeRoute(routeToDelete);

    routeToDelete = NULL;
    for (r = d_routes; r; r = r->next())
    {
        VrmlNode *newToNode = r->newToNode();
        if (newToNode != NULL)
        {
            routeToDelete = r;
            r->newToRoute(newToNode);
        }
    }
    if (routeToDelete != NULL)
        removeRoute(routeToDelete);
}

// Pass a named event to this node.

void VrmlNode::eventIn(double timeStamp,
                       const char *eventName,
                       const VrmlField *fieldValue)
{
#ifdef DEBUG
    cout << "eventIn "
         << nodeType()->getName()
         << "::"
         << (name() ? name() : "")
         << "."
         << eventName
         << " "
         << *fieldValue
         << endl;
#endif

    // Strip set_ prefix
    const char *origEventName = eventName;
    if (strncmp(eventName, "set_", 4) == 0)
        eventName += 4;

    // Handle exposedFields
    if (nodeType()->hasExposedField(eventName))
    {
        setField(eventName, *fieldValue);
        char eventOutName[256];
        sprintf(eventOutName, "%s_changed", eventName);
        eventOut(timeStamp, eventOutName, *fieldValue);
        setModified();
    }

    // Handle set_field eventIn/field
    else if (nodeType()->hasEventIn(eventName) || (nodeType()->hasEventIn(origEventName) && nodeType()->hasField(eventName)))
    {
        setField(eventName, *fieldValue);
        setModified();
    }
    else if (auto scriptNode = as<VrmlNodeScript>())
    {
        if (scriptNode->hasExposedField(eventName))
        {
            setField(eventName, *fieldValue);
            char eventOutName[256];
            sprintf(eventOutName, "%s_changed", eventName);
            eventOut(timeStamp, eventOutName, *fieldValue);
            setModified();
        }
        else if (scriptNode->hasEventIn(origEventName))
        {
            setField(eventName, *fieldValue);
            setModified();
        }
        else
            std::cerr << "Error: unhandled eventIn " << nodeType()->getName()
                 << "::" << name() << "." << origEventName << std::endl;
    }

    else
        std::cerr << "Error: unhandled eventIn " << nodeType()->getName()
             << "::" << name() << "." << origEventName << std::endl;
}

// Retrieve a named eventOut/exposedField value.

const VrmlField *VrmlNode::getEventOut(const char *fieldName) const
{
    // Strip _changed prefix
    char shortName[256];
    int rootLen = (int)(strlen(fieldName) - strlen("_changed"));
    if (rootLen >= (int)sizeof(shortName))
        rootLen = sizeof(shortName) - 1;

    if (rootLen > 0 && strcmp(fieldName + rootLen, "_changed") == 0)
    {
        strncpy(shortName, fieldName, rootLen);
        shortName[rootLen] = '\0';
    }
    else
    {
        strncpy(shortName, fieldName, sizeof(shortName));
        shortName[255] = '\0';
    }
    const VrmlNodeScript *scriptNode;
    if ((scriptNode = as<VrmlNodeScript>()))
    {
        // Handle exposedFields
        if (scriptNode->hasExposedField(shortName))
            return getField(shortName);
        else if (scriptNode->hasEventOut(fieldName))
            return getField(fieldName);
        else if (scriptNode->hasEventOut(shortName))
            return getField(shortName);
        return 0;
    }

    // Handle exposedFields
    if (nodeType()->hasExposedField(shortName))
        return getField(shortName);
    else if (nodeType()->hasEventOut(fieldName))
        return getField(fieldName);
    else if (nodeType()->hasEventOut(shortName))
        return getField(shortName);
    return 0;
}

void VrmlNode::render(Viewer *)
{
    clearModified();
}

// Accumulate transformations for proper rendering of bindable nodes.
void VrmlNode::accumulateTransform(VrmlNode *)
{
    // Do nothing by default
}

VrmlNode *VrmlNode::getParentTransform() { return nullptr; }

void VrmlNode::inverseTransform(Viewer *v)
{
    VrmlNode *parentTransform = getParentTransform();
    if (parentTransform)
        parentTransform->inverseTransform(v);
}

void VrmlNode::inverseTransform(double *m)
{
    VrmlNode *parentTransform = getParentTransform();
    if (parentTransform)
        parentTransform->inverseTransform(m);
    else
        Midentity(m);
}

bool VrmlNode::isOnlyGeometry() const
{
    if (d_routes || d_incomingRoutes)
        return false;

    if (strstr(name(), "NotCached") != NULL || strstr(name(), "NoCache") != NULL)
        return false;

	if (strstr(name(), "coMirror") != NULL)
		return false;
    
    return true;
}

// Send a named event from this node.

void VrmlNode::eventOut(double timeStamp,
                        const char *eventOut,
                        const VrmlField &fieldValue)
{
#ifdef DEBUG
    fprintf(stderr, "%s::%s 0x%x eventOut %s\n",
            nodeType()->getName(), name(),
            (unsigned)this, eventOut);
#endif

    // Find routes from this eventOut
    Route *r;
    for (r = d_routes; r; r = r->next())
    {
        if ((strcmp(eventOut, r->fromEventOut()) == 0) || ((strncmp(eventOut, r->fromEventOut(), strlen(eventOut)) == 0) && (strlen(r->fromEventOut()) > 8) && (strcmp(r->fromEventOut() + strlen(r->fromEventOut()) - 8, "_changed") == 0)))
        {
#ifdef DEBUG
            cerr << "  => "
                 << r->toNode()->nodeType()->getName()
                 << "::"
                 << r->toNode()->name()
                 << "."
                 << r->toEventIn()
                 << endl;
#endif
            VrmlField *eventValue = fieldValue.clone();
            d_scene->queueEvent(timeStamp, eventValue,
                                r->toNode(), r->toEventIn());
        }
    }
}

// true if this node is on the static node stack
bool VrmlNode::isOnStack(VrmlNode *node)
{
    VrmlNodeList::iterator i;

    for (i = nodeStack.begin(); i != nodeStack.end(); ++i)
        if (*i == node)
        {
            return true;
            break;
        }
    return false;
}

template<typename VrmlType>
void VrmlNode::registerField(VrmlNode *node, const std::string& name, VrmlType *field, const FieldUpdateCallback<VrmlType> &updateCb)
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
void VrmlNode::initFieldsHelperImpl(VrmlNode *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::Private> &field)
{
    if (node) 
        registerField(node, field.name, field.value, field.updateCb);
    if (t) 
        addField(t, field.name, *field.value);
}

template <typename VrmlType>
void VrmlNode::initFieldsHelperImpl(VrmlNode *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::Exposed> &field)
{
    if (node) 
        registerField(node, field.name, field.value, field.updateCb);
    if (t) 
        addExposedField(t, field.name, *field.value);
}

template <typename VrmlType>
void VrmlNode::initFieldsHelperImpl(VrmlNode *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::EventIn> &field)
{
    if (node) 
        registerField(node, field.name, field.value, field.updateCb);
    if (t) 
        t->addEventIn(field.name.c_str(), toEnumType<VrmlType>());
}

template <typename VrmlType>
void VrmlNode::initFieldsHelperImpl(VrmlNode *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::EventOut> &field)
{
    if (node) 
        registerField(node, field.name, field.value, field.updateCb);
    if (t) 
        t->addEventOut(field.name.c_str(), toEnumType<VrmlType>());
}

#define VRMLNODECHILD2_TEMPLATE_IMPL(VrmlType) \
template void VRMLEXPORT VrmlNode::registerField<VrmlType>(VrmlNode *node, const std::string& name, VrmlType *field, const FieldUpdateCallback<VrmlType> &updateCb);
FOR_ALL_VRML_TYPES(VRMLNODECHILD2_TEMPLATE_IMPL)

#define TO_VRML_FIELD_TYPES_IMPL(VrmlType) \
template VrmlField::VrmlFieldType VRMLEXPORT toEnumType(const VrmlType *t);
FOR_ALL_VRML_TYPES(TO_VRML_FIELD_TYPES_IMPL)

#define INIT_FIELDS_HELPER_IMPL(VrmlType) \
template void VRMLEXPORT VrmlNode::initFieldsHelperImpl(VrmlNode *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::Private> &field); 
FOR_ALL_VRML_TYPES(INIT_FIELDS_HELPER_IMPL)

#define INIT_EXPOSED_FIELDS_HELPER_IMPL(VrmlType) \
template void VRMLEXPORT VrmlNode::initFieldsHelperImpl(VrmlNode *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::Exposed> &field); 
FOR_ALL_VRML_TYPES(INIT_EXPOSED_FIELDS_HELPER_IMPL)

#define INIT_EVENT_IN_HELPER_IMPL(VrmlType) \
template void VRMLEXPORT VrmlNode::initFieldsHelperImpl(VrmlNode *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::EventIn> &field); 
FOR_ALL_VRML_TYPES(INIT_EVENT_IN_HELPER_IMPL)

#define INIT_EVENT_OUT_HELPER_IMPL(VrmlType) \
template void VRMLEXPORT VrmlNode::initFieldsHelperImpl(VrmlNode *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::EventOut> &field); 
FOR_ALL_VRML_TYPES(INIT_EVENT_OUT_HELPER_IMPL)


} // vrml
