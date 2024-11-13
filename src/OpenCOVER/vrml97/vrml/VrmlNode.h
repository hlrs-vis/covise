#ifndef COVER_VRMLFIELDTEMPLATE_H
#define COVER_VRMLFIELDTEMPLATE_H


#include <cassert>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "MathUtils.h"
#include "vrmlexport.h"
#include "VrmlField.h"
#include "VrmlRoute.h"

#include "VrmlMFBool.h"
#include "VrmlMFColor.h"
#include "VrmlMFColorRGBA.h"
#include "VrmlMFDouble.h"
#include "VrmlMFFloat.h"
#include "VrmlMFInt.h"
#include "VrmlMFNode.h"
#include "VrmlMFRotation.h"
#include "VrmlMFString.h"
#include "VrmlMFTime.h"
#include "VrmlMFVec2d.h"
#include "VrmlMFVec2f.h"
#include "VrmlMFVec3d.h"
#include "VrmlMFVec3f.h"
#include "VrmlNodeType.h"
#include "VrmlSFBool.h"
#include "VrmlSFColor.h"
#include "VrmlSFColorRGBA.h"
#include "VrmlSFDouble.h"
#include "VrmlSFFloat.h"
#include "VrmlSFImage.h"
#include "VrmlSFInt.h"
#include "VrmlSFMatrix.h"
#include "VrmlSFNode.h"
#include "VrmlSFRotation.h"
#include "VrmlSFString.h"
#include "VrmlSFTime.h"
#include "VrmlSFVec2d.h"
#include "VrmlSFVec2f.h"
#include "VrmlSFVec3d.h"
#include "VrmlSFVec3f.h"
namespace vrml{

class VrmlScene;
class Viewer;

class VrmlNodeUpdateRegistry;
class VRMLEXPORT VrmlNode
{
    friend class VrmlNodeUpdateRegistry;
    friend std::ostream &operator<<(std::ostream &os, const VrmlNode &f);   
public:

    static void initFields(VrmlNode *node, VrmlNodeType *t);

    typedef std::list<VrmlNode *> ParentList;
    ParentList parentList;

    virtual vrml::VrmlNodeType *nodeType() const;
    VrmlNode *clone(VrmlNamespace *); //eventuall virtual
    vrml::VrmlNode *cloneMe() const;
    virtual void cloneChildren(VrmlNamespace *);
    virtual void copyRoutes(VrmlNamespace *ns);

    // Add/remove references to a VrmlNode. This is silly, as it
    // requires the users of VrmlNode to do the reference/derefs...
    VrmlNode *reference();
    void dereference();

    //safe cast to derived class
    template<typename Derived>
    Derived *as() {
        return dynamic_cast<Derived*>(getThisProto());
    }
    template<typename Derived>
    const Derived *as() const {
        return dynamic_cast<const Derived*>(getThisProto());
    }

    //check if the node is of any of the given types
    template<typename...Deriveds>
    bool is() const {
        return (... || dynamic_cast<const Deriveds*>(this));
    }

    template<typename Derived>
    static vrml::VrmlNodeType *defineType(vrml::VrmlNodeType *t = nullptr)
    {
        return defineType_impl<Derived>(t);
    }

    virtual ~VrmlNode();
    virtual std::ostream &printFields(std::ostream &os, int indent) const;
    virtual const VrmlField *getField(const char *fieldName) const;
    
    
    // Node DEF/USE/ROUTE name
    void setName(const char *nodeName, VrmlNamespace *ns = 0);
    inline const char *name() const
    {
        return d_name.c_str();
    };
    VrmlNamespace *getNamespace() const;

    // Add to a scene. A node can belong to at most one scene for now.
    // If it doesn't belong to a scene, it can't be rendered.
    virtual void addToScene(VrmlScene *, const char *relativeUrl);
    // Write self
    std::ostream &print(std::ostream &os, int indent) const;
    static std::ostream &printField(std::ostream &, int, const char *, const VrmlField &);
    // Indicate that the node state has changed, need to re-render
    void setModified();
    void clearModified();

    virtual bool isModified() const;
    void forceTraversal(bool once = true, int increment = 1);
    void decreaseTraversalForce(int num = -1);
    int getTraversalForce();
    bool haveToRender();

    // A generic flag (typically used to find USEd nodes).
    void setFlag()
    {
        d_flag = true;
    }
    virtual void clearFlags(); // Clear childrens flags too.
    bool isFlagSet()
    {
        return d_flag;
    }

    // Add a ROUTE from a field in this node
    Route *addRoute(const char *fromField, VrmlNode *toNode, const char *toField);

    void removeRoute(Route *ir);
    void addRouteI(Route *ir);

    // Delete a ROUTE from a field in this node
    void deleteRoute(const char *fromField, VrmlNode *toNode, const char *toField);

    void repairRoutes();

    // Pass a named event to this node. This method needs to be overridden
    // to support any node-specific eventIns behaviors, but exposedFields
    // (should be) handled here...
    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

    
    // Return an eventOut/exposedField value. Used by the script node
    // to access the node fields.
    const VrmlField *getEventOut(const char *fieldName) const;

 // Do nothing. Renderable nodes need to redefine this.
    virtual void render(Viewer *);

    // Do nothing. Grouping nodes need to redefine this.
    virtual void accumulateTransform(VrmlNode *);

    virtual VrmlNode *getParentTransform();

    // Compute an inverse transform (either render it or construct the matrix)
    virtual void inverseTransform(Viewer *);
    virtual void inverseTransform(double *mat);

    // return number of coordinates in Coordinate/CoordinateDouble node
    virtual int getNumberCoordinates()
    {
        return 0;
    }

    // search for a node with the name of a EXPORT (AS) command
    // either inside a Inline node or inside a node created at real time
    virtual VrmlNode *findInside(const char *)
    {
        return NULL;
    }

    VrmlScene *scene() const
    {
        return d_scene;
    }

    virtual bool isOnlyGeometry() const;

    bool fieldInitialized(const std::string& name) const;
    bool fieldsInitialized(const std::vector<std::string>& names) const;

    bool allFieldsInitialized() const; //skipps metadata

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);


private:
    int d_refCount = 0; // Number of active references
    std::string d_name;

    VrmlSFNode d_metadata = nullptr;
    std::unique_ptr<VrmlNodeUpdateRegistry> m_impl;
    struct Constructors{
        vrml::VrmlNodeType* creator;
        // std::function<vrml::VrmlNode*(const VrmlNode*)> clone;
        vrml::VrmlNode*(*clone)(const VrmlNode*);
    };
    static std::map<std::string, Constructors> m_constructors;
    const std::map<std::string, Constructors>::const_iterator m_constructor;


protected:
    static const unsigned int INDENT_INCREMENT = 4;

    virtual VrmlNode *getThisProto(){return this;}
    virtual const VrmlNode *getThisProto() const {return this;}
    
    // Send a named event from this node.
    void eventOut(double timeStamp,
                  const char *eventName,
                  const VrmlField &fieldValue);


    enum FieldAccessibility{
        Private, Exposed, EventIn, EventOut
    };
    template<typename VrmlType>
    using FieldUpdateCallback = std::function<void(const VrmlType*)>;

    template<typename VrmlType>
    static void registerField(VrmlNode *node, const std::string& name, VrmlType *field, const FieldUpdateCallback<VrmlType> &updateCb = FieldUpdateCallback<VrmlType>{});

    VrmlNode(VrmlScene *scene, const std::string &name);
    VrmlNode(const VrmlNode& other);
    
    bool isOnStack(VrmlNode *);

    // Scene this node belongs to
    VrmlScene *d_scene = nullptr;

    // True if a field changed since last render
    bool d_modified = false;
    bool d_flag = false;

    // Routes from this node (clean this up, add RouteList ...)
    Route *d_routes = nullptr;
    Route *d_incomingRoutes = nullptr;

    //
    VrmlNamespace *d_myNamespace = nullptr;
    // true if this node is on the static node stack
    typedef std::list<VrmlNode *> VrmlNodeList;
    static VrmlNodeList nodeStack;

    // <0:  render every frame
    // >=0: render at this very frame
    int d_traverseAtFrame = 0;

    bool d_isDeletedInline = false;

    template <typename VrmlType, FieldAccessibility FT>
    struct NameValueStruct {
        std::string name;
        VrmlType *value;
        FieldUpdateCallback<VrmlType> updateCb;
    };

#define FOR_ALL_FIELD_TYPES(code)\
    code(field, FieldAccessibility::Private)\
    code(exposedField, FieldAccessibility::Exposed)\
    code(eventInCallBack, FieldAccessibility::EventIn)\
    code(eventOutCallBack, FieldAccessibility::EventOut)\


#define VRML_NAME_VALUE_FIELD(Name, FieldType)\
    template<typename VrmlType>\
    static NameValueStruct<VrmlType, FieldType> Name(const std::string &name, VrmlType &value, const FieldUpdateCallback<VrmlType> &updateCb = FieldUpdateCallback<VrmlType>()) {\
        return NameValueStruct<VrmlType, FieldType>{name, &value, updateCb};\
    }\
    template<typename VrmlType, typename Lambda>\
    static NameValueStruct<VrmlType, FieldType> Name(const std::string &name, VrmlType &value, Lambda &&lambda) {\
        return NameValueStruct<VrmlType, FieldType>{name, &value, lambda};\
    }\

    template<typename VrmlType, typename Lambda>
    static NameValueStruct<VrmlType, FieldAccessibility::EventIn> eventInCallBack(const std::string &name, Lambda &&lambda) {
        return NameValueStruct<VrmlType, FieldAccessibility::EventIn>{name, nullptr, lambda};
    }

    template<typename VrmlType, typename Lambda>
    static NameValueStruct<VrmlType, FieldAccessibility::EventOut> eventOutCallBack(const std::string &name, Lambda &&lambda) {
        return NameValueStruct<VrmlType, FieldAccessibility::EventOut>{name, nullptr, lambda};
    }

    FOR_ALL_FIELD_TYPES(VRML_NAME_VALUE_FIELD)
private:
#define INIT_FIELDS_HELPER(Name, FieldType)\
    template <typename VrmlType>\
    static void initFieldsHelperImpl(VrmlNode *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldType> &field);\

    FOR_ALL_FIELD_TYPES(INIT_FIELDS_HELPER)

protected:
    template <typename...Args>
    static void initFieldsHelper(VrmlNode *node, VrmlNodeType *t, const Args&... fields) {
            (initFieldsHelperImpl(node, t, fields), ...);
    }

    //can be specialized for each node in the derived class header file
    //e.g. VrmlNodeVariant.h
    template<typename Derived>
    static VrmlNode *creator(vrml::VrmlScene *scene){
        auto node = new Derived(scene);
        initFields(node, nullptr);
        Derived::initFields(node, nullptr);
        return node;
    }

private:
    template<typename Derived>
    static vrml::VrmlNodeType *defineType_impl(vrml::VrmlNodeType *t = nullptr)
    {
        assert(Derived::name() != nullptr);
        assert(strcmp(Derived::name(), "") != 0);
        static VrmlNodeType *st = 0;
        if (!t)
        {
            if (st)
                return st; // Only define the type once.
            t = st = new VrmlNodeType(Derived::name(), creator<Derived>);
            Constructors cs;
            cs.creator = t;
            cs.clone = [](const VrmlNode *node){ 
                auto newNode = new Derived(*dynamic_cast<const Derived*>(node)); 
                initFields(newNode, nullptr);
                Derived::initFields(newNode, nullptr);
                return static_cast<vrml::VrmlNode*>(newNode);
            };
            m_constructors[Derived::name()] = cs;
        }

        initFields(nullptr, t);
        Derived::initFields(nullptr, t);

        return t;
    }


};

template<typename VrmlType>
VrmlField::VrmlFieldType toEnumType(const VrmlType *t = nullptr);

struct DummyType{};

#define FOR_ALL_VRML_TYPES(code)\
    code(VrmlSFBool)\
    code(VrmlSFColor)\
    code(VrmlSFColorRGBA)\
    code(VrmlSFDouble)\
    code(VrmlSFFloat)\
    code(VrmlSFInt)\
    code(VrmlSFRotation)\
    code(VrmlSFTime)\
    code(VrmlSFVec2d)\
    code(VrmlSFVec3d)\
    code(VrmlSFVec2f)\
    code(VrmlSFVec3f)\
    code(VrmlSFImage)\
    code(VrmlSFString)\
    code(VrmlMFBool)\
    code(VrmlMFColor)\
    code(VrmlMFColorRGBA)\
    code(VrmlMFDouble)\
    code(VrmlMFFloat)\
    code(VrmlMFInt)\
    code(VrmlMFRotation)\
    code(VrmlMFString)\
    code(VrmlMFTime)\
    code(VrmlMFVec2d)\
    code(VrmlMFVec3d)\
    code(VrmlMFVec2f)\
    code(VrmlMFVec3f)\
    code(VrmlSFNode)\
    code(VrmlMFNode)\
    code(VrmlSFMatrix)\
    code(VrmlField)


#define VRMLNODECHILD2_TEMPLATE_DECL(VrmlType) \
extern template void VRMLEXPORT VrmlNode::registerField<VrmlType>(VrmlNode *node, const std::string& name, VrmlType *field, const FieldUpdateCallback<VrmlType> &updateCb);
FOR_ALL_VRML_TYPES(VRMLNODECHILD2_TEMPLATE_DECL)

#define TO_VRML_FIELD_TYPES_DECL(VrmlType) \
extern template VrmlField::VrmlFieldType VRMLEXPORT toEnumType(const VrmlType *t);
FOR_ALL_VRML_TYPES(TO_VRML_FIELD_TYPES_DECL)

#define INIT_FIELDS_HELPER_DECL(VrmlType) \
extern template void VRMLEXPORT VrmlNode::initFieldsHelperImpl(VrmlNode *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::Private> &field); 
FOR_ALL_VRML_TYPES(INIT_FIELDS_HELPER_DECL)

#define INIT_EXPOSED_FIELDS_HELPER_DECL(VrmlType) \
extern template void VRMLEXPORT VrmlNode::initFieldsHelperImpl(VrmlNode *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::Exposed> &field); 
FOR_ALL_VRML_TYPES(INIT_EXPOSED_FIELDS_HELPER_DECL)

#define INIT_EVENT_IN_HELPER_DECL(VrmlType) \
extern template void VRMLEXPORT VrmlNode::initFieldsHelperImpl(VrmlNode *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::EventIn> &field); 
FOR_ALL_VRML_TYPES(INIT_EVENT_IN_HELPER_DECL)

#define INIT_EVENT_OUT_HELPER_DECL(VrmlType) \
extern template void VRMLEXPORT VrmlNode::initFieldsHelperImpl(VrmlNode *node, VrmlNodeType *t, const NameValueStruct<VrmlType, FieldAccessibility::EventOut> &field); 
FOR_ALL_VRML_TYPES(INIT_EVENT_OUT_HELPER_DECL)

} // vrml


#endif // COVER_VRMLFIELDTEMPLATE_H