/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNode.h

#ifndef _VRMLNODE_
#define _VRMLNODE_

#include "config.h"
#include "System.h"

#include <iostream>
#include <list>

#include "VrmlSFNode.h"

namespace vrml
{

class VrmlNode;
typedef std::list<VrmlNode *> VrmlNodeList;

class Route;
class Viewer;

class VrmlNamespace;
class VrmlNodeType;
class VrmlField;
class VrmlScene;

// For the safe downcasts
class VrmlNodeAnchor;
class VrmlNodeAppearance;
class VrmlNodeWave;
class VrmlNodeBumpMapping;
class VrmlNodeAudioClip;
class VrmlNodeBackground;
class VrmlNodeBox; //LarryD Mar 08/99
class VrmlNodeBooleanSequencer;
class VrmlNodeChild;
class VrmlNodeColor;
class VrmlNodeColorRGBA;
class VrmlNodeCone; //LarryD Mar 08/99
class VrmlNodeCoordinate;
class VrmlNodeCylinder; //LarryD Mar 08/99
class VrmlNodeDirLight; //LarryD Mar 08/99
class VrmlNodeElevationGrid; //LarryD Mar 09/99
class VrmlNodeExtrusion; //LarryD Mar 09/99
class VrmlNodeFog;
class VrmlNodeFontStyle;
class VrmlNodeGeometry;
class VrmlNodeGroup;
class VrmlNodeIFaceSet;
class VrmlNodeIQuadSet;
class VrmlNodeITriangleFanSet;
class VrmlNodeITriangleSet;
class VrmlNodeITriangleStripSet;
class VrmlNodeInline;
class VrmlNodeLight;
class VrmlNodeMaterial;
class VrmlNodeMetadataBoolean;
class VrmlNodeMetadataDouble;
class VrmlNodeMetadataFloat;
class VrmlNodeMetadataInteger;
class VrmlNodeMetadataSet;
class VrmlNodeMetadataString;
class VrmlNodeMovieTexture;
class VrmlNodeMultiTexture;
class VrmlNodeMultiTextureCoordinate;
class VrmlNodeMultiTextureTransform;
class VrmlNodeNavigationInfo;
class VrmlNodeCOVER;
class VrmlNodeNormal;
class VrmlNodePlaneSensor;
class VrmlNodeSpaceSensor;
class VrmlNodeARSensor;
class VrmlNodePointLight;
class VrmlNodeProximitySensor;
class VrmlNodeQuadSet;
class VrmlNodeScript;
class VrmlNodeShape;
class VrmlNodeSphere; //LarryD Mar 08/99
class VrmlNodeSound;
class VrmlNodeSpotLight;
class VrmlNodeSwitch;
class VrmlNodeTexture;
class VrmlNodeTextureCoordinate;
class VrmlNodeTextureCoordinateGenerator;
class VrmlNodeTextureTransform;
class VrmlNodeTimeSensor;
class VrmlNodeTouchSensor;
class VrmlNodeTriangleFanSet;
class VrmlNodeTriangleSet;
class VrmlNodeTriangleStripSet;
class VrmlNodeSphereSensor;
class VrmlNodeCylinderSensor;
class VrmlNodeTransform;
class VrmlNodeViewpoint;
class VrmlNodeImageTexture;
class VrmlNodeCubeTexture;
class VrmlNodePixelTexture;
class VrmlNodeLOD;
class VrmlNodeScalarInt;
class VrmlNodeOrientationInt;
class VrmlNodePositionInt;

class VrmlNodeProto;

class VRMLEXPORT VrmlNode
{
    friend std::ostream &operator<<(std::ostream &os, const VrmlNode &f);

public:
    typedef std::list<VrmlNode *> ParentList;
    ParentList parentList;

    // Define the fields of all built in VrmlNodeTypes
    static VrmlNodeType *defineType(VrmlNodeType *t);
    virtual VrmlNodeType *nodeType() const;

    // VrmlNodes are reference counted, optionally named objects
    // The reference counting is manual (that is, each user of a
    // VrmlNode, such as the VrmlMFNode class, calls reference()
    // and dereference() explicitly). Should make it internal...

    VrmlNode(VrmlScene *s);
    VrmlNode(const VrmlNode &);
    virtual ~VrmlNode() = 0;

    // Copy the node, defining its name in the specified scope.
    // Uses the flag to determine whether the node is a USEd node.
    VrmlNode *clone(VrmlNamespace *);
    virtual VrmlNode *cloneMe() const = 0;
    virtual void cloneChildren(VrmlNamespace *);

    // Copy the ROUTEs
    virtual void copyRoutes(VrmlNamespace *ns);

    // Add/remove references to a VrmlNode. This is silly, as it
    // requires the users of VrmlNode to do the reference/derefs...
    VrmlNode *reference();
    void dereference();

    // Safe node downcasts. These avoid the dangerous casts of VrmlNode* (esp in
    // presence of protos), but are ugly in that this class must know about all
    // the subclasses. These return 0 if the typecast is invalid.
    // Remember to also add new ones to VrmlNodeProto. Protos should
    // return their first implementation node (except toProto()).
    virtual VrmlNodeAnchor *toAnchor() const;
    virtual VrmlNodeAppearance *toAppearance() const;
    virtual VrmlNodeWave *toWave() const;
    virtual VrmlNodeBumpMapping *toBumpMapping() const;
    virtual VrmlNodeAudioClip *toAudioClip() const;
    virtual VrmlNodeBackground *toBackground() const;
    virtual VrmlNodeBox *toBox() const; //LarryD Mar 08/99
    virtual VrmlNodeBooleanSequencer *toBooleanSequencer() const;
    virtual VrmlNodeChild *toChild() const;
    virtual VrmlNodeColor *toColor() const;
    virtual VrmlNodeColorRGBA *toColorRGBA() const;
    virtual VrmlNodeCone *toCone() const; //LarryD Mar 08/99
    virtual VrmlNodeCoordinate *toCoordinate() const;
    //LarryD Mar 08/99
    virtual VrmlNodeCylinder *toCylinder() const;
    //LarryD Mar 08/99
    virtual VrmlNodeDirLight *toDirLight() const;
    //LarryD Mar 09/99
    virtual VrmlNodeElevationGrid *toElevationGrid() const;
    //LarryD Mar 09/99
    virtual VrmlNodeExtrusion *toExtrusion() const;
    virtual VrmlNodeFog *toFog() const;
    virtual VrmlNodeFontStyle *toFontStyle() const;
    virtual VrmlNodeGeometry *toGeometry() const;
    virtual VrmlNodeGroup *toGroup() const;
    virtual VrmlNodeIFaceSet *toIFaceSet() const;
    virtual VrmlNodeIQuadSet *toIQuadSet() const;
    virtual VrmlNodeITriangleFanSet *toITriangleFanSet() const;
    virtual VrmlNodeITriangleSet *toITriangleSet() const;
    virtual VrmlNodeITriangleStripSet *toITriangleStripSet() const;
    virtual VrmlNodeImageTexture *toImageTexture() const;
    virtual VrmlNodeCubeTexture *toCubeTexture() const;
    virtual VrmlNodePixelTexture *toPixelTexture() const;
    virtual VrmlNodeInline *toInline() const;
    virtual VrmlNodeLight *toLight() const;
    virtual VrmlNodeMaterial *toMaterial() const;
    virtual VrmlNodeMetadataBoolean *toMetadataBoolean() const;
    virtual VrmlNodeMetadataDouble *toMetadataDouble() const;
    virtual VrmlNodeMetadataFloat *toMetadataFloat() const;
    virtual VrmlNodeMetadataInteger *toMetadataInteger() const;
    virtual VrmlNodeMetadataSet *toMetadataSet() const;
    virtual VrmlNodeMetadataString *toMetadataString() const;
    virtual VrmlNodeMovieTexture *toMovieTexture() const;
    virtual VrmlNodeMultiTexture *toMultiTexture() const;
    virtual VrmlNodeMultiTextureCoordinate *toMultiTextureCoordinate() const;
    virtual VrmlNodeMultiTextureTransform *toMultiTextureTransform() const;
    virtual VrmlNodeNavigationInfo *toNavigationInfo() const;
    virtual VrmlNodeCOVER *toCOVER() const;
    virtual VrmlNodeNormal *toNormal() const;
    virtual VrmlNodePlaneSensor *toPlaneSensor() const;
    virtual VrmlNodeSpaceSensor *toSpaceSensor() const;
    virtual VrmlNodeARSensor *toARSensor() const;
    virtual VrmlNodePointLight *toPointLight() const;
    virtual VrmlNodeProximitySensor *toProximitySensor() const;
    virtual VrmlNodeQuadSet *toQuadSet() const;
    virtual VrmlNodeScript *toScript() const;
    virtual VrmlNodeShape *toShape() const;
    virtual VrmlNodeSphere *toSphere() const; //LarryD Mar 08/99
    virtual VrmlNodeSound *toSound() const;
    virtual VrmlNodeSpotLight *toSpotLight() const;
    virtual VrmlNodeSwitch *toSwitch() const; //LarryD Mar 08/99
    virtual VrmlNodeTexture *toTexture() const;
    virtual VrmlNodeTextureCoordinate *toTextureCoordinate() const;
    virtual VrmlNodeTextureCoordinateGenerator *toTextureCoordinateGenerator() const;
    virtual VrmlNodeTextureTransform *toTextureTransform() const;
    virtual VrmlNodeTimeSensor *toTimeSensor() const;
    virtual VrmlNodeTouchSensor *toTouchSensor() const;
    virtual VrmlNodeTriangleFanSet *toTriangleFanSet() const;
    virtual VrmlNodeTriangleSet *toTriangleSet() const;
    virtual VrmlNodeTriangleStripSet *toTriangleStripSet() const;
    virtual VrmlNodeSphereSensor *toSphereSensor() const;
    virtual VrmlNodeCylinderSensor *toCylinderSensor() const;
    //LarryD Feb 24/99
    virtual VrmlNodeTransform *toTransform() const;
    virtual VrmlNodeViewpoint *toViewpoint() const;

    virtual VrmlNodeLOD *toLOD() const;
    virtual VrmlNodeScalarInt *toScalarInt() const;
    virtual VrmlNodeOrientationInt *toOrientationInt() const;
    virtual VrmlNodePositionInt *toPositionInt() const;

    virtual VrmlNodeProto *toProto() const;

    // Node DEF/USE/ROUTE name
    void setName(const char *nodeName, VrmlNamespace *ns = 0);
    inline const char *name() const
    {
        return d_name;
    };
    VrmlNamespace *getNamespace() const;

    // Add to a scene. A node can belong to at most one scene for now.
    // If it doesn't belong to a scene, it can't be rendered.
    virtual void addToScene(VrmlScene *, const char *relativeUrl);

    // Write self
    std::ostream &print(std::ostream &os, int indent) const;
    virtual std::ostream &printFields(std::ostream &os, int indent);
    static std::ostream &printField(std::ostream &, int, const char *, const VrmlField &);

    // Indicate that the node state has changed, need to re-render
    void setModified();
    void clearModified()
    {
        d_modified = false;
    }
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

    // Set a field by name (used by the parser, not for external consumption).
    virtual void setField(const char *fieldName,
                          const VrmlField &fieldValue);

    // Get a field or eventOut by name.
    virtual const VrmlField *getField(const char *fieldName) const;

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

protected:
    enum
    {
        INDENT_INCREMENT = 4
    };

    // Send a named event from this node.
    void eventOut(double timeStamp,
                  const char *eventName,
                  const VrmlField &fieldValue);

    // Scene this node belongs to
    VrmlScene *d_scene;

    // True if a field changed since last render
    bool d_modified;
    bool d_flag;

    // Routes from this node (clean this up, add RouteList ...)
    Route *d_routes;
    Route *d_incomingRoutes;

    //
    VrmlNamespace *d_myNamespace;
    // true if this node is on the static node stack
    bool isOnStack(VrmlNode *);
    static VrmlNodeList nodeStack;

    // <0:  render every frame
    // >=0: render at this very frame
    int d_traverseAtFrame;

    bool d_isDeletedInline;

private:
    int d_refCount; // Number of active references
    char *d_name;

    VrmlSFNode d_metadata;
};

// Routes
class Route
{
public:
    Route(const char *fromEventOut, VrmlNode *toNode, const char *toEventIn, VrmlNode *fromNode);
    Route(const Route &);
    ~Route();

    char *fromEventOut()
    {
        return d_fromEventOut;
    }
    char *toEventIn()
    {
        return d_toEventIn;
    }
    VrmlNode *toNode()
    {
        return d_toNode;
    }
    VrmlNode *fromNode()
    {
        return d_fromNode;
    }

    void addFromImportName(const char *name);
    void addToImportName(const char *name);

    Route *newFromRoute(VrmlNode *newFromNode);
    Route *newToRoute(VrmlNode *newToNode);

    VrmlNode *newFromNode(void);
    VrmlNode *newToNode(void);

    Route *prev()
    {
        return d_prev;
    }
    Route *next()
    {
        return d_next;
    }
    void setPrev(Route *r)
    {
        d_prev = r;
    }
    void setNext(Route *r)
    {
        d_next = r;
    }
    Route *prevI()
    {
        return d_prevI;
    }
    Route *nextI()
    {
        return d_nextI;
    }
    void setPrevI(Route *r)
    {
        d_prevI = r;
    }
    void setNextI(Route *r)
    {
        d_nextI = r;
    }

private:
    char *d_fromEventOut;
    VrmlNode *d_toNode;
    VrmlNode *d_fromNode;
    char *d_toEventIn;

    char *d_fromImportName;
    char *d_toImportName;

    Route *d_prev, *d_next;
    Route *d_prevI, *d_nextI;
};
}
// Ugly macro used in printFields
#define PRINT_FIELD(_f) printField(os, indent + INDENT_INCREMENT, #_f, d_##_f)

// Ugly macros used in setField

#define TRY_FIELD(_f, _t)                                                                                                                                  \
    (strcmp(fieldName, #_f) == 0)                                                                                                                          \
    {                                                                                                                                                      \
        if (fieldValue.to##_t())                                                                                                                           \
            d_##_f = (Vrml##_t &)fieldValue;                                                                                                               \
        else                                                                                                                                               \
            System::the->error("Invalid type (%s) for %s field of %s node (expected %s).\n", fieldValue.fieldTypeName(), #_f, nodeType()->getName(), #_t); \
    }

// For SFNode fields. Allow un-fetched EXTERNPROTOs to succeed...
#define TRY_SFNODE_FIELD(_f, _n)                                                                                                                           \
    (strcmp(fieldName, #_f) == 0)                                                                                                                          \
    {                                                                                                                                                      \
        VrmlSFNode *x = (VrmlSFNode *)&fieldValue;                                                                                                         \
        if (fieldValue.toSFNode() && ((!x->get()) || x->get()->to##_n() || x->get()->toProto()))                                                           \
            d_##_f = (VrmlSFNode &)fieldValue;                                                                                                             \
        else                                                                                                                                               \
            System::the->error("Invalid type (%s) for %s field of %s node (expected %s).\n", fieldValue.fieldTypeName(), #_f, nodeType()->getName(), #_n); \
    }

#define TRY_NAMED_FIELD(_f, _n, _t)                                                                                                                                \
    (strcmp(fieldName, #_n) == 0)                                                                                                                                  \
    {                                                                                                                                                              \
        if (fieldValue.to##_t())                                                                                                                                   \
            d_##_f = (Vrml##_t &)fieldValue;                                                                                                                       \
        else                                                                                                                                                       \
            System::the->error("Invalid type (%s) for %s/%s field of %s node (expected %s).\n", fieldValue.fieldTypeName(), #_f, #_n, nodeType()->getName(), #_t); \
    }

#define TRY_SFNODE_FIELD2(_f, _n1, _n2)                                                                                                                                 \
    (strcmp(fieldName, #_f) == 0)                                                                                                                                       \
    {                                                                                                                                                                   \
        VrmlSFNode *x = (VrmlSFNode *)&fieldValue;                                                                                                                      \
        if (fieldValue.toSFNode() && ((!x->get()) || x->get()->to##_n1() || x->get()->to##_n2() || x->get()->toProto()))                                                \
            d_##_f = (VrmlSFNode &)fieldValue;                                                                                                                          \
        else                                                                                                                                                            \
            System::the->error("Invalid type (%s) for %s field of %s node (expected %s or %s).\n", fieldValue.fieldTypeName(), #_f, nodeType()->getName(), #_n1, #_n2); \
    }

#define TRY_SFNODE_FIELD3(_f, _n1, _n2, _n3)                                                                                                                                        \
    (strcmp(fieldName, #_f) == 0)                                                                                                                                                   \
    {                                                                                                                                                                               \
        VrmlSFNode *x = (VrmlSFNode *)&fieldValue;                                                                                                                                  \
        if (fieldValue.toSFNode() && ((!x->get()) || x->get()->to##_n1() || x->get()->to##_n2() || x->get()->to##_n3() || x->get()->toProto()))                                     \
            d_##_f = (VrmlSFNode &)fieldValue;                                                                                                                                      \
        else                                                                                                                                                                        \
            System::the->error("Invalid type (%s) for %s field of %s node (expected %s or %s or %s).\n", fieldValue.fieldTypeName(), #_f, nodeType()->getName(), #_n1, #_n2, #_n3); \
    }

#define TRY_SFNODE_FIELD4(_f, _n1, _n2, _n3, _n4)                                                                                                                                               \
    (strcmp(fieldName, #_f) == 0)                                                                                                                                                               \
    {                                                                                                                                                                                           \
        VrmlSFNode *x = (VrmlSFNode *)&fieldValue;                                                                                                                                              \
        if (fieldValue.toSFNode() && ((!x->get()) || x->get()->to##_n1() || x->get()->to##_n2() || x->get()->to##_n3() || x->get()->to##_n4() || x->get()->toProto()))                          \
            d_##_f = (VrmlSFNode &)fieldValue;                                                                                                                                                  \
        else                                                                                                                                                                                    \
            System::the->error("Invalid type (%s) for %s field of %s node (expected %s or %s or %s or $s).\n", fieldValue.fieldTypeName(), #_f, nodeType()->getName(), #_n1, #_n2, #_n3, #_n4); \
    }

#define TRY_SFNODE_FIELD5(_f, _n1, _n2, _n3, _n4, _n5)                                                                                                                                        \
    (strcmp(fieldName, #_f) == 0)                                                                                                                                                             \
    {                                                                                                                                                                                         \
        VrmlSFNode *x = (VrmlSFNode *)&fieldValue;                                                                                                                                            \
        if (fieldValue.toSFNode() && ((!x->get()) || x->get()->to##_n1() || x->get()->to##_n2() || x->get()->to##_n3() || x->get()->to##_n4() || x->get()->to##_n5() || x->get()->toProto())) \
            d_##_f = (VrmlSFNode &)fieldValue;                                                                                                                                                \
        else                                                                                                                                                                                  \
            System::the->error("Invalid type (%s) for %s field of %s node (expected %s or %s or %s or %s or %s).\n", fieldValue.fieldTypeName(), #_f, nodeType()->getName(), #_n1, #_n2, #_n3, #_n4, #_n5);

#define TRY_SFNODE_FIELD6(_f, _n1, _n2, _n3, _n4, _n5, _n6)                                                                                                                                                             \
    (strcmp(fieldName, #_f) == 0)                                                                                                                                                                                       \
    {                                                                                                                                                                                                                   \
        VrmlSFNode *x = (VrmlSFNode *)&fieldValue;                                                                                                                                                                      \
        if (fieldValue.toSFNode() && ((!x->get()) || x->get()->to##_n1() || x->get()->to##_n2() || x->get()->to##_n3() || x->get()->to##_n4() || x->get()->to##_n5() || x->get()->to##_n6() || x->get()->toProto()))    \
            d_##_f = (VrmlSFNode &)fieldValue;                                                                                                                                                                          \
        else                                                                                                                                                                                                            \
            System::the->error("Invalid type (%s) for %s field of %s node (expected %s or %s or %s or %s or %s or %s).\n", fieldValue.fieldTypeName(), #_f, nodeType()->getName(), #_n1, #_n2, #_n3, #_n4, #_n5, #_n6); \
    }
#endif //_VRMLNODE_
