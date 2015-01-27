/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//  %W% %G%
#ifndef _VRMLNODEPROTO_
#define _VRMLNODEPROTO_

//
// A VrmlNodeProto object represents an instance of a PROTOd node.
// The definition of the PROTO is stored in a VrmlNodeType object;
// the VrmlNodeProto object stores a local copy of the implementation
// nodes.
//

#include "config.h"
#include "VrmlNode.h"
#include "VrmlNodeType.h"
#include "Viewer.h"

namespace vrml
{

class VrmlMFNode;

class VRMLEXPORT VrmlNodeProto : public VrmlNode
{

public:
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeProto(VrmlNodeType *nodeDef, VrmlScene *scene);
    VrmlNodeProto(const VrmlNodeProto &);
    virtual ~VrmlNodeProto();

    virtual VrmlNode *cloneMe() const;

    virtual void addToScene(VrmlScene *, const char *relUrl);
    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual VrmlNodeProto *toProto() const;

    // These are passed along to the first implementation node of the proto.
    virtual VrmlNodeAnchor *toAnchor() const;
    virtual VrmlNodeAppearance *toAppearance() const;
    virtual VrmlNodeWave *toWave() const;
    virtual VrmlNodeBumpMapping *toBumpMapping() const;
    virtual VrmlNodeAudioClip *toAudioClip() const;
    virtual VrmlNodeBackground *toBackground() const;
    virtual VrmlNodeChild *toChild() const;
    virtual VrmlNodeColor *toColor() const;
    virtual VrmlNodeCoordinate *toCoordinate() const;
    virtual VrmlNodeFog *toFog() const;
    virtual VrmlNodeFontStyle *toFontStyle() const;
    virtual VrmlNodeGeometry *toGeometry() const;
    virtual VrmlNodeGroup *toGroup() const;
    virtual VrmlNodeInline *toInline() const;
    virtual VrmlNodeLight *toLight() const;
    virtual VrmlNodeMaterial *toMaterial() const;
    virtual VrmlNodeMovieTexture *toMovieTexture() const;
    virtual VrmlNodeNavigationInfo *toNavigationInfo() const;
    virtual VrmlNodeCOVER *toCOVER() const;
    virtual VrmlNodeNormal *toNormal() const;
    virtual VrmlNodePlaneSensor *toPlaneSensor() const;
    virtual VrmlNodeSpaceSensor *toSpaceSensor() const;
    virtual VrmlNodeARSensor *toARSensor() const;
    virtual VrmlNodePointLight *toPointLight() const;
    virtual VrmlNodeScript *toScript() const;
    virtual VrmlNodeSound *toSound() const;
    virtual VrmlNodeSpotLight *toSpotLight() const;
    virtual VrmlNodeTexture *toTexture() const;
    virtual VrmlNodeTextureCoordinate *toTextureCoordinate() const;
    virtual VrmlNodeTextureTransform *toTextureTransform() const;
    virtual VrmlNodeTimeSensor *toTimeSensor() const;
    virtual VrmlNodeTouchSensor *toTouchSensor() const;
    virtual VrmlNodeViewpoint *toViewpoint() const;

    // Larry
    virtual VrmlNodeBox *toBox() const;
    virtual VrmlNodeCone *toCone() const;
    virtual VrmlNodeCylinder *toCylinder() const;
    virtual VrmlNodeDirLight *toDirLight() const;
    virtual VrmlNodeElevationGrid *toElevationGrid() const;
    virtual VrmlNodeExtrusion *toExtrusion() const;
    virtual VrmlNodeIFaceSet *toIFaceSet() const;
    virtual VrmlNodeShape *toShape() const;
    virtual VrmlNodeSphere *toSphere() const;
    virtual VrmlNodeSwitch *toSwitch() const;
    virtual VrmlNodeTransform *toTransform() const;
    virtual VrmlNodeImageTexture *toImageTexture() const;
    virtual VrmlNodeCubeTexture *toCubeTexture() const;
    virtual VrmlNodePixelTexture *toPixelTexture() const;
    virtual VrmlNodeLOD *toLOD() const;
    virtual VrmlNodeScalarInt *toScalarInt() const;
    virtual VrmlNodeOrientationInt *toOrientationInt() const;
    virtual VrmlNodePositionInt *toPositionInt() const;

    virtual void render(Viewer *);

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

    virtual bool isModified() const;

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);

    virtual const VrmlField *getField(const char *fieldName) const;

    virtual void accumulateTransform(VrmlNode *);

    // LarryD  Feb 11/99
    int size();
    // LarryD  Feb 11/99
    VrmlNode *child(int index);

    // LarryD Feb 11/99
    VrmlMFNode *getNodes()
    {
        return d_nodes;
    }

    // Field name/value pairs specified in PROTO instantiation
    typedef struct
    {
        char *name;
        VrmlField *value;
    } NameValueRec;

    virtual Viewer::Object getViewerObject()
    {
        return d_viewerObject;
    };

private:
    VrmlNode *firstNode() const;

    // Instantiate the proto by cloning the node type implementation nodes.
    void instantiate();

    // Find a field by name
    NameValueRec *findField(const char *fieldName) const;

    VrmlNodeType *d_nodeType; // Definition

    bool d_instantiated;
    VrmlNamespace *d_scope; // Node type and name bindings

    VrmlMFNode *d_nodes; // Local copy of implementation nodes.

    std::list<NameValueRec *> d_fields; // Field values

    // Dispatch eventIns from outside the PROTO to internal eventIns
    typedef struct
    {
        char *name;
        VrmlNodeType::ISMap ismap;
    } EventDispatch;

    typedef std::list<EventDispatch *> EventDispatchList;

    EventDispatchList d_eventDispatch;

    Viewer::Object d_viewerObject; // move to VrmlNode.h ? ...
};
}
#endif //_VRMLNODEPROTO_
