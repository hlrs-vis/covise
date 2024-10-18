/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  The VrmlNode class is the base node class.
#ifndef VRML_NODE_H
#define VRML_NODE_H

#include "config.h"
#include "VrmlNode.h"
#include "VrmlNamespace.h"
#include "VrmlNodeType.h"
#include "VrmlNodeScript.h"
#include "VrmlScene.h"
#include "MathUtils.h"
#include "coEventQueue.h"
#include <stdio.h>

using std::cerr;
using std::endl;
using namespace vrml;

VrmlNodeType *VrmlNode::defineType(VrmlNodeType *t)
{
    if (t)
        t->addExposedField("metadata", VrmlField::SFNODE);
    return t;
}

VrmlNodeType *VrmlNode::nodeType() const { return 0; }

VrmlNode::VrmlNode(VrmlScene *scene)
    : d_scene(scene)
    , d_modified(false)
    , d_routes(0)
    , d_incomingRoutes(0)
    , d_myNamespace(0)
    , d_traverseAtFrame(0)
    , d_refCount(0)
    , d_metadata(0)
{
    d_isDeletedInline = false;
}

VrmlNode::VrmlNode(const VrmlNode &)
    : d_scene(0)
    , d_modified(true)
    , d_routes(0)
    , d_incomingRoutes(0)
    , d_myNamespace(0)
    , d_traverseAtFrame(0)
    , d_refCount(0)
    , d_metadata(0)
{
    d_isDeletedInline = false;
}

// Free name (if any) and route info.

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
VrmlNode *VrmlNode::reference()
{
    ++d_refCount;
    return this;
}

// Remove a reference to a node

void VrmlNode::dereference()
{
    if (--d_refCount == 0)
        delete this;
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

// Node type tests

VrmlNodeAnchor *VrmlNode::toAnchor() const { return 0; }
VrmlNodeAppearance *VrmlNode::toAppearance() const { return 0; }
VrmlNodeWave *VrmlNode::toWave() const { return 0; }
VrmlNodeBumpMapping *VrmlNode::toBumpMapping() const { return 0; }
VrmlNodeAudioClip *VrmlNode::toAudioClip() const { return 0; }
VrmlNodeBackground *VrmlNode::toBackground() const { return 0; }
VrmlNodeBooleanSequencer *VrmlNode::toBooleanSequencer() const { return 0; }
VrmlNodeBox *VrmlNode::toBox() const //LarryD Mar 08/99
{
    return 0;
}

VrmlNodeChild *VrmlNode::toChild() const { return 0; }
VrmlNodeColor *VrmlNode::toColor() const { return 0; }
VrmlNodeColorRGBA *VrmlNode::toColorRGBA() const { return 0; }
VrmlNodeCone *VrmlNode::toCone() const //LarryD Mar 08/99
{
    return 0;
}

VrmlNodeCoordinate *VrmlNode::toCoordinate() const { return 0; }
VrmlNodeCylinder *VrmlNode::toCylinder() const //LarryD Mar 08/99
{
    return 0;
}

VrmlNodeDirLight *VrmlNode::toDirLight() const //LarryD Mar 04/99
{
    return 0;
}

//LarryD Mar 09/99
VrmlNodeElevationGrid *VrmlNode::toElevationGrid() const
{
    return 0;
}

//LarryD Mar 09/99
VrmlNodeExtrusion *VrmlNode::toExtrusion() const
{
    return 0;
}

VrmlNodeFog *VrmlNode::toFog() const { return 0; }
VrmlNodeFontStyle *VrmlNode::toFontStyle() const { return 0; }
VrmlNodeGeometry *VrmlNode::toGeometry() const { return 0; }
VrmlNodeGroup *VrmlNode::toGroup() const { return 0; }
VrmlNodeIFaceSet *VrmlNode::toIFaceSet() const { return 0; }
VrmlNodeIQuadSet *VrmlNode::toIQuadSet() const { return 0; }
VrmlNodeITriangleFanSet *VrmlNode::toITriangleFanSet() const { return 0; }
VrmlNodeITriangleSet *VrmlNode::toITriangleSet() const { return 0; }
VrmlNodeITriangleStripSet *VrmlNode::toITriangleStripSet() const { return 0; }
VrmlNodeInline *VrmlNode::toInline() const { return 0; }
VrmlNodeLight *VrmlNode::toLight() const { return 0; }
VrmlNodeMaterial *VrmlNode::toMaterial() const { return 0; }
VrmlNodeMetadataBoolean *VrmlNode::toMetadataBoolean() const { return 0; }
VrmlNodeMetadataDouble *VrmlNode::toMetadataDouble() const { return 0; }
VrmlNodeMetadataFloat *VrmlNode::toMetadataFloat() const { return 0; }
VrmlNodeMetadataInteger *VrmlNode::toMetadataInteger() const { return 0; }
VrmlNodeMetadataSet *VrmlNode::toMetadataSet() const { return 0; }
VrmlNodeMetadataString *VrmlNode::toMetadataString() const { return 0; }
VrmlNodeMovieTexture *VrmlNode::toMovieTexture() const { return 0; }
VrmlNodeMultiTexture *VrmlNode::toMultiTexture() const { return 0; }
VrmlNodeMultiTextureCoordinate *VrmlNode::toMultiTextureCoordinate() const { return 0; }
VrmlNodeMultiTextureTransform *VrmlNode::toMultiTextureTransform() const { return 0; }
VrmlNodeNavigationInfo *VrmlNode::toNavigationInfo() const { return 0; }
VrmlNodeCOVER *VrmlNode::toCOVER() const { return 0; }
VrmlNodeNormal *VrmlNode::toNormal() const { return 0; }
VrmlNodePlaneSensor *VrmlNode::toPlaneSensor() const { return 0; }
VrmlNodeSpaceSensor *VrmlNode::toSpaceSensor() const { return 0; }
VrmlNodeARSensor *VrmlNode::toARSensor() const { return 0; }
VrmlNodePointLight *VrmlNode::toPointLight() const { return 0; }
VrmlNodeProximitySensor *VrmlNode::toProximitySensor() const { return 0; }
VrmlNodeQuadSet *VrmlNode::toQuadSet() const { return 0; }
VrmlNodeScript *VrmlNode::toScript() const { return 0; }
VrmlNodeShape *VrmlNode::toShape() const { return 0; }
VrmlNodeSound *VrmlNode::toSound() const { return 0; }
VrmlNodeSphere *VrmlNode::toSphere() const //LarryD Mar 08/99
{
    return 0;
}

VrmlNodeSpotLight *VrmlNode::toSpotLight() const { return 0; }
VrmlNodeSwitch *VrmlNode::toSwitch() const //LarryD Mar 08/99
{
    return 0;
}

VrmlNodeTexture *VrmlNode::toTexture() const { return 0; }
VrmlNodeTextureCoordinate *VrmlNode::toTextureCoordinate() const { return 0; }
VrmlNodeTextureCoordinateGenerator *VrmlNode::toTextureCoordinateGenerator() const { return 0; }
VrmlNodeTextureTransform *VrmlNode::toTextureTransform() const { return 0; }
VrmlNodeTimeSensor *VrmlNode::toTimeSensor() const { return 0; }
VrmlNodeTouchSensor *VrmlNode::toTouchSensor() const { return 0; }
VrmlNodeTriangleFanSet *VrmlNode::toTriangleFanSet() const { return 0; }
VrmlNodeTriangleSet *VrmlNode::toTriangleSet() const { return 0; }
VrmlNodeTriangleStripSet *VrmlNode::toTriangleStripSet() const { return 0; }
VrmlNodeCylinderSensor *VrmlNode::toCylinderSensor() const { return 0; }
VrmlNodeSphereSensor *VrmlNode::toSphereSensor() const { return 0; }
VrmlNodeTransform *VrmlNode::toTransform() const //LarryD Feb 24/99
{
    return 0;
}

VrmlNodeViewpoint *VrmlNode::toViewpoint() const { return 0; }

VrmlNodeImageTexture *VrmlNode::toImageTexture() const { return 0; }
VrmlNodeCubeTexture *VrmlNode::toCubeTexture() const { return 0; }
VrmlNodePixelTexture *VrmlNode::toPixelTexture() const { return 0; }

VrmlNodeLOD *VrmlNode::toLOD() const { return 0; }
VrmlNodeScalarInt *VrmlNode::toScalarInt() const { return 0; }
VrmlNodeOrientationInt *VrmlNode::toOrientationInt() const { return 0; }
VrmlNodePositionInt *VrmlNode::toPositionInt() const { return 0; }

VrmlNodeProto *VrmlNode::toProto() const { return 0; }

// Routes

Route::Route(const char *fromEventOut, VrmlNode *toNode, const char *toEventIn, VrmlNode *fromNode)
    : d_prev(0)
    , d_next(0)
    , d_prevI(0)
    , d_nextI(0)
{
    d_fromEventOut = new char[strlen(fromEventOut) + 1];
    strcpy(d_fromEventOut, fromEventOut);
    //if((strlen(d_fromEventOut) > 8)&&(strcmp(d_fromEventOut+strlen(d_fromEventOut)-8,"_changed")==0))
    //    d_fromEventOut[strlen(d_fromEventOut)-8]='\0';
    d_toNode = toNode;
    toNode->addRouteI(this);
    d_fromNode = fromNode;
    d_toEventIn = new char[strlen(toEventIn) + 1];
    //if(strncmp(toEventIn,"set_",4) == 0)
    //    strcpy(d_toEventIn, toEventIn+4);
    //else
    strcpy(d_toEventIn, toEventIn);

    d_fromImportName = new char[1];
    d_fromImportName[0] = '\0';

    d_toImportName = new char[1];
    d_toImportName[0] = '\0';
}

Route::Route(const Route &r)
{
    d_fromEventOut = new char[strlen(r.d_fromEventOut) + 1];
    strcpy(d_fromEventOut, r.d_fromEventOut);
    d_toNode = r.d_toNode;
    d_fromNode = r.d_fromNode;
    d_toEventIn = new char[strlen(r.d_toEventIn) + 1];
    strcpy(d_toEventIn, r.d_toEventIn);

    d_fromImportName = new char[strlen(r.d_fromImportName) + 1];
    if (strlen(r.d_fromImportName) > 0)
        strcpy(d_fromImportName, r.d_fromImportName);
    else
        d_fromImportName[0] = '\0';

    d_toImportName = new char[strlen(r.d_toImportName) + 1];
    if (strlen(r.d_toImportName) > 0)
        strcpy(d_toImportName, r.d_toImportName);
    else
        d_toImportName[0] = '\0';
}

Route::~Route()
{
    if (d_toNode)
    {
        d_toNode->removeRoute(this);
    }
    if (d_fromNode)
    {
        d_fromNode->removeRoute(this);
    }
    delete[] d_fromEventOut;
    delete[] d_toEventIn;

    delete[] d_fromImportName;
    d_fromImportName = NULL;
    delete[] d_toImportName;
    d_toImportName = NULL;
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

void Route::addFromImportName(const char *name)
{
    delete[] d_fromImportName;
    d_fromImportName = new char[strlen(name) + 1];
    strcpy(d_fromImportName, name);
}

void Route::addToImportName(const char *name)
{
    delete[] d_toImportName;
    d_toImportName = new char[strlen(name) + 1];
    strcpy(d_toImportName, name);
}

Route *Route::newFromRoute(VrmlNode *newFromNode)
{
    return newFromNode->addRoute(d_fromEventOut, d_toNode, d_toEventIn);
}

Route *Route::newToRoute(VrmlNode *newToNode)
{
    return d_fromNode->addRoute(d_fromEventOut, newToNode, d_toEventIn);
}

VrmlNode *Route::newFromNode(void)
{
    if (strlen(d_fromImportName) != 0)
        return d_fromNode->findInside(d_fromImportName);
    return NULL;
}

VrmlNode *Route::newToNode(void)
{
    if (strlen(d_toImportName) != 0)
        return d_toNode->findInside(d_toImportName);
    return NULL;
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

bool VrmlNode::haveToRender()
{
    return true;
    return (d_traverseAtFrame < 0 || d_traverseAtFrame == System::the->frame());
}

int VrmlNode::getTraversalForce()
{
    if (d_traverseAtFrame < 0)
        return -d_traverseAtFrame;
    else
        return 0;
}

void VrmlNode::forceTraversal(bool once, int increment)
{
    //fprintf(stderr, "name=%s, once=%d, inc=%d\n", name(), int(once), increment);
    if (once)
    {
        if (d_traverseAtFrame < 0)
        {
            return;
        }

        if (d_traverseAtFrame == System::the->frame())
        {
            return;
        }

        if (d_traverseAtFrame >= 0)
        {
            d_traverseAtFrame = System::the->frame();
        }
    }
    else
    {
        if (d_traverseAtFrame >= 0)
        {
            d_traverseAtFrame = -increment;
        }
        else
        {
            d_traverseAtFrame -= increment;
        }
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
    {
        return;
    }

    if (num == -1)
    {
        num = -d_traverseAtFrame;
    }

    d_traverseAtFrame += num;

    for (ParentList::iterator it = parentList.begin();
         it != parentList.end();
         it++)
    {
        (*it)->decreaseTraversalForce(num);
    }
}

bool VrmlNode::isModified() const
{
    return d_modified;
}

void VrmlNode::clearFlags()
{
    d_flag = false;
}

// Render

void VrmlNode::render(Viewer *)
{
    clearModified();
}

// Accumulate transformations for proper rendering of bindable nodes.

void VrmlNode::accumulateTransform(VrmlNode *)
{
    ;
}

VrmlNode *VrmlNode::getParentTransform() { return 0; }

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
    if (d_routes)
    {
        //std::cerr << "Nr" << std::flush;
        return false;
    }
    if (d_incomingRoutes)
    {
        //std::cerr << "Ni" << std::flush;
        return false;
    }

    if (strstr(name(), "NotCached") != NULL || strstr(name(), "NoCache") != NULL)
    {
        //std::cerr << "Nn" << std::flush;
        return false;
    }

	if (strstr(name(), "coMirror") != NULL)
	{
		//std::cerr << "Nn" << std::flush;
		return false;
	}
    //std::cerr << "N(" << nodeType()->getName() << ")" << std::flush;
    return true;
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
    else if (auto scriptNode = toScript())
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
            cerr << "Error: unhandled eventIn " << nodeType()->getName()
                 << "::" << name() << "." << origEventName << endl;
    }

    else
        cerr << "Error: unhandled eventIn " << nodeType()->getName()
             << "::" << name() << "." << origEventName << endl;
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

namespace vrml
{
std::ostream &operator<<(std::ostream &os, const VrmlNode &f)
{
    return f.print(os, 0);
}
}

std::ostream &VrmlNode::print(std::ostream &os, int indent) const
{
    const char *nm = name();
    for (int i = 0; i < indent; ++i)
        os << ' ';

    if (nm && *nm)
        os << "DEF " << nm << " ";

    os << nodeType()->getName() << " { ";

    // cast away const-ness for now...
    if (d_metadata.get())
        PRINT_FIELD(metadata);
    VrmlNode *n = (VrmlNode *)this;
    n->printFields(os, indent + INDENT_INCREMENT);

    os << " }";

    return os;
}

// This should probably generate an error...
// Might be nice to make this non-virtual (each node would have
// to provide a getField(const char* name) method and specify
// default values in the addField(). The VrmlNodeType class would
// have to make the fields list public.

std::ostream &VrmlNode::printFields(std::ostream &os, int /*indent*/)
{
    os << "# Error: " << nodeType()->getName()
       << "::printFields unimplemented.\n";
    return os;
}

std::ostream &VrmlNode::printField(std::ostream &os,
                                   int indent,
                                   const char *name,
                                   const VrmlField &f)
{
    os << endl;
    for (int i = 0; i < indent; ++i)
        os << ' ';
    os << name << ' ' << f;
    return os;
}

// Set the value of one of the node fields. No fields exist at the
// top level, so reaching this indicates an error.

void VrmlNode::setField(const char *fieldName, const VrmlField &fieldValue)
{
    if
        TRY_SFNODE_FIELD6(metadata, MetadataBoolean,
                          MetadataDouble,
                          MetadataFloat,
                          MetadataInteger,
                          MetadataSet,
                          MetadataString)
    else
        System::the->error("%s::setField: no such field (%s)\n",
                           nodeType()->getName(), fieldName);
}

// Get the value of a field or eventOut.

const VrmlField *VrmlNode::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "metadata") == 0)
        return &d_metadata;

    System::the->error("%s(%s)::getField: no such field or eventOut (%s)\n",
                       name(), nodeType()->getName(), fieldName);

    return 0;
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
    VrmlNodeScript *scriptNode;
    if ((scriptNode = toScript()))
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

//
//  VrmlNodeChild- should move to its own file
//
#include "VrmlNodeChild.h"

void VrmlNodeChild::initFields(VrmlNodeChild *node, VrmlNodeType *t)
{
    //space for future implementations
}

VrmlNodeChild::VrmlNodeChild(VrmlScene *scene, const std::string& name)
    : VrmlNodeTemplate(scene, name)
{
}

VrmlNodeChild *VrmlNodeChild::toChild() const
{
    return (VrmlNodeChild *)this; // discards const...
}

#endif
