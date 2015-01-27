/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//  Copyrights on some portions of the code are held by others as documented
//  in the code. Permission to use this code for any purpose is granted as
//  long as all other copyrights in the code are respected and this copyright
//  statement is retained in the code and accompanying documentation both
//  online and printed.
//
/**************************************************
 * VRML 2.0 Parser
 * Copyright (C) 1996 Silicon Graphics, Inc.
 *
 * Author(s)    : Gavin Bell
 *                Daniel Woods (first port)
 **************************************************
 */

#include "config.h"
#include "VrmlNamespace.h"
#include "VrmlNodeType.h"
#include "VrmlNode.h"
#include "System.h"
#include <stdio.h>
#include <list>
using std::list;

using namespace vrml;

list<VrmlNodeType *> VrmlNamespace::builtInList;
// This is at least be a sorted vector...
NamespaceList VrmlNamespace::allNamespaces;
bool VrmlNamespace::definedBuiltins;

static int numNamespaces = 0;

VrmlNamespace::VrmlNamespace(VrmlNamespace *parent)
    : d_parent(parent)
{
    // Initialize typeList with built in nodes
    if (!definedBuiltins)
    {
        definedBuiltins = true;
        defineBuiltIns();
    }

    namespaceNum = numNamespaces++;
    allNamespaces.push_back(this);
    //fprintf(stderr,"new Namespace %d",namespaceNum);
    //fprintf(stderr,".");
}

VrmlNamespace::~VrmlNamespace()
{
    // Free nameList
    list<VrmlNode *>::iterator n;
    for (n = d_nameList.begin(); n != d_nameList.end(); ++n)
        (*n)->dereference();

    // Free typeList
    list<VrmlNodeType *>::iterator i;
    for (i = d_typeList.begin(); i != d_typeList.end(); ++i)
        (*i)->dereference();

    //fprintf(stderr,"remove Namespace %d",namespaceNum);
    // remove myself from allNamespaces

    allNamespaces.remove(this);
    /*       NamespaceList::iterator it;
   for (it = allNamespaces.begin(); it != allNamespaces.end(); it++)
   {
      if(*it==this)
      {
         break;
      }
   }
   */
}

//
//  Built in nodes.
//  This code replaces the reading of the "standardNodes.wrl" file
//  of empty PROTOs so I don't need to carry that file around.
//

void
VrmlNamespace::addBuiltIn(VrmlNodeType *type)
{
    builtInList.push_front(type->reference());
}

#include "VrmlNodeAnchor.h"
#include "VrmlNodeAppearance.h"
#include "VrmlNodeWave.h"
#include "VrmlNodeBooleanSequencer.h"
#include "VrmlNodeBumpMapping.h"
#include "VrmlNodeAudioClip.h"
#include "VrmlNodeBackground.h"
#include "VrmlNodeBillboard.h"
#include "VrmlNodeBox.h"
#include "VrmlNodeCollision.h"
#include "VrmlNodeColor.h"
#include "VrmlNodeColorInt.h"
#include "VrmlNodeColorRGBA.h"
#include "VrmlNodeCone.h"
#include "VrmlNodeCoordinate.h"
#include "VrmlNodeCoordinateInt.h"
#include "VrmlNodeCylinder.h"
#include "VrmlNodeCylinderSensor.h"
#include "VrmlNodeDirLight.h"
#include "VrmlNodeElevationGrid.h"
#include "VrmlNodeExtrusion.h"
#include "VrmlNodeFog.h"
#include "VrmlNodeFontStyle.h"
#include "VrmlNodeGroup.h"
#include "VrmlNodeIFaceSet.h"
#include "VrmlNodeILineSet.h"
#include "VrmlNodeIQuadSet.h"
#include "VrmlNodeITriangleFanSet.h"
#include "VrmlNodeITriangleSet.h"
#include "VrmlNodeITriangleStripSet.h"
#include "VrmlNodeImageTexture.h"
#include "VrmlNodeCubeTexture.h"
#include "VrmlNodeInline.h"
#include "VrmlNodeLOD.h"
#include "VrmlNodeMaterial.h"
#include "VrmlNodeMetadataBoolean.h"
#include "VrmlNodeMetadataDouble.h"
#include "VrmlNodeMetadataFloat.h"
#include "VrmlNodeMetadataInteger.h"
#include "VrmlNodeMetadataSet.h"
#include "VrmlNodeMetadataString.h"
#include "VrmlNodeMovieTexture.h"
#include "VrmlNodeMultiTexture.h"
#include "VrmlNodeMultiTextureCoordinate.h"
#include "VrmlNodeMultiTextureTransform.h"
#include "VrmlNodeNavigationInfo.h"
#include "VrmlNodeCOVER.h"
#include "VrmlNodeNormal.h"
#include "VrmlNodeNormalInt.h"
#include "VrmlNodeOrientationInt.h"
#include "VrmlNodePixelTexture.h"
#include "VrmlNodePlaneSensor.h"
#include "VrmlNodeSpaceSensor.h"
#include "VrmlNodePointLight.h"
#include "VrmlNodePointSet.h"
#include "VrmlNodePositionInt.h"
#include "VrmlNodeProximitySensor.h"
#include "VrmlNodeQuadSet.h"
#include "VrmlNodeScalarInt.h"
#include "VrmlNodeScript.h"
#include "VrmlNodeShape.h"
#include "VrmlNodeSound.h"
#include "VrmlNodeSphere.h"
#include "VrmlNodeSphereSensor.h"
#include "VrmlNodeSpotLight.h"
#include "VrmlNodeSwitch.h"
#include "VrmlNodeText.h"
#include "VrmlNodeTextureCoordinate.h"
#include "VrmlNodeTextureCoordinateGenerator.h"
#include "VrmlNodeTextureTransform.h"
#include "VrmlNodeTimeSensor.h"
#include "VrmlNodeTouchSensor.h"
#include "VrmlNodeTransform.h"
#include "VrmlNodeTriangleFanSet.h"
#include "VrmlNodeTriangleSet.h"
#include "VrmlNodeTriangleStripSet.h"
#include "VrmlNodeViewpoint.h"
#include "VrmlNodeVisibilitySensor.h"
#include "VrmlNodeWorldInfo.h"

void VrmlNamespace::defineBuiltIns()
{
    addBuiltIn(VrmlNodeAnchor::defineType());
    addBuiltIn(VrmlNodeAppearance::defineType());
    addBuiltIn(VrmlNodeAudioClip::defineType());
    addBuiltIn(VrmlNodeBackground::defineType());
    addBuiltIn(VrmlNodeBillboard::defineType());
    addBuiltIn(VrmlNodeBooleanSequencer::defineType());
    addBuiltIn(VrmlNodeBox::defineType());
    addBuiltIn(VrmlNodeCollision::defineType());
    addBuiltIn(VrmlNodeColor::defineType());
    addBuiltIn(VrmlNodeColorInt::defineType());
    addBuiltIn(VrmlNodeColorRGBA::defineType());
    addBuiltIn(VrmlNodeCone::defineType());
    addBuiltIn(VrmlNodeCoordinate::defineType());
    addBuiltIn(VrmlNodeCoordinateInt::defineType());
    addBuiltIn(VrmlNodeCylinder::defineType());
    addBuiltIn(VrmlNodeCylinderSensor::defineType());
    addBuiltIn(VrmlNodeDirLight::defineType());
    addBuiltIn(VrmlNodeElevationGrid::defineType());
    addBuiltIn(VrmlNodeExtrusion::defineType());
    addBuiltIn(VrmlNodeFog::defineType());
    addBuiltIn(VrmlNodeFontStyle::defineType());
    addBuiltIn(VrmlNodeGroup::defineType());
    addBuiltIn(VrmlNodeIFaceSet::defineType());
    addBuiltIn(VrmlNodeILineSet::defineType());
    addBuiltIn(VrmlNodeIQuadSet::defineType());
    addBuiltIn(VrmlNodeITriangleFanSet::defineType());
    addBuiltIn(VrmlNodeITriangleSet::defineType());
    addBuiltIn(VrmlNodeITriangleStripSet::defineType());
    addBuiltIn(VrmlNodeImageTexture::defineType());
    addBuiltIn(VrmlNodeCubeTexture::defineType());
    addBuiltIn(VrmlNodeInline::defineType());
    addBuiltIn(VrmlNodeLOD::defineType());
    addBuiltIn(VrmlNodeMaterial::defineType());
    addBuiltIn(VrmlNodeMetadataBoolean::defineType());
    addBuiltIn(VrmlNodeMetadataDouble::defineType());
    addBuiltIn(VrmlNodeMetadataFloat::defineType());
    addBuiltIn(VrmlNodeMetadataInteger::defineType());
    addBuiltIn(VrmlNodeMetadataSet::defineType());
    addBuiltIn(VrmlNodeMetadataString::defineType());
    addBuiltIn(VrmlNodeMovieTexture::defineType());
    addBuiltIn(VrmlNodeMultiTexture::defineType());
    addBuiltIn(VrmlNodeMultiTextureCoordinate::defineType());
    addBuiltIn(VrmlNodeMultiTextureTransform::defineType());
    addBuiltIn(VrmlNodeNavigationInfo::defineType());
    addBuiltIn(VrmlNodeCOVER::defineType());
    addBuiltIn(VrmlNodeNormal::defineType());
    addBuiltIn(VrmlNodeNormalInt::defineType());
    addBuiltIn(VrmlNodeOrientationInt::defineType());
    addBuiltIn(VrmlNodePixelTexture::defineType());
    addBuiltIn(VrmlNodePlaneSensor::defineType());
    addBuiltIn(VrmlNodeSpaceSensor::defineType());
    addBuiltIn(VrmlNodePointLight::defineType());
    addBuiltIn(VrmlNodePointSet::defineType());
    addBuiltIn(VrmlNodePositionInt::defineType());
    addBuiltIn(VrmlNodeProximitySensor::defineType());
    addBuiltIn(VrmlNodeQuadSet::defineType());
    addBuiltIn(VrmlNodeScalarInt::defineType());
    addBuiltIn(VrmlNodeScript::defineType());
    addBuiltIn(VrmlNodeShape::defineType());
    addBuiltIn(VrmlNodeSound::defineType());
    addBuiltIn(VrmlNodeSphere::defineType());
    addBuiltIn(VrmlNodeSphereSensor::defineType());
    addBuiltIn(VrmlNodeSpotLight::defineType());
    addBuiltIn(VrmlNodeSwitch::defineType());
    addBuiltIn(VrmlNodeText::defineType());
    addBuiltIn(VrmlNodeTextureCoordinate::defineType());
    addBuiltIn(VrmlNodeTextureCoordinateGenerator::defineType());
    addBuiltIn(VrmlNodeTextureTransform::defineType());
    addBuiltIn(VrmlNodeTimeSensor::defineType());
    addBuiltIn(VrmlNodeTouchSensor::defineType());
    addBuiltIn(VrmlNodeTransform::defineType());
    addBuiltIn(VrmlNodeTriangleFanSet::defineType());
    addBuiltIn(VrmlNodeTriangleSet::defineType());
    addBuiltIn(VrmlNodeTriangleStripSet::defineType());
    addBuiltIn(VrmlNodeViewpoint::defineType());
    addBuiltIn(VrmlNodeVisibilitySensor::defineType());
    addBuiltIn(VrmlNodeWave::defineType());
    addBuiltIn(VrmlNodeBumpMapping::defineType());
    addBuiltIn(VrmlNodeWorldInfo::defineType());
}

// A safer version for reading PROTOs from files.

void
VrmlNamespace::addNodeType(VrmlNodeType *type)
{
    if (findOnlyType(type->getName()) != NULL)
        System::the->warn("PROTO %s already defined\n",
                          type->getName());
    else
        d_typeList.push_front(type->reference());
}

const VrmlNodeType *
VrmlNamespace::findType(const char *name)
{
    // Look through the PROTO stack:
    const VrmlNodeType *nt = findPROTO(name);
    if (nt)
        return nt;

    // Look in parent scope for the type
    if (d_parent)
        return d_parent->findOnlyType(name);

    // Look through the built ins
    list<VrmlNodeType *>::iterator i;
    for (i = builtInList.begin(); i != builtInList.end(); ++i)
    {
        nt = *i;
        //printf(" %s %s\n", name, nt->getName());
        if (nt != NULL && strcmp(nt->getName(), name) == 0)
            return nt;
    }
    // Neither Nodetype nor Proto found, so try to load a plugin with the name of the Node
    if (System::the->loadPlugin(name))
    { // if we managed to load the plugin, we can try to find its nodeType, if the plugin defined a type

        for (i = builtInList.begin(); i != builtInList.end(); ++i)
        {
            const VrmlNodeType *nt = *i;
            if (nt != NULL && strcmp(nt->getName(), name) == 0)
                return nt;
        }
    }

    return NULL;
}

const VrmlNodeType *
VrmlNamespace::findOnlyType(const char *name)
{
    // Look through the PROTO stack:
    const VrmlNodeType *nt = findPROTO(name);
    if (nt)
        return nt;

    // Look in parent scope for the type
    if (d_parent)
        return d_parent->findOnlyType(name);

    // Look through the built ins
    list<VrmlNodeType *>::iterator i;
    for (i = builtInList.begin(); i != builtInList.end(); ++i)
    {
        nt = *i;
        if (nt != NULL && strcmp(nt->getName(), name) == 0)
            return nt;
    }

    return NULL;
}

const VrmlNodeType * // LarryD
    VrmlNamespace::findPROTO(const char *name)
{
    // Look through the PROTO list ONLY:
    list<VrmlNodeType *>::iterator i;
    for (i = d_typeList.begin(); i != d_typeList.end(); ++i)
    {
        const VrmlNodeType *nt = *i;
        if (nt != NULL && strcmp(nt->getName(), name) == 0)
            return nt;
    }

    return NULL;
}

const VrmlNodeType *
VrmlNamespace::firstType()
{
    // Top of the PROTO stack (should make sure it has an implementation...)
    if (d_typeList.size() > 0)
        return d_typeList.front()->reference();
    return NULL;
}

void
VrmlNamespace::addNodeName(VrmlNode *namedNode)
{
    // We could remove any existing node with this name, but
    // since we are just pushing this one onto the front of
    // the list, the other name won't be found. If we do
    // something smart with this list (like sorting), this
    // will need to change.
    d_nameList.push_front(namedNode->reference());
}

void
VrmlNamespace::removeNodeName(VrmlNode *namedNode)
{
    if (d_nameList.size() > 0)
    {
        d_nameList.remove(namedNode);
        // namedNode->dereference();
    }
}

VrmlNode *VrmlNamespace::findNode(const char *name)
{
    list<VrmlNode *>::iterator n;
    for (n = d_nameList.begin(); n != d_nameList.end(); ++n)
        if (strcmp((*n)->name(), name) == 0)
            return *n;

    return 0;
}

VrmlNode *VrmlNamespace::findNode(const char *name, int num)
{
    VrmlNamespace *ns = NULL;
    if (num >= 0)
    {
        NamespaceList::iterator it;

        for (it = allNamespaces.begin(); it != allNamespaces.end(); it++)
        {
            if ((*it)->getNumber() == num)
            {
                ns = *it;
                break;
            }
        }
    }
    if (ns)
        return ns->findNode(name);
    else
        return NULL;
}

void VrmlNamespace::repairRoutes()
{
    list<VrmlNode *>::iterator n;
    for (n = d_nameList.begin(); n != d_nameList.end(); ++n)
    {
        findNode((*n)->name())->repairRoutes();
    }
}

std::string VrmlNamespace::getExportAs(std::string name)
{
    if (exportAsMap.count(name) > 0)
        return exportAsMap[name];
    return "";
}

void VrmlNamespace::addExportAs(std::string name, std::string exportName)
{
    exportAsMap[name] = exportName;
}
