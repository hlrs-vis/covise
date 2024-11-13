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
#include <set>
using std::list;

using namespace vrml;

list<VrmlNodeType *> VrmlNamespace::builtInList;
// This is at least be a sorted vector...
NamespaceList VrmlNamespace::allNamespaces;
bool VrmlNamespace::definedBuiltins;

static std::map<int, std::set<int>> numNamespaces{ std::pair<int, std::set<int>>(-1, std::set<int>{0}) };

VrmlNamespace::VrmlNamespace(VrmlNamespace *parent)
    : d_parent(parent)
{
	if (parent)
	{
		init(parent->getNumber().first);
	}
	else
	{
		init(-1);
	}
}

vrml::VrmlNamespace::VrmlNamespace(int parentId)
	:d_parent(nullptr)
{
	init(parentId);

}

VrmlNamespace::~VrmlNamespace()
{
    // Free nameList
    for (auto &n: d_nameList)
        n.second->dereference();

    // Free typeList
    list<VrmlNodeType *>::iterator i;
    for (i = d_typeList.begin(); i != d_typeList.end(); ++i)
        (*i)->dereference();

    //fprintf(stderr,"remove Namespace %d",namespaceNum);
    // remove myself from allNamespaces

    allNamespaces.remove(this);
	//reset namespace counter 
	bool found = false;
	for (auto ns : allNamespaces)
	{
		if (ns->namespaceNum ==namespaceNum)
		{
			found = true;
			break;
		}
	}
	if (!found)
	{
		numNamespaces[namespaceNum.first].erase(namespaceNum.second);
	}

}

void vrml::VrmlNamespace::init(int parentId)
{
	// Initialize typeList with built in nodes
	if (!definedBuiltins)
	{
		definedBuiltins = true;
		defineBuiltIns();
	}
	auto &it = numNamespaces[parentId];
	int id = 0;
	do
	{
		++id;
	} while (!it.insert(id).second);
	namespaceNum = NamespaceNum(parentId, id);
	allNamespaces.push_back(this);
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
#include "VrmlNodeAudioClip.h"
#include "VrmlNodeBackground.h"
#include "VrmlNodeBillboard.h"
#include "VrmlNodeBooleanSequencer.h"
#include "VrmlNodeBox.h"
#include "VrmlNodeBumpMapping.h"
#include "VrmlNodeCollision.h"
#include "VrmlNodeColor.h"
#include "VrmlNodeColorInt.h"
#include "VrmlNodeColorRGBA.h"
#include "VrmlNodeCone.h"
#include "VrmlNodeCoordinate.h"
#include "VrmlNodeCoordinateInt.h"
#include "VrmlNodeCOVER.h"
#include "VrmlNodeCubeTexture.h"
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
#include "VrmlNodeImageTexture.h"
#include "VrmlNodeInline.h"
#include "VrmlNodeIQuadSet.h"
#include "VrmlNodeITriangleFanSet.h"
#include "VrmlNodeITriangleSet.h"
#include "VrmlNodeITriangleStripSet.h"
#include "VrmlNodeLOD.h"
#include "VrmlNodeMaterial.h"
#include "VrmlNodeMetadataNumeric.h"
#include "VrmlNodeMetadataSet.h"
#include "VrmlNodeMovieTexture.h"
#include "VrmlNodeMultiTexture.h"
#include "VrmlNodeMultiTextureCoordinate.h"
#include "VrmlNodeMultiTextureTransform.h"
#include "VrmlNodeNavigationInfo.h"
#include "VrmlNodeNormal.h"
#include "VrmlNodeNormalInt.h"
#include "VrmlNodeOrientationInt.h"
#include "VrmlNodePixelTexture.h"
#include "VrmlNodePlaneSensor.h"
#include "VrmlNodePointLight.h"
#include "VrmlNodePointSet.h"
#include "VrmlNodePositionInt.h"
#include "VrmlNodeProto.h"
#include "VrmlNodeProximitySensor.h"
#include "VrmlNodeQuadSet.h"
#include "VrmlNodeScalarInt.h"
#include "VrmlNodeScript.h"
#include "VrmlNodeShape.h"
#include "VrmlNodeSound.h"
#include "VrmlNodeSpaceSensor.h"
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
#include "VrmlNodeWave.h"
#include "VrmlNodeWorldInfo.h"


void vrml::VrmlNamespace::resetNamespaces(int parentId)
{
	auto it = allNamespaces.begin();
	while (it != allNamespaces.end())
	{
		if ((*it)->getNumber().first == parentId)
		{
			it = allNamespaces.erase(it);
		}
		else
		{
			++it;
		}
	}
	numNamespaces.erase(parentId);
}

void VrmlNamespace::defineBuiltIns()
{
    addBuiltIn(VrmlNode::defineType<VrmlNodeAnchor>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeAppearance>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeAudioClip>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeBackground>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeBillboard>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeBooleanSequencer>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeBox>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeBumpMapping>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeCollision>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeColor>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeColorInt>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeColorRGBA>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeCone>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeCoordinate>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeCoordinateInt>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeCOVER>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeCubeTexture>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeCylinder>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeCylinderSensor>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeDirLight>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeElevationGrid>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeExtrusion>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeFog>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeFontStyle>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeGroup>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeIFaceSet>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeILineSet>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeImageTexture>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeInline>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeIQuadSet>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeITriangleFanSet>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeITriangleSet>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeITriangleStripSet>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeLOD>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeMaterial>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeMetadataBoolean>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeMetadataDouble>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeMetadataFloat>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeMetadataInteger>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeMetadataSet>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeMetadataString>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeMovieTexture>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeMultiTexture>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeMultiTextureCoordinate>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeMultiTextureTransform>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeNavigationInfo>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeNormal>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeNormalInt>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeOrientationInt>());
    addBuiltIn(VrmlNode::defineType<VrmlNodePixelTexture>());
    addBuiltIn(VrmlNode::defineType<VrmlNodePlaneSensor>());
    addBuiltIn(VrmlNode::defineType<VrmlNodePointLight>());
    addBuiltIn(VrmlNode::defineType<VrmlNodePointSet>());
    addBuiltIn(VrmlNode::defineType<VrmlNodePositionInt>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeProto>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeProximitySensor>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeQuadSet>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeScalarInt>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeScript>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeShape>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeSound>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeSpaceSensor>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeSphere>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeSphereSensor>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeSpotLight>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeSwitch>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeText>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeTextureCoordinate>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeTextureCoordinateGenerator>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeTextureTransform>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeTimeSensor>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeTouchSensor>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeTransform>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeTriangleFanSet>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeTriangleSet>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeTriangleStripSet>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeViewpoint>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeVisibilitySensor>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeWave>());
    addBuiltIn(VrmlNode::defineType<VrmlNodeWorldInfo>());
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
    d_nameList[namedNode->name()] = namedNode->reference();
}

void
VrmlNamespace::removeNodeName(VrmlNode *namedNode)
{
    auto it = d_nameList.find(namedNode->name());
    if (it != d_nameList.end())
    {
        d_nameList.erase(it);
        // namedNode->dereference();
    }
}

VrmlNode *VrmlNamespace::findNode(const char *name)
{
    auto it = d_nameList.find(name);
    if (it != d_nameList.end())
    {
        return it->second;
    }

    if(strncmp(name,"global::",8)==0) // search in all namespaces
    {
        NamespaceList::iterator it;

        for (it = allNamespaces.begin(); it != allNamespaces.end(); it++)
        {
                VrmlNode *n = (*it)->findNode(name+8);
                if(n)
                    return n;
        }
    }
    return 0;
}

VrmlNode *VrmlNamespace::findNode(const char *name, NamespaceNum num)
{
    VrmlNamespace *ns = NULL;
    if (num.second >= 0)
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
    for (auto &n: d_nameList)
    {
        auto node = findNode(n.second->name());
        if (node)
            node->repairRoutes();
        else
            std::cerr << "VrmlNamespace::repairRoutes: did not find node " << n.second->name() << std::endl;
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
