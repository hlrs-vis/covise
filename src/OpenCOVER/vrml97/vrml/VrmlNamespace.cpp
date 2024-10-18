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
#include "VrmlNodeMetadataSet.h"
#include "VrmlNodeMetadataNumeric.h"
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
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeAnchor>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeAppearance>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeAudioClip>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeBackground>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeBillboard>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeBooleanSequencer>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeBox>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeBumpMapping>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeCollision>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeColor>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeColorInt>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeColorRGBA>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeCone>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeCoordinate>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeCoordinateInt>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeCOVER>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeCubeTexture>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeCylinder>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeCylinderSensor>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeDirLight>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeElevationGrid>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeExtrusion>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeFog>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeFontStyle>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeGroup>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeIFaceSet>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeILineSet>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeImageTexture>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeInline>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeIQuadSet>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeITriangleFanSet>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeITriangleSet>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeITriangleStripSet>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeLOD>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeMaterial>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeMetadataBoolean>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeMetadataDouble>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeMetadataFloat>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeMetadataInteger>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeMetadataSet>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeMetadataString>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeMovieTexture>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeMultiTexture>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeMultiTextureCoordinate>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeMultiTextureTransform>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeNavigationInfo>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeNormal>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeNormalInt>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeOrientationInt>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodePixelTexture>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodePlaneSensor>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodePointLight>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodePointSet>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodePositionInt>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeProximitySensor>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeQuadSet>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeScalarInt>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeScript>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeShape>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeSound>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeSpaceSensor>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeSphere>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeSphereSensor>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeSpotLight>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeSwitch>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeText>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeTextureCoordinate>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeTextureCoordinateGenerator>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeTextureTransform>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeTimeSensor>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeTouchSensor>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeTransform>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeTriangleFanSet>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeTriangleSet>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeTriangleStripSet>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeViewpoint>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeVisibilitySensor>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeWave>());
    addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeWorldInfo>());
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
