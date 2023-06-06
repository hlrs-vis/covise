/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2013 Visenso  **
 **                                                                        **
 ** Description: CoviseConfigShader - add shader via config entries        **
 **                                                                        **
 ** Author: C. Spenrath                                                    **
 **                                                                        **
\****************************************************************************/

#include "CoviseConfigShader.h"

#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRShader.h>
#include <config/CoviseConfig.h>

#include <grmsg/coGRObjSetTransparencyMsg.h>

#include <osgUtil/SmoothingVisitor>

using namespace covise;
using namespace grmsg;

CoviseConfigShader *CoviseConfigShader::plugin = NULL;

//
// Constructor
//
CoviseConfigShader::CoviseConfigShader()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nCoviseConfigShader::CoviseConfigShader\n");
}

//
// Destructor
//
CoviseConfigShader::~CoviseConfigShader()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nCoviseConfigShader::~CoviseConfigShader\n");
}

//
// INIT
//
bool CoviseConfigShader::init()
{
    if (plugin)
        return false;
    if (cover->debugLevel(3))
        fprintf(stderr, "\nCoviseConfigShader::init\n");

    // set plugin
    CoviseConfigShader::plugin = this;

    readConfig();

    return true;
}

void CoviseConfigShader::readConfig()
{
    definitions.clear();

    std::string pluginpath("COVER.Plugin.CoviseConfigShader");

    coCoviseConfig::ScopeEntries viewpointEntries = coCoviseConfig::getScopeEntries(pluginpath, "Scope");
    for (const auto &viewpoint : viewpointEntries)
    {
        const std::string &name = viewpoint.first.c_str();
        std::string regexp;
        std::string shader;
        bool smooth;
        float transparency;
        regexp = coCoviseConfig::getEntry("regexp", pluginpath + "." + name, "");
        shader = coCoviseConfig::getEntry("shader", pluginpath + "." + name, "");
        smooth = coCoviseConfig::isOn("smooth", pluginpath + "." + name, false);
        transparency = coCoviseConfig::getFloat("transparency", pluginpath + "." + name, -1.0);

        if (regexp != "")
        {
            Definition d;
            d.regexp = QRegularExpression(regexp.c_str());
            d.shader = shader;
            d.smooth = smooth;
            d.transparency = transparency;
            definitions.push_back(d);
        }
    }
}

void CoviseConfigShader::addNode(osg::Node *node, const RenderObject *)
{
    // check if we have a PerformerScene node (in which case the geometry is not in node)
    bool isVrml = false;
    osg::Group *group = dynamic_cast<osg::Group *>(node);
    while (group)
    {
        if (group->getName().compare(0, 14, "PerformerScene") == 0)
        {
            isVrml = true;
            break;
        }
        if (group->getNumChildren() > 0)
        {
            group = dynamic_cast<osg::Group *>(group->getChild(0));
        }
        else
        {
            group = NULL;
        }
    }

    if (isVrml)
    {
        node = cover->getObjectsRoot();
    }

    addShader(node);
}

void CoviseConfigShader::addShader(osg::Node *node)
{
    // check node name against all definitions
    for (size_t index = 0; index < definitions.size(); ++index)
    {
        auto nodeName = QString::fromStdString(node->getName());
        auto match = definitions[index].regexp.match(nodeName);
        if (match.hasMatch() && match.capturedLength() == nodeName.length())
        {

            // Transparency
            if (definitions[index].transparency > -0.0001f)
            {
                setTransparency(node, definitions[index].transparency);
            }

            // Shader
            if (definitions[index].shader != "")
            {
                // This is not perfect.
                // vr-prepare adds the shader only on leaf nodes.
                // The problem is, I don't know which nodes are vr-prepare leaf nodes (because of EOT).
                coVRShader *shaderObj = coVRShaderList::instance()->get(definitions[index].shader);
                if (shaderObj)
                {
                    shaderObj->apply(node);
                }
            }

            // Smoothing
            if (definitions[index].smooth)
            {
                osgUtil::SmoothingVisitor sv;
                node->accept(sv);
            }
        }
    }

    osg::Group *group = dynamic_cast<osg::Group *>(node);
    if (group)
    {
        for (size_t i = 0; i < group->getNumChildren(); ++i)
        {
            addShader(group->getChild(i));
        }
    }
}

void CoviseConfigShader::guiToRenderMsg(const grmsg::coGRMsg &msg) 
{
    if (msg.isValid() && msg.getType() == coGRMsg::SET_TRANSPARENCY)
    {
        auto &setTransparencyMsg = msg.as<coGRObjSetTransparencyMsg>();
        const char *objectName = setTransparencyMsg.getObjName();
        transparencyList.push_back(objectName); // we have to delay setting the shader because the transparency needs to be set first
    }
}

void CoviseConfigShader::preFrame()
{
    if (transparencyList.size() == 0)
    {
        return;
    }

    for (size_t i = 0; i < transparencyList.size(); ++i)
    {
        osg::Node *node = VRSceneGraph::instance()->findFirstNode<osg::Node>(transparencyList[i].c_str());
        if (!node)
        {
            return;
        }

        int definitionIndex = getDefinitionIndex(node);
        if (definitionIndex >= 0)
        {

            // Transparency
            if (definitions[definitionIndex].transparency > -0.0001f)
            {
                setTransparency(node, definitions[definitionIndex].transparency);
            }

            // Shader
            std::string shader = definitions[definitionIndex].shader;
            if (shader != "")
            {
                coVRShader *shaderObj = coVRShaderList::instance()->get(shader);
                if (shaderObj)
                {
                    shaderObj->apply(node);
                }
            }
        }
    }

    transparencyList.clear();
}

int CoviseConfigShader::getDefinitionIndex(osg::Node *node)
{
    // check node name against all definitions
    for (size_t i = 0; i < definitions.size(); ++i)
    {
        auto nodeName = QString::fromStdString(node->getName());
        auto match = definitions[i].regexp.match(nodeName);
        if (match.hasMatch() && match.capturedLength() == nodeName.length())
        {
            return i;
        }
    }

    if (node->getNumParents() > 0)
    {
        return getDefinitionIndex(node->getParent(0));
    }
    return -1;
}

void CoviseConfigShader::setTransparency(osg::Node *node, float transparency)
{
    if (node)
    {
        osg::Geode *geode = dynamic_cast<osg::Geode *>(node);
        osg::Group *group = dynamic_cast<osg::Group *>(node);
        if (geode)
        {
            VRSceneGraph::instance()->setTransparency(geode, transparency);
        }
        else if (group)
        {
            for (size_t i = 0; i < group->getNumChildren(); i++)
            {
                setTransparency(group->getChild(i), transparency);
            }
        }
    }
}

COVERPLUGIN(CoviseConfigShader)
