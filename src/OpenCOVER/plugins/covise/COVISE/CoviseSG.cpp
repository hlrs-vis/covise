#include "CoviseSG.h"

#include <iostream>

#include <osg/Sequence>
#include <osg/MatrixTransform>

#include <cover/coVRPluginSupport.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRSelectionManager.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRPluginList.h>
#include <cover/VRRegisterSceneGraph.h>
#include <cover/VRSceneGraph.h>

using namespace opencover;
using osg::Vec3;
using osg::Vec4;

CoviseSG::CoviseSG(coVRPlugin *plugin): m_plugin(plugin)
{
    sgDebug_ = getenv("COVISE_SG_DEBUG") != NULL;
    hostName_ = getenv("HOST");
}

CoviseSG::~CoviseSG()
{
    m_addedNodeList.clear();
}

void CoviseSG::deleteNode(const char *nodeName, bool isGroup)
{
    if (cover->debugLevel(3))
    {
        if (nodeName)
            fprintf(stderr, "CoviseSG::deleteNode %s\n", nodeName);
        else
            fprintf(stderr, "CoviseSG::deleteNode NULL\n");
    }
    if (nodeName == NULL)
    {
        return;
    }

    auto itf = m_attachedFileList.find(nodeName);
    if (itf != m_attachedFileList.end())
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "deleteNode: unloading file attached to %s\n", nodeName);
        coVRFileManager::instance()->unloadFile(itf->second.c_str());
        m_attachedFileList.erase(itf);
    }

    NodeList::iterator it = m_attachedNodeList.find(nodeName);
    if (it != m_attachedNodeList.end())
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "deleteNode: deleting node attached to %s\n", nodeName);
        if (it->second)
        {
            VRRegisterSceneGraph::instance()->unregisterNode(it->second, cover->getObjectsRoot()->getName());
            cover->getObjectsRoot()->removeChild(it->second);
        }
        m_attachedNodeList.erase(it);
    }

    LabelList::iterator it2 = m_attachedLabelList.find(nodeName);
    if (it2 != m_attachedLabelList.end())
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "deleteNode: deleting label attached to %s\n", nodeName);
        if (it2->second)
        {
            delete it2->second;
        }
        m_attachedLabelList.erase(it2);
    }

    //printf("...... looking for node %s\n", (char *) data);

    osg::Node *node = findNode(nodeName);
    if (node)
    {
        m_addedNodeList.erase(nodeName);
        VRSceneGraph::instance()->setNodeBounds(node, NULL);

        osg::Group *dcs = node->getNumParents() > 0 ? node->getParent(0) : NULL;
        while (coVRSelectionManager::instance()->isHelperNode(dcs))
        {
            if (dcs->getNumParents() == 0)
            {
                std::cerr << "ERROR: dcs w/o parent: " << dcs->getName() << ", node: " << node->getName() << std::endl;
                break;
            }
            dcs = dcs->getNumParents() > 0 ? dcs->getParent(0) : NULL;
        }
        if (dcs->getNumParents() > 0 && dcs->getParent(0)->getName() == nodeName)
            dcs = dcs->getParent(0);
        coVRPluginList::instance()->removeNode(dcs, isGroup, node);

        if (osg::Sequence *seq = dynamic_cast<osg::Sequence *>(node))
        {
            coVRAnimationManager::instance()->removeSequence(seq);
        }

        //osg::Group *parent = dcs->getParent(0);

        //dcs->removeChild(node);
        cover->removeNode(node, isGroup);

        //parent->removeChild(dcs);
        //cover->removeNode(dcs);
    }
    else
    {
        if (cover->debugLevel(3))
            printf("CoviseSG::deleteNode: node %s not found\n", nodeName);
    }
}

osg::Node *CoviseSG::findNode(const std::string &name)
{
    NodeList::iterator it = m_addedNodeList.find(name);
    if (it != m_addedNodeList.end())
    {
        if (it->second)
            return it->second;
    }
    return NULL;
}

void CoviseSG::attachNode(const char *attacheeName, osg::Node *node, const char *filename)
{
    m_attachedNodeList[attacheeName] = node;
    if (filename)
    {
        m_attachedFileList[attacheeName] = filename;
    }
}

void CoviseSG::attachLabel(const char *attacheeName, const char *text)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "attachLabel(name=%s, string=%s)\n", attacheeName, text);

    coVRLabel *label = new coVRLabel(text, 20.f, 20.f,
                                     Vec4(1., 1., 1., 1.), Vec4(.0, .0, .0, 1.));

    osg::Node *node = findNode(attacheeName);
    if (node)
    {
        osg::Group *dcs = node->getParent(0);
        if (dcs)
        {
            label->reAttachTo(dcs);
        }
    }
    label->show();

    m_attachedLabelList[attacheeName] = label;
}

void CoviseSG::addNode(osg::Node *node, const char *parentName, RenderObject *ro)
{
    //fprintf(stderr,"CoviseSG::addNode1 node=%p, parentName=%s objectName= %s\n", node, parentName, ro->getName());
    osg::Group *parentGroup = dynamic_cast<osg::Group *>(findNode(parentName));
    if (parentName && !parentGroup && cover->debugLevel(3))
        fprintf(stderr, "CoviseSG::addNode: no Group node with name %s found\n", parentName);

    addNode(node, parentGroup, ro);
}

void CoviseSG::addNode(osg::Node *node, osg::Group *parent, RenderObject *ro)
{

    if (sgDebug_)
        fprintf(stderr, "CoviseSG(%s)::addNode2 node=%p, parentNode=%p objectName= %s\n", hostName_, node, parent, ro->getName());


    //put a dcs above a geode - this is used by VRRotator
    osg::MatrixTransform *dcs = new osg::MatrixTransform();
    dcs->addChild(node);
    std::string name = node->getName();
    node->setName(name + "_Geom");
    osg::Group *root = dcs;
    if (parent)
    {
        dcs->setName(name);
    }
    else
    {
        root = new osg::ClipNode;
        root->setName(name);
        dcs->setName(name + "_Mat");
        root->addChild(dcs);
    }

    // disable intersection with ray
    if (const char *isect = ro->getAttribute("DO_ISECT"))
    {
    }
    else
    {
        node->setNodeMask(node->getNodeMask() & (~Isect::Intersection) & (~Isect::Update));
    }

    m_addedNodeList[root->getName()] = node;

    if (parent == NULL)
    {
        VRSceneGraph::instance()->objectsRoot()->addChild(root);
        if (sgDebug_)
            fprintf(stderr, "CoviseSG(%s)::addNode2 adding node to objectsRoot\n", hostName_);

        coVRPluginList::instance()->addNode(root, ro, m_plugin);
    }
    else
    {
        parent->addChild(root);
        if (sgDebug_)
            fprintf(stderr, "CoviseSG(%s)::addNode2 adding node to parent\n", hostName_);

        //m_addedNodeList[dcs->getName()] = node;
    }

    VRSceneGraph::instance()->adjustScale();
}
