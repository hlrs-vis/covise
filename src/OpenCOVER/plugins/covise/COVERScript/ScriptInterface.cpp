/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ScriptInterface.h"
#include "ScriptPlugin.h"
#include <cover/coVRFileManager.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRSelectionManager.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRPluginList.h>
#include <PluginUtil/PluginMessageTypes.h>
#include <QString>
#include <net/message.h>
#include <net/tokenbuffer.h>

ScriptInterface::ScriptInterface(ScriptPlugin *p)
{
    plugin = p;
}

void ScriptInterface::loadFile(QString file)
{
    coVRFileManager::instance()->loadFile(file.toLatin1().data());
}

void ScriptInterface::snapshotDir(QString dirName)
{
    static bool firstTime = true;
    if (firstTime)
    {
        firstTime = false;
        coVRPluginList::instance()->addPlugin("PBufferSnapShot");
    }
    TokenBuffer tb;
    tb << dirName.toLatin1().data();
    cover->sendMessage(plugin, "PBufferSnapShot", PluginMessageTypes::PBufferDoSnap, tb.get_length(), tb.get_data());
    tb.delete_data();
}
void ScriptInterface::snapshot(QString fileName)
{
    static bool firstTime = true;
    if (firstTime)
    {
        firstTime = false;
        coVRPluginList::instance()->addPlugin("PBufferSnapShot");
    }
    TokenBuffer tb;
    tb << fileName.toLatin1().data();
    cover->sendMessage(plugin, "PBufferSnapShot", PluginMessageTypes::PBufferDoSnapFile, tb.get_length(), tb.get_data());
    tb.delete_data();
}

void ScriptInterface::setVisible(const QString &name, bool on)
{
    osg::Node *node = VRSceneGraph::instance()->findFirstNode<osg::Node>(name.toLatin1().data());
    if (node == 0 || node->getParent(0) == 0)
    {
        std::cerr << "ScriptInterface::setVisible err: no valid node for name '" << qPrintable(name) << "' found" << std::endl;
    }
    else
    {
        osg::Node *parent = node->getParent(0);
        while (parent && coVRSelectionManager::instance()->isHelperNode(parent))
            parent = parent->getParent(0);
        if (parent)
        {
            std::string nodePath = coVRSelectionManager::generatePath(node);
            std::string parentNodePath = coVRSelectionManager::generatePath(parent);

            int type = (on ? PluginMessageTypes::SGBrowserShowNode : PluginMessageTypes::SGBrowserHideNode);

            std::cerr << "ScriptInterface::setVisible info: setting visibility of node '" << qPrintable(name) << "' to " << (on ? "on" : "off") << std::endl;

            TokenBuffer tb;
            tb << nodePath;
            tb << parentNodePath;
            cover->sendMessage(plugin, "SGBrowser", type, tb.get_length(), tb.get_data());
            tb.delete_data();
        }
    }
}

void ScriptInterface::setVariant(const QString &name, bool on)
{

    int type = (on ? PluginMessageTypes::VariantShow : PluginMessageTypes::VariantHide);

    std::cerr << "ScriptInterface::setVariant info: setting variant '" << qPrintable(name) << "' to " << (on ? "on" : "off") << std::endl;

    TokenBuffer tb;
    tb << name.toStdString();
    cover->sendMessage(plugin, "Variant", type, tb.get_length(), tb.get_data());
    tb.delete_data();
}

void ScriptInterface::viewAll(bool resetView)
{
    opencover::VRSceneGraph::instance()->viewAll(resetView);
}

void ScriptInterface::setCuCuttingSurface(int number, float x, float y, float z, float h, float p, float r)
{
    TokenBuffer tb;
    tb << number;
    osg::Matrix mat;
    MAKE_EULER_MAT(mat, h, p, r);
    mat.setTrans(osg::Vec3(x, y, z));
    for (int i = 0; i < 16; i++)
        tb << mat.ptr()[i];
    cover->sendMessage(plugin, "cuCuttingSurface", PluginMessageTypes::HLRS_cuCuttingSurface, tb.get_length(), tb.get_data());
    tb.delete_data();
}
