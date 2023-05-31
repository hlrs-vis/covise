/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                            (C)2008 HLRS   **
 **                                                                           **
 ** Description: WSInterface Plugin                                           **
 **                                                                           **
 **                                                                           **
 ** Author: Andreas Kopecki                                                   **
 **         M. Baalcke  		                                      **
 **                                                                           **
 **                                                                           **
 **                                                                           **
\****************************************************************************/

#include "WSInterfacePlugin.h"
#include "WSServer.h"

#include <cover/coVRPluginSupport.h>
#include <cover/OpenCOVER.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRSelectionManager.h>
#include <vrb/client/VRBClient.h>

#include <PluginUtil/PluginMessageTypes.h>

#include <QRegExp>
#include <QTextStream>

WSInterfacePlugin *WSInterfacePlugin::singleton = 0;

WSInterfacePlugin *WSInterfacePlugin::instance()
{
    return WSInterfacePlugin::singleton;
}

WSInterfacePlugin::WSInterfacePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    WSInterfacePlugin::singleton = this;
    new WSServer();
}

// this is called if the plugin is removed at runtime
WSInterfacePlugin::~WSInterfacePlugin()
{
}

void WSInterfacePlugin::openFile(const QString &filename)
{
    lock.lock();
    queue.append("load");
    queue.append(filename);
    lock.unlock();
}

void WSInterfacePlugin::addFile(const QString &filename)
{
    lock.lock();
    queue.append("addFile");
    queue.append(filename);
    lock.unlock();
}

void WSInterfacePlugin::quit()
{
    lock.lock();
    queue.append("quit");
    lock.unlock();
}

void WSInterfacePlugin::connectToVnc(const QString &host, unsigned int port, const QString &password)
{
    lock.lock();
    queue.append("connectToVnc");
    queue.append(host);
    queue.append(QString::number(port));
    queue.append(password);
    lock.unlock();
}

void WSInterfacePlugin::disconnectFromVnc()
{
    lock.lock();
    queue.append("disconnectFromVnc");
    lock.unlock();
}

void WSInterfacePlugin::setVisibleVnc(bool on)
{
    lock.lock();
    queue.append("setVisibleVnc");
    queue.append(on ? "on" : "off");
    lock.unlock();
}

void WSInterfacePlugin::snapshot(const QString &path)
{
    lock.lock();
    queue.append("snapshot");
    queue.append(path);
    lock.unlock();
}

void WSInterfacePlugin::sendCustomMessage(const QString &message)
{
    std::cerr << "WSInterfacePlugin::sendCustomMessage info: got custom message '"
              << qPrintable(message) << "'" << std::endl;
    lock.lock();
    queue.append("sendCustomMessage");
    queue.append(message);
    lock.unlock();
}

void WSInterfacePlugin::show(const QString &object)
{
    std::cerr << "WSInterfacePlugin::show info: showing '"
              << qPrintable(object) << "'" << std::endl;
    lock.lock();
    queue.append("show");
    queue.append(object);
    lock.unlock();
}

void WSInterfacePlugin::hide(const QString &object)
{
    std::cerr << "WSInterfacePlugin::hide info: hiding '"
              << qPrintable(object) << "'" << std::endl;
    lock.lock();
    queue.append("hide");
    queue.append(object);
    lock.unlock();
}

void WSInterfacePlugin::viewAll()
{
    lock.lock();
    queue.append("viewall");
    lock.unlock();
}

void WSInterfacePlugin::resetView()
{
    lock.lock();
    queue.append("resetview");
    lock.unlock();
}

void WSInterfacePlugin::walk()
{
    lock.lock();
    queue.append("walk");
    lock.unlock();
}

void WSInterfacePlugin::fly()
{
    lock.lock();
    queue.append("fly");
    lock.unlock();
}

void WSInterfacePlugin::drive()
{
    lock.lock();
    queue.append("drive");
    lock.unlock();
}

void WSInterfacePlugin::scale()
{
    lock.lock();
    queue.append("scale");
    lock.unlock();
}

void WSInterfacePlugin::xform()
{
    lock.lock();
    queue.append("xform");
    lock.unlock();
}

void WSInterfacePlugin::wireframe(bool on)
{
    lock.lock();
    queue.append("wireframe");
    queue.append(on ? "on" : "off");
    lock.unlock();
}

void WSInterfacePlugin::preFrame()
{

    unsigned int queueLength;

    lock.lock();
    if (coVRMSController::instance()->isCluster())
    {
        if (coVRMSController::instance()->isMaster())
        {
            queueLength = queue.size();
            coVRMSController::instance()->sendSlaves((char *)&queueLength, sizeof(unsigned int));

            for (QStringList::const_iterator i = queue.begin(); i != queue.end(); ++i)
            {
                int length = (i->length() + 1) * sizeof(ushort);
                const ushort *buf = i->utf16();
                coVRMSController::instance()->sendSlaves((char *)&length, sizeof(length));
                coVRMSController::instance()->sendSlaves((char *)buf, length);
            }
        }
        else
        {
            coVRMSController::instance()->readMaster((char *)&queueLength, sizeof(unsigned int));

            int length = 0;

            int bufsize = -1;
            char *buf = 0;

            for (int ctr = 0; ctr < queueLength; ++ctr)
            {
                coVRMSController::instance()->readMaster((char *)&length, sizeof(length));

                if (bufsize < length)
                {
                    delete[] buf;
                    buf = new char[length];
                    bufsize = length;
                }

                coVRMSController::instance()->readMaster(buf, length);

                queue.push_back(QString::fromUtf16((ushort *)buf));
            }
        }
    }

    while (!queue.empty())
    {
        QString command = queue.takeFirst();

        if (command == "quit")
        {
            OpenCOVER::instance()->setExitFlag(true);
            coVRPluginList::instance()->requestQuit();
            if (vrbc)
                delete vrbc;
            vrbc = 0;
            OpenCOVER::instance()->setExitFlag(1);
        }
        else if (command.startsWith("load"))
        {
            QString uri = queue.takeFirst();
            if (uri.startsWith("file://"))
                uri.remove(QRegExp("^file://"));
            // TODO replace and load
            coVRFileManager::instance()->replaceFile(uri.toLatin1().data());
        }
        else if (command == "addFile")
        {
            QString uri = queue.takeFirst();
            if (uri.startsWith("file://"))
                uri.remove(QRegExp("^file://"));
            coVRFileManager::instance()->loadFile(uri.toLatin1().data());
        }
        else if (command == "viewall")
        {
            VRSceneGraph::instance()->viewAll();
        }
        else if (command == "resetview")
        {
            VRSceneGraph::instance()->viewAll(true);
        }
        else if (command == "walk")
        {
            cover->enableNavigation("Walk");
        }
        else if (command == "fly")
        {
            cover->enableNavigation("Fly");
        }
        else if (command == "drive")
        {
            cover->enableNavigation("Drive");
        }
        else if (command == "scale")
        {
            cover->enableNavigation("Scale");
        }
        else if (command == "xform")
        {
            cover->enableNavigation("XForm");
        }
        else if (command == "wireframe")
        {
            if (queue.takeFirst() == "on")
                VRSceneGraph::instance()->setWireframe(true);
            else
                VRSceneGraph::instance()->setWireframe(false);
        }
        else if (command == "show")
        {
            QString objectName = queue.takeFirst();
            setVisible(objectName, true);
        }
        else if (command == "hide")
        {
            QString objectName = queue.takeFirst();
            setVisible(objectName, false);
        }
        else if (command == "connectToVnc")
        {
            QString host = queue.takeFirst();
            unsigned int port = queue.takeFirst().toUInt();
            QString passwd = queue.takeFirst();

            TokenBuffer tb;
            tb << host.toLatin1().data();
            tb << port;
            tb << passwd.toLatin1().data();
            cover->sendMessage(this, "VNC", PluginMessageTypes::RemoteDTConnectToHost, tb.get_length(), tb.get_data());
            tb.delete_data();
        }
        else if (command == "disconnectFromVnc")
        {
            cover->sendMessage(this, "VNC", PluginMessageTypes::RemoteDTDisconnect, 0, 0);
        }
        else if (command == "setVisibleVnc")
        {
            if (queue.takeFirst() == "on")
            {
                cover->sendMessage(this, "VNC", PluginMessageTypes::RemoteDTShowDesktop, 0, 0);
            }
            else
            {
                cover->sendMessage(this, "VNC", PluginMessageTypes::RemoteDTHideDesktop, 0, 0);
            }
        }
        else if (command == "snapshot")
        {
            QString path = queue.takeFirst();
            TokenBuffer tb;
            tb << path.toLatin1().data();
            cover->sendMessage(this, "PBufferSnapShot", PluginMessageTypes::PBufferDoSnap, tb.get_length(), tb.get_data());
            tb.delete_data();
        }
        else if (command == "sendCustomMessage")
        {
            QString message = queue.takeFirst();

            std::cerr << "WSInterfacePlugin::preFrame info: got custom message '"
                      << qPrintable(message) << "'" << std::endl;

            cover->sendMessage(this, coVRPluginSupport::TO_ALL,
                               PluginMessageTypes::WSInterfaceCustomMessage,
                               strlen(message.toLatin1().data()), message.toLatin1().data());
        }
    }
    lock.unlock();
}

void WSInterfacePlugin::setVisible(const QString &name, bool on)
{
    osg::Node *node = VRSceneGraph::instance()->findFirstNode<osg::Node>(name.toLatin1().data(), true);
    if (node == 0 || node->getParent(0) == 0)
    {
        std::cerr << "WSInterfacePlugin::setVisible err: no valid node for name '" << qPrintable(name) << "' found" << std::endl;
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

            std::cerr << "WSInterfacePlugin::setVisible info: setting visibility of node '" << qPrintable(name) << "' to " << (on ? "on" : "off") << std::endl;

            TokenBuffer tb;
            tb << nodePath;
            tb << parentNodePath;
            cover->sendMessage(this, "SGBrowser", type, tb.get_length(), tb.get_data());
            tb.delete_data();
        }
    }
}

COVERPLUGIN(WSInterfacePlugin)
