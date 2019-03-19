/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/byteswap.h>
#include "VRAvatar.h"
#include "VRSceneGraph.h"
#include "coVRCollaboration.h"
#include "coVRPluginSupport.h"
#include "coVRCommunication.h"
#include "coVRFileManager.h"
#include "coBillboard.h"
#include <osg/MatrixTransform>
#include <osg/Texture2D>

using namespace opencover;
using namespace covise;

VRAvatar::VRAvatar(int clientID, const std::string &hostAdress)
    :m_clientID(clientID)
{
    handTransform = new osg::MatrixTransform;
    headTransform = new osg::MatrixTransform;
    feetTransform = new osg::MatrixTransform;
    avatarNodes = new osg::Group;
    avatarNodes->addChild(feetTransform);
    avatarNodes->addChild(headTransform);
    avatarNodes->addChild(handTransform);
    char *NodeName = new char[100 + hostAdress.length()];
    sprintf(NodeName, "Avatar %s", hostAdress.c_str());
    avatarNodes->setName(NodeName);
    osg::StateSet *ss = avatarNodes->getOrCreateStateSet();
    for (int i = 0; i < cover->getNumClipPlanes(); i++)
    {
        ss->setAttributeAndModes(cover->getClipPlane(i), osg::StateAttribute::OFF);
    }

    delete[] NodeName;
    brilleNode = coVRFileManager::instance()->loadIcon("brille");
    handNode = coVRFileManager::instance()->loadIcon("hand");
    schuheNode = coVRFileManager::instance()->loadIcon("schuhe");
    char *hostIcon = new char[6 + hostAdress.length() + 4];
    strcpy(hostIcon, "hosts/");
    strcat(hostIcon, hostAdress.c_str());
    hostIconNode = coVRFileManager::instance()->loadIcon(hostIcon);
    if (hostIconNode == NULL)
    {
        cerr << "Hosticon not found " << hostIcon << endl;
    }
    if (brilleNode)
    {
        headTransform->addChild(brilleNode);
    }
    if (handNode)
    {
        handTransform->addChild(handNode);
    }
    if (schuheNode)
    {
        feetTransform->addChild(schuheNode);
    }
    coBillboard *bb = new coBillboard;
    feetTransform->addChild(bb);
    if (hostIconNode)
    {
        bb->addChild(hostIconNode);
    }
    if (coVRCollaboration::instance()->showAvatar)
    {
        cover->getObjectsRoot()->addChild(avatarNodes.get());
    }
}

VRAvatar::~VRAvatar()
{
    cover->getObjectsRoot()->removeChild(avatarNodes.get());
}
void VRAvatar::show()
{
    if (avatarNodes->getNumParents() == 0)
    {
        cover->getObjectsRoot()->addChild(avatarNodes.get());
    }
}

void VRAvatar::hide()
{
    if (avatarNodes->getNumParents() == 1)
    {
        cover->getObjectsRoot()->removeChild(avatarNodes.get());
    }
}


//float VRAvatar::rc[10] = { 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.5f, 0.2f, 0.1f, 0.2f };
//float VRAvatar::gc[10] = { 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.5f, 0.4f, 0.4f, 0.0f };
//float VRAvatar::bc[10] = { 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.1f, 0.6f, 0.7f, 0.7f };
