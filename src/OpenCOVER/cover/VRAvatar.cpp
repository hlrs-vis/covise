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
#include "coVRPartner.h"
#include <osg/MatrixTransform>
#include <osg/Texture2D>

namespace opencover
{
using namespace covise;

opencover::VRAvatar::VRAvatar()
{
    osg::Matrix invbase = cover->getInvBaseMat();
    osg::Matrix handmat = cover->getPointerMat();
    handmat *= invbase;
    osg::Matrix headmat = cover->getViewerMat();
    osg::Vec3 toFeet;
    toFeet = headmat.getTrans();
    toFeet[2] = VRSceneGraph::instance()->floorHeight();
    osg::Matrix feetmat;
    feetmat.makeTranslate(toFeet[0], toFeet[1], toFeet[2]);
    headmat *= invbase;
    feetmat *= invbase;

    handTransform = new osg::MatrixTransform(handmat);
    headTransform = new osg::MatrixTransform(headmat);
    feetTransform = new osg::MatrixTransform(feetmat);

}

VRAvatar::VRAvatar(coVRPartner *partner)
    :m_partner(partner)
{
    
}

bool VRAvatar::init(const std::string &hostAdress)
{
    if (!initialized)
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
        initialized = true;
        return true;
    }
    return false;
}

VRAvatar::~VRAvatar()
{
    cover->getObjectsRoot()->removeChild(avatarNodes.get());


}
void VRAvatar::show()
{
    if (initialized && avatarNodes->getNumParents() == 0)
    {
        cover->getObjectsRoot()->addChild(avatarNodes.get());
    }
}

void VRAvatar::hide()
{
    if (initialized && avatarNodes->getNumParents() == 1)
    {
        cover->getObjectsRoot()->removeChild(avatarNodes.get());
    }
}

//float VRAvatar::rc[10] = { 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.5f, 0.2f, 0.1f, 0.2f };
//float VRAvatar::gc[10] = { 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.5f, 0.4f, 0.4f, 0.0f };
//float VRAvatar::bc[10] = { 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.1f, 0.6f, 0.7f, 0.7f };

covise::TokenBuffer &operator<<(covise::TokenBuffer &tb, const opencover::VRAvatar &avatar)
{
    vrb::serialize(tb, avatar.headTransform->getMatrix());
    vrb::serialize(tb, avatar.handTransform->getMatrix());
    vrb::serialize(tb, avatar.feetTransform->getMatrix());
    return tb;
}
covise::TokenBuffer &operator>>(covise::TokenBuffer &tb, opencover::VRAvatar &avatar)
{
    osg::Matrix head, hand, feet;
    vrb::deserialize(tb, head);
    vrb::deserialize(tb, hand);
    vrb::deserialize(tb, feet);
    avatar.headTransform->setMatrix(head);
    avatar.handTransform->setMatrix(hand);
    avatar.feetTransform->setMatrix(feet);
    return tb;
}
}

