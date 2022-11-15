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
#include <config/CoviseConfig.h>
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

bool VRAvatar::init(const std::string &nodeName)
{
    if (!initialized)
    {
        avatarNodes = new osg::Group;
        avatarNodes->addChild(feetTransform);
        avatarNodes->addChild(headTransform);
        avatarNodes->addChild(handTransform);
        avatarNodes->setName(nodeName);
        osg::StateSet *ss = avatarNodes->getOrCreateStateSet();
        for (int i = 0; i < cover->getNumClipPlanes(); i++)
        {
            ss->setAttributeAndModes(cover->getClipPlane(i), osg::StateAttribute::OFF);
        }

        brilleNode = coVRFileManager::instance()->loadIcon("brille");
        handNode = coVRFileManager::instance()->loadIcon("hand");
        schuheNode = coVRFileManager::instance()->loadIcon("schuhe");
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
        return true;
    }
    return false;
}

PartnerAvatar::PartnerAvatar(coVRPartner *partner)
    :VRAvatar(0)
    ,m_partner(partner)
{
    handTransform = new osg::MatrixTransform;
    headTransform = new osg::MatrixTransform;
    feetTransform = new osg::MatrixTransform;
    avatarNodes = nullptr;
}

bool PartnerAvatar::init(const std::string &hostAdress)
{
    if (!initialized)
    {
        if(!VRAvatar::init("Avatar " + hostAdress))
            return false;
        initialized = true;

        if (coVRCollaboration::instance()->showAvatar)
        {
            cover->getObjectsRoot()->addChild(avatarNodes.get());
        }

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

void PartnerAvatar::loadPartnerIcon()
{
    if (!m_partner->userInfo().icon.empty())
    {
        hostIconNode = coVRFileManager::instance()->loadIcon(m_partner->userInfo().icon.c_str());
        if (!hostIconNode)
        {
            auto iconFile = coVRFileManager::instance()->findOrGetFile(m_partner->userInfo().icon, m_partner->ID());
            hostIconNode = coVRFileManager::instance()->loadIcon(iconFile.c_str());
            if (!hostIconNode)
                cerr << "host icon not found " << iconFile << endl;
        }
    }
    coBillboard *bb = new coBillboard;
    feetTransform->addChild(bb);
    if (hostIconNode)
    {
        bb->addChild(hostIconNode);
    }
}


RecordedAvatar::RecordedAvatar() : m_icon(covise::coCoviseConfig::getEntry("value", "COVER.Collaborative.Icon", "$COVISE_PATH/share/covise/icons/hosts/localhost.obj"))
{
    m_head.push_back(headTransform->getMatrix());
    m_hand.push_back(handTransform->getMatrix());
    m_feet.push_back(feetTransform->getMatrix());
}

bool RecordedAvatar::init()
{
    if(!initialized)
    {
        static size_t num = 0;
        if(!VRAvatar::init("Recorded avatar num " + std::to_string(num++)))
            return false;
        hostIconNode = coVRFileManager::instance()->loadIcon(UserInfo(Program::opencover).icon.c_str());
        if (hostIconNode)
        {
            coBillboard *bb = new coBillboard;
            feetTransform->addChild(bb);
            bb->addChild(hostIconNode);
        }
        initialized = true;
        return true;
    }
    return false;
}

covise::TokenBuffer &operator<<(covise::TokenBuffer &tb, const opencover::VRAvatar &avatar)
{
    covise::serialize(tb, avatar.headTransform->getMatrix());
    covise::serialize(tb, avatar.handTransform->getMatrix());
    covise::serialize(tb, avatar.feetTransform->getMatrix());
    return tb;
}
covise::TokenBuffer &operator>>(covise::TokenBuffer &tb, opencover::VRAvatar &avatar)
{
    osg::Matrix head, hand, feet;
    covise::deserialize(tb, head);
    covise::deserialize(tb, hand);
    covise::deserialize(tb, feet);
    avatar.headTransform->setMatrix(head);
    avatar.handTransform->setMatrix(hand);
    avatar.feetTransform->setMatrix(feet);
    return tb;
}
}

