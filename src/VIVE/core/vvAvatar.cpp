/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "vvAvatar.h"
#include "vvSceneGraph.h"
#include "vvCollaboration.h"
#include "vvPluginSupport.h"
#include "vvCommunication.h"
#include "vvFileManager.h"
#include "vvBillboard.h"
#include "vvPartner.h"
#include <config/CoviseConfig.h>
#include <vsg/nodes/MatrixTransform.h>
#include <vsg/state/Sampler.h>

namespace vive
{
using namespace covise;

vive::vvAvatar::vvAvatar()
{
    vsg::dmat4 invbase = vv->getInvBaseMat();
    vsg::dmat4 handmat;
    handmat = invbase * vv->getPointerMat();
    vsg::dmat4 headmat = vv->getViewerMat();
    vsg::vec3 toFeet;
    toFeet = getTrans(headmat);
    toFeet[2] = vvSceneGraph::instance()->floorHeight();
    vsg::dmat4 feetmat;
    feetmat = vsg::translate(toFeet[0], toFeet[1], toFeet[2]);
    headmat = invbase * headmat;
    feetmat = invbase * feetmat;

    handTransform = vsg::MatrixTransform::create(handmat);
    headTransform = vsg::MatrixTransform::create(headmat);
    feetTransform = vsg::MatrixTransform::create(feetmat);

}

bool vvAvatar::init(const std::string &nodeName)
{
    if (!initialized)
    {
        avatarNodes = new vsg::Group;
        avatarNodes->addChild(feetTransform);
        avatarNodes->addChild(headTransform);
        avatarNodes->addChild(handTransform);

        brilleNode = vvFileManager::instance()->loadIcon("brille");
        handNode = vvFileManager::instance()->loadIcon("hand");
        schuheNode = vvFileManager::instance()->loadIcon("schuhe");
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

PartnerAvatar::PartnerAvatar(vvPartner *partner)
    :vvAvatar(0)
    ,m_partner(partner)
{
    handTransform = vsg::MatrixTransform::create();
    headTransform = vsg::MatrixTransform::create();
    feetTransform = vsg::MatrixTransform::create();
    avatarNodes = nullptr;
}

bool PartnerAvatar::init(const std::string &hostAdress)
{
    if (!initialized)
    {
        if(!vvAvatar::init("Avatar " + hostAdress))
            return false;
        initialized = true;

        if (vvCollaboration::instance()->showAvatar)
        {
            vv->getObjectsRoot()->addChild(avatarNodes);
        }

        return true;
    }
    return false;
}

vvAvatar::~vvAvatar()
{
    vvPluginSupport::removeChild(vv->getObjectsRoot(), avatarNodes);
}

void vvAvatar::show()
{
    if (!visible)
    {
        vv->getObjectsRoot()->addChild(avatarNodes);
        visible = true;
    }
}

void vvAvatar::hide()
{
    if (!visible)
    {
        vvPluginSupport::removeChild(vv->getObjectsRoot(),avatarNodes);
        visible = false;
    }
}

void PartnerAvatar::loadPartnerIcon()
{
    if (!m_partner->userInfo().icon.empty())
    {
        hostIconNode = vvFileManager::instance()->loadIcon(m_partner->userInfo().icon.c_str());
        if (!hostIconNode)
        {
            auto iconFile = vvFileManager::instance()->findOrGetFile(m_partner->userInfo().icon, m_partner->ID());
            hostIconNode = vvFileManager::instance()->loadIcon(iconFile.c_str());
            if (!hostIconNode)
                cerr << "host icon not found " << iconFile << endl;
        }
    }
    vsg::ref_ptr<vvBillboard> bb = vvBillboard::create();
    feetTransform->addChild(bb);
    if (hostIconNode)
    {
        bb->addChild(hostIconNode);
    }
}


RecordedAvatar::RecordedAvatar() : m_icon(covise::coCoviseConfig::getEntry("value", "VIVE.Collaborative.Icon", "$COVISE_PATH/share/covise/icons/hosts/localhost.obj"))
{
    m_head.push_back(headTransform->matrix);
    m_hand.push_back(handTransform->matrix);
    m_feet.push_back(feetTransform->matrix);
}

bool RecordedAvatar::init()
{
    if(!initialized)
    {
        static size_t num = 0;
        if(!vvAvatar::init("Recorded avatar num " + std::to_string(num++)))
            return false;
        hostIconNode = vvFileManager::instance()->loadIcon(UserInfo(Program::vive).icon.c_str());
        if (hostIconNode)
        {
            vsg::ref_ptr<vvBillboard> bb = vvBillboard::create();
            feetTransform->addChild(bb);
            bb->addChild(hostIconNode);
        }
        initialized = true;
        return true;
    }
    return false;
}

covise::TokenBuffer &operator<<(covise::TokenBuffer &tb, const vive::vvAvatar &avatar)
{
    covise::serialize(tb, avatar.headTransform->matrix);
    covise::serialize(tb, avatar.handTransform->matrix);
    covise::serialize(tb, avatar.feetTransform->matrix);
    return tb;
}
covise::TokenBuffer &operator>>(covise::TokenBuffer &tb, vive::vvAvatar &avatar)
{
    vsg::dmat4 head, hand, feet;
    covise::deserialize(tb, head);
    covise::deserialize(tb, hand);
    covise::deserialize(tb, feet);
    avatar.headTransform->matrix = (head);
    avatar.handTransform->matrix = (hand);
    avatar.feetTransform->matrix = (feet);
    return tb;
}
}

