/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*! \file
 \brief  avatars for partners in colaborative VR sessions

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C)
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */
#pragma once

#include <util/common.h>
#include <net/tokenbuffer.h>
#include <vsg/core/ref_ptr.h>
#include <vsg/maths/mat4.h>
#include <vsg/nodes/MatrixTransform.h>
#include "vvMatrixSerializer.h"
namespace osg
{
class Node;
class Group;
class MatrixTransform;
}

namespace vive
{
class vvPartner;


class VVCORE_EXPORT vvAvatar
{
public:
    bool initialized = false;
    vsg::ref_ptr<vsg::MatrixTransform> handTransform;
    vsg::ref_ptr<vsg::Node> handNode;
    vsg::ref_ptr<vsg::MatrixTransform> headTransform;
    vsg::ref_ptr<vsg::Node> brilleNode;
    vsg::ref_ptr<vsg::MatrixTransform> feetTransform;
    vsg::ref_ptr<vsg::Node> schuheNode;
    vsg::ref_ptr<vsg::Node> hostIconNode;
    vsg::ref_ptr<vsg::Group> avatarNodes;
    ///create an Avatar that only holds the local tramsfom matrices 
    ///Used to collect the data to send it to the partners
    vvAvatar();

    virtual ~vvAvatar();
    void show();
    void hide();

protected:
    //to skip initialization
    vvAvatar(int dummy){};
    /// initalize avatar if not initialized. Return true if sth. is done
    bool init(const std::string &nodeName);
    bool visible = false;
};

///create an Avatar for a remote partner that hold his informations
class VVCORE_EXPORT PartnerAvatar : public vvAvatar
{
public:
    PartnerAvatar(vvPartner *partner);
    bool init(const std::string &hostAdress);
    void loadPartnerIcon();

private:
    vvPartner *m_partner;
};

///Create an avatar that represents a recorded movement
class VVCORE_EXPORT RecordedAvatar : public vvAvatar
{
public:
    RecordedAvatar();
    bool init();

private:
    const std::string m_icon;
    std::vector<vsg::dmat4> m_hand;
    std::vector<vsg::dmat4> m_head;
    std::vector<vsg::dmat4> m_feet;
};

VVCORE_EXPORT covise::TokenBuffer &operator<<(covise::TokenBuffer &tb, const vive::vvAvatar &avatar);
VVCORE_EXPORT covise::TokenBuffer &operator>>(covise::TokenBuffer &tb, vive::vvAvatar &avatar);
}
