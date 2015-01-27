/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * MotionState.h
 *
 *  Created on: 11.10.2012
 *      Author: jw_te
 */

#ifndef MOTIONSTATE_H_
#define MOTIONSTATE_H_

#include <string>
#include <bullet/btBulletDynamicsCommon.h>

namespace osg
{
class MatrixTransform;
}

class KinematicsBehavior;

namespace KardanikXML
{

///The btDefaultMotionState provides a common implementation to synchronize world transforms with offsets.
class MotionState : public btMotionState
{
public:
    MotionState(osg::MatrixTransform *anchorNode,
                KinematicsBehavior *kinematicsBehavior,
                const btTransform &centerOfMass,
                const btTransform &centerOfMassOffset, const std::string &bodyName);

    ///synchronizes world transform from user to physics
    virtual void getWorldTransform(btTransform &centerOfMassWorldTrans) const;

    ///synchronizes world transform from physics to user
    ///Bullet only calls the update of worldtransform for active objects
    virtual void setWorldTransform(const btTransform &centerOfMassWorldTrans);

private:
    KinematicsBehavior *m_KinematicsBehavior;
    osg::MatrixTransform *m_AnchorNode;
    btTransform m_CenterOfMassOffset;
    btTransform m_CenterOfMass;
    std::string m_BodyName;
};
}

#endif /* MOTIONSTATE_H_ */
