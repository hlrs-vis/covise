/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LISTENER_COVER_
#define _LISTENER_COVER_

#include <osg/Matrix>
#include <vrml97/vrml/Listener.h>

namespace vrml
{
class Player;
}

namespace opencover
{
class coVRPluginSupport;
}

using namespace vrml;
using namespace opencover;

class ViewerOsg;

class VRML97PLUGINEXPORT ListenerCover : public Listener
{
public:
    ListenerCover(coVRPluginSupport *cover);
    virtual ~ListenerCover();

    Player *createPlayer();
    void destroyPlayer(Player *player);

    virtual osg::Matrix getVrmlBaseMat() const;
    virtual osg::Matrix getCurrentTransform() const;

    virtual vec WCtoOC(vec pos) const;
    virtual vec WCtoVC(vec pos) const;
    virtual vec VCtoOC(vec pos) const;
    virtual vec VCtoWC(vec pos) const;
    virtual vec OCtoWC(vec pos) const;
    virtual vec OCtoVC(vec pos) const;

    virtual vec getPositionWC() const;
    virtual vec getPositionVC() const;
    virtual vec getPositionOC() const;

    virtual vec getVelocity() const;
    virtual void getOrientation(vec *at, vec *up) const;

    virtual double getTime() const;

private:
    osg::Matrix *matIdent;
    coVRPluginSupport *cover;
    mutable double lastTime;
    mutable vec lastPos;
    mutable vec velocity;
};
#endif
