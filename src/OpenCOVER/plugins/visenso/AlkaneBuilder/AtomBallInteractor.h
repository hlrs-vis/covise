/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// 2012 VISENSO GmbH D. Rainer
//
// class for the ball of the "ball and stick model" (deutsch Kugel-Staebchen-Modell)
//
// sphere geometry
// if picked with the ray it translates on a plane
// it knows the sticks which are connected
//
//                         scene
//                          |
//                      xformDCS
//                          |
//                       scaleDCS
//                       |      |
//       coviseObjectsRoot   moveTransform
//       |                      |
//      ...                scaleTransform = parent_ solange kein case transform
//                              |
//                     geometryNode=group_
//                              |
//                         sphereGeode
//
//

#ifndef _ATOM_BALL_INTERACTOR_H
#define _ATOM_BALL_INTERACTOR_H

#include <PluginUtil/coVR2DTransInteractor.h>
#include <osg/Vec3>

class AtomBallInteractor : public opencover::coVR2DTransInteractor
{

private:
    osg::Group *group_;
    osg::Vec3 initialPos_, lastPos_;
    std::string symbol_;
    osg::Vec4 color_;
    osg::Matrix diffMat_, oldInvMat_;
    osg::BoundingBox box_;

protected:
    void createGeometry();

public:
    // position and normal in object coordinates
    // size in world coordinates (mm)
    AtomBallInteractor(std::string symbol, const char *interactorName, osg::Vec3 pos, osg::Vec3 normal, float size, osg::Vec4 color);

    virtual ~AtomBallInteractor();
    virtual void preFrame();
    virtual void updateTransform(osg::Matrix mat);
    virtual void updateTransform(osg::Vec3 pos, osg::Vec3 normal);
    virtual void startInteraction();
    virtual void doInteraction();
    virtual void stopInteraction();
    void resetPosition();
    void setBoundingBox(osg::BoundingBox box)
    {
        box_ = box;
    };
    osg::Vec3 restrictToBox(osg::Vec3);
    std::string getSymbol()
    {
        return symbol_;
    };
    osg::Matrix getDiffMat();
    osg::Vec3 getLastPosition()
    {
        return lastPos_;
    };
};

#endif
