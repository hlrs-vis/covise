/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// 2012 VISENSO GmbH D. Rainer
//
// class for the stick of the "ball and stick model" (deutsch Kugel-Staebchen-Modell)
//
// cylinder geometry
// if picked with the ray it rotates around the origin
// it knows to which atom ball it belongs
// and if it is connected to another stick
//
//                         scene
//                          |
//                       xformDCS
//                          |
//                       scaleDCS
//                       |      |
//       coviseObjectsRoot   moveTransform
//       |                      |
//      ...                scaleTransform = parent_ solange kein case transform
//                              |
//                     geometryNode=group_
//                              |
//                          transform_
//                              |
//                          cylinderGeode

#ifndef ATOM_STICK_INTERACTOR_H
#define ATOM_STICK_INTERACTOR_H

#include <osg/Vec3>
#include <osg/MatrixTransform>

#include <PluginUtil/coVR3DRotCenterInteractor.h>

class Atom;
class AtomStickInteractor : public opencover::coVR3DRotCenterInteractor
{
private:
    osg::Vec3 dir_; // direcion of connection when atom ist not transformed
    osg::MatrixTransform *transform_; // transform z axis to direction
    osg::Matrix initialOrientation_;
    osg::Vec3 initialPos_;
    std::string symbol_;
    osg::Vec4 color_;
    AtomStickInteractor *connectedStick_;
    Atom *myAtom_;
    osg::Matrix diffMat_, oldInvMat_;

public:
    AtomStickInteractor(std::string symbol, const char *interactorName, Atom *myAtom, osg::Matrix initialOrientation, osg::Vec3 initialPos, osg::Vec3 stickDir, float size, osg::Vec4 color);
    virtual ~AtomStickInteractor();
    virtual void preFrame();
    osg::Vec3 getDir();
    void resetPosition();
    void setConnectedStick(AtomStickInteractor *otherStick);
    AtomStickInteractor *getConnectedStick();
    Atom *getAtom();
    osg::Matrix getDiffMat();
    void updateTransform(osg::Matrix m, osg::Vec3 p);
};
#endif
