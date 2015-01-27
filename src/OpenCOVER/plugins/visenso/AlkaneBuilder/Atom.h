/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// 2012 D. Rainer
//
// base class for atom of a "ball and stick model"
//
// it creates the ball and the sticks
// it checks if one stick is near to a stick of another atom
// it drawns a line between the two connections
// if the Atom is released it snaps it to the other atom
// it is possible to disconnect it from the other atom if it is picked an moved very fast

#ifndef _ATOM_H
#define _ATOM_H

#include <osg/Geode>
#include <string>
using namespace std;

class AtomStickInteractor;
class AtomBallInteractor;

#include <PluginUtil/coPlane.h>
class Atom
{

private:
    float size_;

    osg::ref_ptr<osg::Geode> lineGeode_;
    osg::ref_ptr<osg::Vec3Array> lineCoord_;
    osg::ref_ptr<osg::Vec4Array> lineColor_;

    AtomStickInteractor *otherSnapAtomStick_; //direction of other atom which is near
    AtomStickInteractor *mySnapAtomStick_; // my direction which has to be snapped

    osg::Matrix initialMat_, oldMat_, oldInvMat_, diffMat_;

    osg::Vec3 connectionLineStartPoint_, connectionLineEndPoint_;
    float angle_;
    string symbol_;
    osg::Vec4 color_;

    double lastTime_, currentTime_;
    osg::Vec3 lastPos_, currentPos_;

protected:
    void createGeometry();

public:
    std::vector<AtomStickInteractor *> atomSticks_;
    AtomBallInteractor *atomBall_;

    // position and normal in object coordinates
    // size in world coordinates (mm)
    Atom(string symbol, const char *interactorName, osg::Matrix m, float size, std::vector<osg::Vec3> connections, osg::Vec4 color);

    virtual ~Atom();
    virtual void preFrame();
    void updateTransform(osg::Matrix m);
    bool isIdle();
    bool wasStopped();
    void enableIntersection(bool);
    void show(bool);

    void snap(AtomStickInteractor *myStick, Atom *otherAtom, AtomStickInteractor *otherStick);
    void updateConnectedAtoms(osg::Matrix diffMat);
    void updateConnectedAtoms(AtomStickInteractor *alreadyUpdated, osg::Matrix diffMat);
    //const char *getInteractorName(){return _interactorName;};
    void reset();
    string getSymbol()
    {
        return symbol_;
    };
    bool checkNear(Atom *a);
    AtomStickInteractor *getMySnapAtomStick()
    {
        return mySnapAtomStick_;
    };
    AtomStickInteractor *getOtherSnapAtomStick()
    {
        return otherSnapAtomStick_;
    };
    void moveToPlane(opencover::coPlane *plane);
    bool allConnectionsConnected(AtomStickInteractor *ommit);
    bool isUnconnected();
};

#endif
