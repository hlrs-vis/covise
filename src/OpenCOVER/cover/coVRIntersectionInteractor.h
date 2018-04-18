/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_VR_INTERSECTION_INTERACTOR_H
#define _CO_VR_INTERSECTION_INTERACTOR_H

#include "coIntersection.h"
#include <OpenVRUI/coCombinedButtonInteraction.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/osg/OSGVruiNode.h>
#include <osg/MatrixTransform>

namespace osg
{
class Geode;
class Node;
}

namespace opencover
{
class coVRLabel;

// selectable sphere which can be positioned
// the sphere is modeled in the origin
// transDCS contains the matrix in object coordinates
//
//
//                         scene
//                          |
//                      xformDCS
//                          |
//                      scaleDCS
//                       |      |
//       coviseObjectsRoot   moveTransform
//       |                      |
//      dcs                 scaleTransform
//       |                      |
//      TracerSeq           geometryNode
//      |       |
//   TrGeode TrGeode
//
//

class COVEREXPORT coVRIntersectionInteractor : public vrui::coAction, public vrui::coCombinedButtonInteraction
{
private:
    osg::StateSet *loadDefaultGeostate();
    bool constantInteractorSize_;
    float iconSize_;
    bool firstTime;

    osg::Geode *findGeode(osg::Node *n);

protected:
    osg::ref_ptr<osg::Node> geometryNode; ///< Geometry node
    osg::ref_ptr<osg::MatrixTransform> moveTransform;
    osg::ref_ptr<osg::MatrixTransform> scaleTransform;
    osg::ref_ptr<osg::MatrixTransform> interactorCaseTransform;
    osg::MatrixTransform *parent;
    char *_interactorName;
    char *labelStr_;

    bool _hit;
    bool _intersectionEnabled;
    bool _justHit;
    bool _wasHit;
    bool _standardHL;
    osg::ref_ptr<osg::Node> _hitNode;
    osg::Vec3 _hitPos;
    vrui::OSGVruiNode *vNode;

    osg::ref_ptr<osg::StateSet> _selectedHl, _intersectedHl, _oldHl;

    coVRLabel *label_;

    float _interSize; // size in mm in world coordinates
    float _scale = 1.; // scale factor for retaining screen size of interactor

    // the geosets are created in the derived classes
    virtual void createGeometry() = 0;

    // scale sphere to keep the size when the world scaling changes
    virtual void keepSize();
    float getScale() const;

    osg::Vec3 restrictToVisibleScene(osg::Vec3);

    const osg::Matrix &getPointerMat() const;

public:
    // size: size in world coordinates, the size of the sphere is fixed, even if the user scales the world
    // buttonId: ButtonA, ButtonB etc.
    // iconName: name of the inventor file in covise/icons
    // interactorName: name which appears in the scene graph
    // priority: interaction priority, default medium
    coVRIntersectionInteractor(float size, coInteraction::InteractionType buttonId, const char *iconName, const char *interactorName, enum coInteraction::InteractionPriority priority);

    // delete scene graph
    virtual ~coVRIntersectionInteractor();

    // make the interactor intersection sensitive
    void enableIntersection();

    // check whether interactor is enabled
    bool isEnabled();

    // make the interactor intersection insensitive
    void disableIntersection();

    // called every time when the geometry is intersected
    virtual int hit(vrui::vruiHit *hit);

    // called once when the geometry is not intersected any more
    virtual void miss();

    // start the interaction (set selected hl, store dcsmat)
    virtual void startInteraction();

    // move the interactor relatively to it's old position
    // according to the hand movements
    virtual void doInteraction();

    // stop the interaction
    virtual void stopInteraction();

    // make the interactor visible
    void show();

    // make the interactor invisible
    void hide();

    virtual void addIcon(); // highlight and add

    virtual void removeIcon(); // remove

    virtual void resetState(); // un-highlight

    // return the intersected state
    int isIntersected()
    {
        return _hit;
    }

    // return true if just intesected
    bool wasHit()
    {
        return _wasHit;
    }

    // return hit positon
    osg::Vec3 getHitPos()
    {
        return _hitPos;
    }

    osg::Node *getHitNode();

    // called in preframe, does the interaction
    virtual void preFrame();

    //return interactor name
    char *getInteractorName()
    {
        return _interactorName;
    }

    ///< class methods for traversing children
    //static vector<coVRIntersectionInteractor*> *interactors; ///< class variable for storing references of children
    //static int traverseIndex; ///< class variable for traversing children

    //static void startTraverseInteractors();
    //static void traverseInteractors();
    //static void stopTraverseInteractors();

    //static osg::Vec3 currentInterPos;
    //static bool isTraverseInteractors;

    osg::Matrix getMatrix()
    {
        return moveTransform->getMatrix();
    }

    void setCaseTransform(osg::MatrixTransform *);
};
}
#endif
