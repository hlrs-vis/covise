/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#pragma once

#include "vvIntersection.h"
#include <OpenVRUI/coCombinedButtonInteraction.h>
#include <OpenVRUI/sginterface/vruiIntersection.h>
#include <OpenVRUI/vsg/VSGVruiNode.h>
#include <vsg/nodes/MatrixTransform.h>
#include <memory>

namespace osg
{
class Geode;
class Node;
}
namespace vrb
{
class SharedStateBase;
template<class T>
class SharedState;
}

namespace vive
{
class vvLabel;

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

struct LineSegment
{
    vsg::dvec3 start;
    vsg::dvec3 end;
};

class VVCORE_EXPORT vvIntersectionInteractor : public vrui::coAction, public vrui::coCombinedButtonInteraction
{
private:
    bool constantInteractorSize_ = true;
    float iconSize_;
    bool firstTime = true;
    bool _highliteHitNodeOnly;

    osg::Geode *findGeode(vsg::Node *n);

protected:
    vsg::ref_ptr<vsg::Node> geometryNode; ///< Geometry node
    vsg::ref_ptr<vsg::MatrixTransform> moveTransform;
    vsg::ref_ptr<vsg::MatrixTransform> scaleTransform;
    vsg::ref_ptr<vsg::MatrixTransform> interactorCaseTransform;
    vsg::MatrixTransform *parent = nullptr;
    char *_interactorName = nullptr;
    char *labelStr_ = nullptr;

    bool _hit = false;
    bool _intersectionEnabled = true;
    bool _justHit = false;
    bool _wasHit = false;
    bool _standardHL = true;
    vsg::ref_ptr<vsg::Node> _hitNode; // current node under cursor
    vsg::Node *_interactionHitNode = nullptr; // this is the node which was hit, when interaction started

    vsg::vec3 _hitPos;
    vrui::VSGVruiNode *vNode = nullptr;


    vvLabel *label_ = nullptr;
    bool m_isInitializedThroughSharedState = false;
    float _interSize; // size in mm in world coordinates
    float _scale = 1.; // scale factor for retaining screen size of interactor
    std::unique_ptr<vrb::SharedStateBase> m_sharedState;
    // the geosets are created in the derived classes
    virtual void createGeometry() = 0;

    // scale sphere to keep the size when the world scaling changes
    virtual void keepSize();
    float getScale() const;

    vsg::vec3 restrictToVisibleScene(vsg::vec3);

    const vsg::dmat4 &getPointerMat() const;

    //! reimplement in derived class for updating value of m_sharedState
    virtual void updateSharedState();

    
public:
    // size: size in world coordinates, the size of the sphere is fixed, even if the user scales the world
    // buttonId: ButtonA, ButtonB etc.
    // iconName: name of the inventor file in covise/icons
    // interactorName: name which appears in the scene graph
    // priority: interaction priority, default medium
    // highliteHitNodeOnly:  true: only the node under the cursor gets highlited - false: if any child node of the geometryNode gets hit all children are highlited
    vvIntersectionInteractor(float size, coInteraction::InteractionType buttonId, const char *iconName, const char *interactorName, enum coInteraction::InteractionPriority priority, bool highliteHitNodeOnly = false);

    // delete scene graph
    virtual ~vvIntersectionInteractor();

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

    // gives information whether this item has been initialized through a sharedState call
    bool isInitializedThroughSharedState();
    //! make state shared among partners in a collaborative session
    virtual void setShared(bool state);

    //! query whether Element state is shared among collaborative partners
    virtual bool isShared() const;

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
    vsg::vec3 getHitPos()
    {
        return _hitPos;
    }

    // called in preframe, does the interaction
    virtual void preFrame();

    //return interactor name
    char *getInteractorName()
    {
        return _interactorName;
    }

    ///< class methods for traversing children
    //static vector<vvIntersectionInteractor*> *interactors; ///< class variable for storing references of children
    //static int traverseIndex; ///< class variable for traversing children

    //static void startTraverseInteractors();
    //static void traverseInteractors();
    //static void stopTraverseInteractors();

    //static vsg::vec3 currentInterPos;
    //static bool isTraverseInteractors;

    vsg::dmat4 getMatrix()
    {
        return moveTransform->matrix;
    }

    const vsg::dmat4& getMatrix() const
    {
        return moveTransform->matrix;
    }

    void setCaseTransform(vsg::MatrixTransform *);
};
}
