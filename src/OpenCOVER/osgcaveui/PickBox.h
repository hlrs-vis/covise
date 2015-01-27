/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_PICK_BOX_H_
#define _CUI_PICK_BOX_H_

// C++:
#include <string>

// OSG:
#include <osg/Geometry>
#include <osg/BoundingBox>
#include <osg/Switch>
#include <osg/Vec4>

// Cover:
#include <cover/coVRPluginSupport.h>

// Local:
#include "Widget.h"
#include "Events.h"

namespace vrui
{
class coTrackerButtonInteraction;
}

namespace cui
{
class Interaction;
class PickBoxListener;

/** This is an abstract class for axis oriented box-like widgets.
*/
class CUIEXPORT PickBox : public Widget, public Events
{
public:
    osg::BoundingBox _bbox;

    PickBox(Interaction *, const osg::Vec3 &, const osg::Vec3 &, const osg::Vec4 &,
            const osg::Vec4 &, const osg::Vec4 &);
    virtual ~PickBox();
    virtual void setBoxSize(const osg::Vec3 &);
    virtual void setBoxSize(osg::Vec3 &center, osg::Vec3 &size);
    virtual osg::Vec3 getBoxSize();
    virtual void setMovable(bool);
    virtual bool getMovable();
    virtual void cursorEnter(InputDevice *);
    virtual void cursorUpdate(InputDevice *);
    virtual void cursorLeave(InputDevice *);
    virtual void buttonEvent(InputDevice *, int);
    virtual void joystickEvent(InputDevice *);
    virtual void wheelEvent(InputDevice *, int);
    virtual void addListener(PickBoxListener *);
    virtual void setShowWireframe(bool);
    virtual void setSelected(bool);
    virtual bool getSelected();
    virtual bool getIntersected();
    virtual void updateWireframe();
    virtual void setScale(float);
    virtual float getScale();
    virtual void setPosition(osg::Vec3 &);
    virtual void setPosition(float, float, float);
    virtual osg::Matrix getB2W();
    virtual void move(osg::Matrix &, osg::Matrix &);

protected:
    vrui::coTrackerButtonInteraction *_interactionA; ///< interaction for first button
    vrui::coTrackerButtonInteraction *_interactionB; ///< interaction for second button
    vrui::coTrackerButtonInteraction *_interactionC; ///< interaction for third button
    static const int NUM_BOX_COLORS; ///< number of frame colors
    osg::ref_ptr<osg::Switch> _switch; ///< switches geometries for picked and non-picked
    osg::ref_ptr<osg::MatrixTransform> _scale; ///< switches geometries for picked and non-picked
    osg::Geometry *_geom[3]; ///< wireframe geometry: 0=non-selected & non-intersected, 1=intersected, 2=selected & non-intersected
    osg::Matrix _lastWand2w; ///< wand matrix from previous run
    Interaction *_interaction;
    bool _isMovable; ///< true=moves with mouse while button pressed
    std::list<PickBoxListener *> _listeners;
    bool _isIntersected; ///< true if cursor is in box
    bool _showWireframe; ///< true: wireframe display selected by user
    bool _isSelected; ///< true if box is the one selected for interactions

    virtual void createGeometry(const osg::Vec4 &, const osg::Vec4 &, const osg::Vec4 &);
    virtual osg::Geometry *createWireframe(const osg::Vec4 &);
    virtual void updateVertices(osg::Geometry *);
};

class CUIEXPORT PickBoxListener
{
public:
    virtual ~PickBoxListener()
    {
    }
    virtual void pickBoxButtonEvent(PickBox *, InputDevice *, int) = 0;
    virtual void pickBoxMoveEvent(PickBox *, InputDevice *) = 0;
};
}

#endif
