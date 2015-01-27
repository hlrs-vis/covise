/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_MEASURE_H
#define _CUI_MEASURE_H

// CUI:
#include "Widget.h"

#include <osg/BoundingBox>

#include "PickBox.h"

namespace cui
{
class Bar;
class Arrow;
class Interaction;
class MeasureListener;

/** 
      This class provides a tape measure or ruler like tool. It consists
      of a rubberband bar and an arrow at each end. By grabbing the arrows, the
      end points of the bar can be changed.
  */
class CUIEXPORT Measure : public cui::Widget
{
protected:
    osg::Vec3 _volRight;
    osg::Vec3 _volLeft;
    cui::Bar *_bar;
    cui::Arrow *_rightEnd;
    cui::Arrow *_leftEnd;
    osg::Matrix _leftTransBBox;
    osg::Matrix _rightTransBBox;
    PickBox *_pickBox;
    osg::BoundingBox _bbox;
    float _pos;
    std::list<MeasureListener *> _listeners;
    std::list<MeasureListener *>::iterator _iter;

public:
    Measure(cui::Interaction *, PickBox *);
    virtual ~Measure();
    void setRotate();
    void addMeasureListener(MeasureListener *);
    osg::Vec3 getRightEnd();
    osg::Vec3 getLeftEnd();
    void setBBox(osg::BoundingBox);
    osg::BoundingBox getBBox();
    PickBox *getPickBox();
};

class CUIEXPORT MeasureListener
{
public:
    virtual ~MeasureListener()
    {
    }
    virtual void measureUpdate() = 0;
};
}

#endif
