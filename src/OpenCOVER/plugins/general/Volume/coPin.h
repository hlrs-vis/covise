/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef CO_PIN_H
#define CO_PIN_H
#include "util/coTypes.h"

#include <osg/Array>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/StateSet>

#include <virvo/vvtfwidget.h>

namespace vrui
{
class vruiTransformNode;
class OSGVruiTransformNode;
}

class coPin
{
    friend class coPinEditor;

public:
    coPin(osg::Group *root, float Height, float Width, vvTFWidget *myPin, bool bottom=true);
    virtual ~coPin();

    virtual void setPos(float x, float minv = 0.0f, float maxv = 1.0f); // only update Position
    float getPosValue() const; // position in [minv..maxv]
    float getPos01() const; // position in [0..1]
    virtual void select();
    virtual void deSelect();
    int getID();

    virtual vrui::vruiTransformNode *getDCS();
    vvTFWidget *jPin;
    void setHandleTrans(float t);
    float handleTrans()
    {
        return _handleTrans;
    }

protected:
    static int numAlphaPins;
    static int numPins;
    int id;
    bool selected;
	bool onBottom;
    vrui::OSGVruiTransformNode *myDCS;
    osg::ref_ptr<osg::MatrixTransform> selectionDCS;
    osg::Vec3 oldScale;
    float myX;
    float A;
    float B;
    float H;
    float W;
    virtual osg::Geode *createLineGeode();
    virtual void createLists();

    osg::ref_ptr<osg::Vec4Array> color;
    osg::ref_ptr<osg::Vec3Array> coord;
    osg::ref_ptr<osg::Vec3Array> normal;
    osg::ref_ptr<osg::StateSet> normalGeostate;
    osg::ref_ptr<osg::Geode> geode;
    float _handleTrans;
};
#endif
