/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MEASURE_H
#define _MEASURE_H

#include <osg/Node>
#include <osg/MatrixTransform>
#include <osg/Material>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osgText/Text>
#include <cover/coVRPluginSupport.h>
#include <util/DLinkList.h>
#include <OpenVRUI/coAction.h>
#include <OpenVRUI/osg/OSGVruiNode.h>
#include <OpenVRUI/coValuePoti.h>
#include <OpenVRUI/coPotiMenuItem.h>

namespace vrui
{
class coButtonMenuItem;
class coSubMenuItem;
class coRowMenu;
class coCheckboxMenuItem;
class coPotiItem;
class coTrackerButtonInteraction;
}

#define NEW_DIMENSION 'n'
#define MOVE_MARK 'm'
#define REMOVE 'r'

using namespace vrui;
using namespace opencover;

class Measure;
class Mark;

/** Label for distance between markers. */
class Dimension
{
protected:
    Measure *plugin;
    static osg::ref_ptr<osg::Material> globalWhitemtl;
    static osg::ref_ptr<osg::Material> globalRedmtl;
    int id;
    int placedMarks;
    bool placing;
    double oldDist;
    osg::MatrixTransform *myDCS;
    osg::Switch *geos;
    osgText::Text *labelText; ///< label text string in Performer format
    char labelString[100]; ///< text string which is displayed on the label

    void MakeText();

public:
    Mark *marks[2]; // two cones: first and second placed

    Dimension(int id, Measure *);
    virtual ~Dimension();
    virtual bool isplaced();
    virtual void update();
    int getID()
    {
        return id;
    };
    bool isSelected();
};

/** Line between markers */
class LinearDimension : public Dimension
{
protected:
    osg::ref_ptr<osg::MatrixTransform> line;

public:
    LinearDimension(int id, Measure *);
    virtual ~LinearDimension();
    virtual void update();
};

/** Start and end point of line; represented as cones with spheres on top */
class Mark : public coAction
{
private:
    int id;
    bool moveStarted;
    osg::MatrixTransform *pos;
    osg::MatrixTransform *sc;
    osg::Node *geo;
    osg::Switch *icons;
    Dimension *dim;
    osg::Matrix startPos;
    osg::Matrix invStartHand;
    OSGVruiNode *vNode;
    coTrackerButtonInteraction *interactionA; ///< interaction for first button

public:
    bool placing;
    bool moveMarker;

    Mark(int id, Dimension *dim);
    virtual ~Mark();
    virtual int hit(vruiHit *hit);
    virtual void miss();
    void update();
    int getID()
    {
        return id;
    };
    void setPos(osg::Matrix &mat);
    void resize();
    float getDist(osg::Vec3 &a);
    void getMat(osg::Matrix &);
    void setIcon(int i);
};

/** Main plugin class */
class Measure : public coVRPlugin, public coMenuListener
{
private:
    Mark *currentMeasure;
    osg::Matrix invStartHand;
    covise::DLinkList<Mark *> marker;
    osg::Group *objectsRoot; // COVER root node for covise objects
    bool moving;
    //int snapToEdges;
    int maxDimID;

    coButtonMenuItem *areaItem;
    coButtonMenuItem *clearItem;
    //coCheckboxMenuItem *snapItem;
    coPotiMenuItem *markerScalePoti, *fontScalePoti, *lineWidthPoti;
    coSubMenuItem *measureMenuItem, *unitsMenuItem;
    coRowMenu *measureMenu, *unitsMenu;
    coTrackerButtonInteraction *interactionA; ///< interaction for first button

    covise::DLinkList<Dimension *> dims;

    void menuEvent(coMenuItem *);
    void createMenuEntry(); // create a menu items
    void removeMenuEntry(); // remove the menu items

public:
    int menuSelected; // TRUE if menu itme "Cube" was selected
    static Measure *plugin;

    Measure();
    virtual ~Measure();
    bool init();
    void preFrame();
    void message(int toWhom, int type, int len, const void *buf);
    void setCurrentMeasure(Mark *m);
};

#endif
