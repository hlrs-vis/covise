/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _BULLETPROBE_H
#define _BULLETPROBE_H

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

class BulletProbe;
class Mark;

// ----------------------------------------------------------------------------
//! label for distance between markers
// ----------------------------------------------------------------------------
class Dimension
{
public:
    Dimension(int idParam, BulletProbe *m);
    virtual ~Dimension();
    
    virtual bool isplaced();
    virtual void update();
    int getID();

    Mark *marks[2]; //!< two cones: first and second placed

protected:

    //void makeText();

    static osg::ref_ptr<osg::Material> globalWhitemtl;
    static osg::ref_ptr<osg::Material> globalRedmtl;
    
    BulletProbe *plugin;
    int id;
    int placedMarks;
    bool placing;
    double oldDist;
    osg::MatrixTransform *myDCS;
    osg::Switch *geos;
    // osgText::Text *labelText; ///< label text string in Performer format
    // char labelString[100]; ///< text string which is displayed on the label
};

// ----------------------------------------------------------------------------
//! line between markers 
// ----------------------------------------------------------------------------
class LinearDimension : public Dimension
{
public:
    
    LinearDimension(int idParam, BulletProbe *m);
    virtual ~LinearDimension();
    
    virtual void update();
    
protected:
    
    osg::ref_ptr<osg::MatrixTransform> line;
};







// ----------------------------------------------------------------------------
//! start and end point of line; represented as cones with spheres on top
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
//! main plugin class
// ----------------------------------------------------------------------------
class BulletProbe
    : public coVRPlugin,
      public coMenuListener
{
public:

    static BulletProbe *plugin;

    BulletProbe();
    virtual ~BulletProbe();

    bool init();
    void preFrame();
    
    void message(int toWhom, int type, int len, const void *buf);
    
    void setCurrentMeasure(Mark *m);
    
private:
    
    Mark *currentProbe;
    
    covise::DLinkList<Dimension *> dims; 
    
    osg::Matrix invStartHand;
    covise::DLinkList<Mark *> marker;
    osg::Group *objectsRoot; //!< cover root node for covise objects
    bool moving;
    int maxDimID;

    // menu

    coButtonMenuItem *clearItem;
    coButtonMenuItem *bmiHideAll;
    coButtonMenuItem *bmiLoadFromFile;
    coButtonMenuItem *bmiSaveToFile;
    coButtonMenuItem* bmiItem1;
    coButtonMenuItem* bmiItem2;
    coPotiMenuItem *markerScalePoti;
    coPotiMenuItem *fontScalePoti;
    coPotiMenuItem *lineWidthPoti;
    coSubMenuItem *measureMenuItem;
    coSubMenuItem *unitsMenuItem;
    coRowMenu *measureMenu;
    coRowMenu *unitsMenu;
    
    coTrackerButtonInteraction *interactionA; //!< interaction for first button

    void menuEvent(coMenuItem *);
    void createMenuEntry();
    void removeMenuEntry();
};

// ----------------------------------------------------------------------------

#endif
