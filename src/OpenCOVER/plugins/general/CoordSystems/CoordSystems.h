/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CoordSystems_H
#define _CoordSystems_H
/****************************************************************************\
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Template Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner                                               **
 **                                                                          **
 ** History:               **
 ** Nov-01  v1                               **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <cover/coVRSelectionManager.h>
#include <util/coExport.h>
#include <cover/coVRLabel.h>

namespace vrui
{
class coRowMenu;
}

class CoordAxis;
const float ScFactors[] = { 0.001f, 0.01f, 0.1f, 1.0f, 10.0f, 100.0f, 1000.0f };
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
using namespace vrui;
using namespace opencover;

class CoordSystems : public coVRPlugin, public coMenuListener, public coSelectionListener
{
public:
    static CoordSystems *plugin;
    static coRowMenu *coords_menu;

    CoordSystems();
    ~CoordSystems();

    // this will be called in PreFrame
    void preFrame();
    void menuEvent(coMenuItem *item);
    //this will be called by changing the selected object
    virtual bool selectionChanged();
    virtual bool pickedObjChanged();
    //this refreshes the builded coordsystems
    bool refresh_coords();
    bool init();

private:
    coMenu *cover_menu;
    coSubMenuItem *button;
    coRowMenu *coord_menu;
    coRowMenu *Scale_menu;
    coSubMenuItem *Scale;
    coSubMenuItem *coords;
    coCheckboxGroup *scale_factor_Group;
    coMenuItem *menu_showCoord;
    coMenuItem *menue_showSelected;
    coCheckboxMenuItem *scale_factor[8];
    CoordAxis *globalCoordSys;

    std::list<CoordAxis *> CoordList;
};
//------------------------------------------------------------------------------------------------------------------------------
class CoordAxis
{
public:
    CoordAxis();
    ~CoordAxis();
    //adding the Group-Node to the Scenegraph:
    void AddToScenegraph(osg::Group *);
    //removing the Group-Node to the Scenegraph:
    void RemoveFromScenegraph();
    //calculates the global Matrix of the CoordAxis Object and sets the new transformation matrix witout scaling
    void setScaling(float);
    //setting the label of the Coordsystem
    void setLabelValue(float);
    void makeIdentity();
    void printMatrix();

private:
    osg::ref_ptr<osg::Group> parent;
    static osg::ref_ptr<osg::MatrixTransform> axisNode;
    osg::ref_ptr<osg::MatrixTransform> HeadNode;
    coVRLabel *ScLabel;
};

#endif
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
