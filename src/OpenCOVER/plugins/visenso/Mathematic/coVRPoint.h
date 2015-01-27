/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2009 Visenso  **
 **                                                                        **
 ** Description: coVRPoint                                                 **
 **              Draws a point with a sphere interactor, a nameTag,        **
 **               an base vector and an axis projection                    **
 **               only within the bounding box (needs to be set)           **
 **               [not normal points do not have a projection, nor         **
 **                an Interactor, nor a base vector and are orange]        **
 **                                                                        **
 ** Author: M. Theilacker                                                  **
 **                                                                        **
 ** History:                                                               **
 **     4.2009 initial version                                             **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef _COVRPOINT_H
#define _COVRPOINT_H

#include <string>

#include <osg/Vec3>
#include <osg/MatrixTransform>

#include <PluginUtil/coVR3DTransInteractor.h>
#include <cover/coInteractor.h>
#include <cover/coVRLabel.h>
#include <PluginUtil/coArrow.h>

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coLabelMenuItem.h>

#include "coVRAxisProjection.h"

using namespace vrui;
using namespace opencover;

class coVRPoint
{
public:
    // constructor destructor
    coVRPoint(osg::Vec3 position, string name = "P", bool normal = true);
    ~coVRPoint();

    // methods of class
    static void setBoundingBox(osg::BoundingBox *boundingBox);

    // methods
    /// adds the visible checkbox to the parent menu
    int addToMenu(coRowMenu *parentMenu_, int position, bool sep = true);
    /// enables or disables the basePointInteractor
    void enableInteractor(bool enable);
    /// returns the position of the last point relevant menu item; 0 for error
    int getMenuPosition();
    string getName();
    osg::Vec3 getPosition();
    bool isChanged();
    bool isVisible();
    /// takes care of menu events from parent class
    void menuEvent(coMenuItem *menuItem);
    /// preparation for each frame
    void preFrame();
    void removeFromMenu();
    /// sets the drawables (in)visible
    void setVisible(bool visible);
    /// sets the position
    void setPosition(osg::Vec3 position);
    /// shows or hides the projection
    void showProjection(bool show);
    /// shows or hides the base vector
    void showVector(bool show);
    /// updates projection, interactor, base vector and name tag
    void update();
    /// hides/shows the name label
    void hideLabel(bool hide);

    double x();
    double y();
    double z();

private:
    // variables of class
    static int _pointID_;
    static osg::BoundingBox *_boundingBox_;

    // variables
    string name_;
    osg::ref_ptr<osg::MatrixTransform> node_;
    //       osg::StateSet                      *stateSet_;
    //       osg::Material                      *material_;
    osg::Vec4 color_;
    osg::ref_ptr<osg::MatrixTransform> baseVectorNode_;
    osg::Vec3 position_;
    osg::Vec3 oldPosition_;
    coVR3DTransInteractor *basePointInteractor_;
    coVRLabel *nameTag_;
    string labelText_;
    osg::ref_ptr<coArrow> baseVector_;
    coVRAxisProjection *projection_;
    coRowMenu *parentMenu_;
    coCheckboxMenuItem *visibleCheckbox_;
    coLabelMenuItem *sepLabel_;
    bool showVec_;
    bool showProj_;
    bool isVisible_;
    bool isChanged_;
    bool isRunning_;
    bool enableInteractor_;
    bool normal_;
    bool inBoundingBox_;
    bool isBBSet_;
    bool labelsShown_;

    // methods
    /// string with name and position
    string computeNameTagText();
    void hide();
    /// materials for color
    //void makeColor();
    /// makes the base vector of the point
    void makeBaseVector();
    /// makes the interactor for the point
    void makeInteractor();
    /// pinboard tag with the name
    void makeNameTag();
    /// color for the line
    //void setColor( osg::Vec4 color );
    void show();
    /// position for base vector
    void updateBaseVector();
    /// updates position for pinboard name tag and point text
    void updateNameTag();
};

#endif
