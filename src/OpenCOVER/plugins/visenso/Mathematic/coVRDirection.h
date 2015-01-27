/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2009 Visenso  **
 **                                                                        **
 ** Description: coVRDirection                                             **
 **              Draws a direction vector with rotInteractor and a nameTag **
 **               only within the bounding box (needs to be set)           **
 **                                                                        **
 ** Author: M. Theilacker                                                  **
 **                                                                        **
 ** History:                                                               **
 **     4.2009 initial version                                             **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef _COVRDIR_H
#define _COVRDIR_H

#include <osg/Vec3>
#include <osg/MatrixTransform>

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coCheckboxMenuItem.h>

#include <PluginUtil/coVR3DRotInteractor.h>
#include <cover/coVRLabel.h>

class coVRDirection
{
public:
    // constructor destructor
    coVRDirection(osg::Vec3 direction, osg::Vec3 position, string name = "V", double radius = 0.1);
    ~coVRDirection();

    // methods of class
    static void setBoundingBox(osg::BoundingBox *boundingBox);

    // methods
    /// adds the visible checkbox to the parent menu
    int addToMenu(coRowMenu *parentMenu_, int position);
    osg::Vec3 getDirection();
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
    void setPosition(osg::Vec3 position);
    void setDirection(osg::Vec3 direction);
    /// updates direction vector, interactor and name tag
    void update();
    void hideLabel(bool hide);

    double x();
    double y();
    double z();

private:
    // variables of class
    static int _directionID_;
    static osg::BoundingBox *_boundingBox_;

    // variables
    string name_;
    osg::ref_ptr<osg::MatrixTransform> node_;
    osg::Vec3 direction_;
    osg::Vec3 oldDirection_;
    osg::Vec3 position_;
    coVR3DRotInteractor *directionInteractor_;
    osg::Cylinder *dirLine_;
    osg::Geode *dirLineGeode_;
    osg::ShapeDrawable *dirLineDraw_;
    double dirLineRadius_;
    coVRLabel *nameTag_;
    coRowMenu *parentMenu_;
    coCheckboxMenuItem *visibleCheckbox_;
    bool isChanged_;
    bool isRunning_;
    bool isVisible_;
    bool isBBSet_;
    bool labelsShown_;

    // methods
    /// string with name and position
    string computeNameTagText();
    /// position for interactor
    void makeDirectionInteractor();
    /// pinboard tag with the name
    void makeNameTag();
    /// position for direction vector
    void updateDirectionVector();
    /// updates position for pinboard name tag and direction text
    void updateNameTag();
};

#endif
