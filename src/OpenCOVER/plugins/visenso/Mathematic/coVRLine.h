/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2009 Visenso  **
 **                                                                        **
 ** Description: coVRLine                                                  **
 **              Draws a line according to mode                            **
 **               either as a line through two points                      **
 **               or a line with a base point and a direction              **
 **               with a nameTag and equation                              **
 **               only within the bounding box (needs to be set)           **
 **               [not normal lines do not have interactors]               **
 **                                                                        **
 ** Author: M. Theilacker                                                  **
 **                                                                        **
 ** History:                                                               **
 **     4.2009 initial version                                             **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef _COVRLINE_H
#define _COVRLINE_H

#include <string>

#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/BoundingBox>
#include <osg/MatrixTransform>

#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <PluginUtil/coPlane.h>

#include "coVRPoint.h"
#include "coVRDirection.h"

class coVRLine
{
public:
    // constructor destructor
    coVRLine(osg::Vec3 vec1, osg::Vec3 vec2, int mode, string name = "g", bool normal = true, double radius = 0.09);
    ~coVRLine();

    enum
    {
        POINT_POINT = 0,
        POINT_DIR = 1,
        PARALLEL = 10,
        INTERSECT = 11,
        SKEW = 12
    };

    // methods of class
    static void setBoundingBox(osg::BoundingBox *boundingBox);
    static void deleteBoundingBox();

    // methods
    /// adds the mode checkbox, the 2 points, the name and
    ///   parametric equation to the parent menu
    int addToMenu(coRowMenu *parentMenu_, int position);
    /// calculates the distance between this line and otherLine
    /// and returns the perpendicular points on the lines
    double distance(coVRLine *otherLine, osg::Vec3 *perpendicularL1, osg::Vec3 *perpendicularL2);
    /// calculates the distance between this line and point
    /// and returns the perpendicular point on the line
    double distance(coVRPoint *point, osg::Vec3 *perpendicular);
    /// intersects line with otherLine, saving intersection point to isectPoint
    /// and intersection angle to angle
    /// testing for skew and parallel first
    bool intersect(coVRLine *otherLine, osg::Vec3 *isectPoint, double *angle);
    /// for check from extern (will reset the variable isChanged_ to false)
    bool isChanged();
    /// is line parallel to otherLine
    bool isParallel(coVRLine *otherLine);
    bool isVisible();
    /// is line skew to otherLine
    bool isSkew(coVRLine *otherLine);
    osg::Vec3 getBasePoint();
    osg::Vec3 getDirection();
    string getName();
    /// takes care of menu events from parent class
    void menuEvent(coMenuItem *menuItem);
    /// preparation for each frame
    void preFrame();
    void removeFromMenu();
    /// color for the line
    void setColor(osg::Vec4 color);
    osg::Vec4 color()
    {
        return color_;
    };
    void setMode(int mode);
    void setPoints(osg::Vec3 point1, osg::Vec3 point2);
    /// sets the drawables (in)visible
    void setVisible(bool visible);
    /// tests line for intersection, parallel or skew with otherLine
    /// returning state from enum, computing intersection point
    /// and intersection angle (in degrees)
    int test(coVRLine *otherLine, osg::Vec3 *isectPoint, double *angle);
    /// updates menu, the second point, the direction, drawables and name tag
    void update();
    /// hides/shows the name label
    void hideLabel(bool hide);

private:
    // variables of class
    static int _lineID_;
    static osg::BoundingBox *_boundingBox_;
    static vector<coPlane *> _boundingPlanes_;

    // variables
    string name_;
    osg::ref_ptr<osg::MatrixTransform> node_;
    int mode_;
    coVRPoint *point1_;
    coVRPoint *point2_;
    coVRDirection *direction_;
    coRowMenu *parentMenu_;
    coCheckboxMenuItem *modeCheckbox_;
    coLabelMenuItem *lineLabel_;
    coLabelMenuItem *sepLabel_;
    osg::Vec3 drawMin_;
    osg::Vec3 drawMax_;
    osg::Cylinder *line_;
    osg::Geode *lineGeode_;
    osg::ShapeDrawable *lineDraw_;
    double lineRadius_;
    osg::StateSet *stateSet_;
    osg::Material *material_;
    osg::Vec4 color_;
    bool isVisible_;
    bool isChanged_;
    coVRLabel *nameTag_;
    bool normal_;
    bool inBoundingBox_;
    bool isBBSet_;
    bool labelsShown_;
    int menuLanguage_;

    // methods
    /// string with name and parametric equation
    string computeNameTagText();
    /// find intersection points of line with bounding box
    int findEndPoints();
    void hide();
    /// intersects line with otherLine, computing intersection point
    /// and intersection angle (in degrees)
    /// no testing for skew and parallel
    bool isIntersection(coVRLine *otherLine, osg::Vec3 *isectPoint, double *angle);
    /// materials for color
    void makeColor();
    /// pinboard tag with the name and parametric equation
    void makeNameTag();
    void show();
    /// computes the drawables for the line
    int updateLine();
    /// updates position for pinboard name tag and equationtext
    void updateNameTag();
};

#endif
