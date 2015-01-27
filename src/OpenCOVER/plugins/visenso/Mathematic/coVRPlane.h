/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2009 Visenso  **
 **                                                                        **
 ** Description: coVRPlane                                                 **
 **              Draws a plane according to mode                           **
 **               either as a plane through three points                   **
 **               or a plane with a base point and a (normal)direction     **
 **               only within the bounding box (needs to be set)           **
 **               with a nameTag and equations                             **
 **                                                                        **
 ** Author: M. Theilacker                                                  **
 **                                                                        **
 ** History:                                                               **
 **     4.2009 initial version                                             **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef _COVRPLANE_H
#define _COVRPLANE_H

#include <osg/Vec3>
#include <osg/Geometry>
#include <osg/BoundingBox>
#include <osg/Array>

#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <PluginUtil/coPlane.h>

#include "coVRPoint.h"
#include "coVRDirection.h"

class coVRPlane : public coPlane
{
public:
    // constructors destructor
    coVRPlane(osg::Vec3 point, osg::Vec3 normal, string name = "E");
    coVRPlane(osg::Vec3 point1, osg::Vec3 point2, osg::Vec3 point3, string name = "E");

    ~coVRPlane();

    enum
    {
        POINT_POINT_POINT = 0,
        POINT_DIR = 1,
        POINT_DIR_DIR = 2,
        PARALLEL = 10,
        INTERSECT = 11,
        LIESIN = 20,
        PARAM_EQU = 31,
        COORD_EQU = 32,
        NORM_EQU = 33,
        P1 = 40,
        P2 = 41,
        P3 = 42,
        DIR1 = 43,
        DIR2 = 44,
        NORM = 45
    };

    // methods of class
    static void setBoundingBox(osg::BoundingBox *boundingBox);

    // methods
    /// adds the mode checkbox, the 2 points, the name and
    /// parametric equation to the parent menu
    int addToMenu(coRowMenu *parentMenu_, int position);
    /// distance between plane and point
    double distance(coVRPoint *point, osg::Vec3 *perpendicular);
    /// distance between plane and line
    double distance(coVRLine *line, osg::Vec3 *perpendicularP, osg::Vec3 *perpendicularL);
    /// distance between plane and otherPlane
    double distance(coVRPlane *otherPlane, osg::Vec3 *perpendicularP1, osg::Vec3 *perpendicularP2);
    osg::Vec3 getBasePoint();
    string getName();
    osg::Vec3 getNormalDirection();
    /// intersects plane with line, saving intersection point to isectPoint
    /// and intersection angle to angle
    bool intersect(coVRLine *line, osg::Vec3 *isectPoint, double *angle);
    /// intersects plane with otherPlane, computing intersection line (2 Points)
    /// and saving intersection angle to angle
    bool intersect(coVRPlane *otherPlane, osg::Vec3 *isectLinePoint1, osg::Vec3 *isectLinePoint2, double *angle);
    /// for check from extern (will reset the variable isChanged_ to false)
    bool isChanged();
    /// is plane parallel to line
    bool isParallel(coVRLine *line);
    /// is plane parallel to otherPlane
    bool isParallel(coVRPlane *otherPlane);
    bool isVisible();
    /// takes care of menu events from parent class
    void menuEvent(coMenuItem *menuItem);
    /// preparation for each frame
    void preFrame();
    void removeFromMenu();
    /// color for the plane
    void setColor(osg::Vec4 color);
    osg::Vec4 color()
    {
        return color_;
    };
    void setMode(int mode);
    /// sets the drawables (in)visible
    void setVisible(bool visible);
    /// defines which equation is shown in nameTag
    void showEquations(bool showParamEqu, bool showCoordEqu, bool showNormEqu);
    /// tests if plane or line intersect or are parallel
    /// returning state from enum, computing intersection point
    /// and intersection angle (in degrees)
    int test(coVRLine *line, osg::Vec3 *isectPoint, double *angle);
    /// tests plane for intersection or parallel with otherPlane
    /// returning state from enum, computing intersection line (2 Points)
    /// and intersection angle (in degrees)
    int test(coVRPlane *otherPlane, osg::Vec3 *isectLinePoint1, osg::Vec3 *isectLinePoint2, double *angle);
    /// updates menu, the points, the directions, drawables and name tag
    /// value is {P1, P2, P3, DIR1, DIR2, NORM} depending on what was changed
    void update(int value);
    /// hides/shows the name label
    void hideLabel(bool hide);

private:
    // variables of class
    static int const MAX_DRAW_POINTS = 8;
    static int _planeID_;
    static osg::BoundingBox *_boundingBox_;

    // variables
    string name_;
    osg::ref_ptr<osg::MatrixTransform> node_;
    int mode_;
    coVRPoint *point1_;
    coVRPoint *point2_;
    coVRPoint *point3_;
    coVRDirection *direction1_;
    coVRDirection *direction2_;
    coVRDirection *normalDirection_;
    coRowMenu *parentMenu_;
    coCheckboxGroup *modeGroup_;
    coCheckboxMenuItem *pointsModeCheckbox_;
    coCheckboxMenuItem *normalModeCheckbox_;
    coCheckboxMenuItem *directionsModeCheckbox_;
    coLabelMenuItem *sepLabel_;
    osg::Geometry *plane_;
    osg::Geode *planeGeode_;
    osg::ShapeDrawable *planeDraw_;
    osg::StateSet *stateSet_;
    osg::Material *material_;
    osg::Vec3 drawPoints_[MAX_DRAW_POINTS];
    osg::Vec4 color_;
    bool isVisible_;
    bool isChanged_;
    coVRLabel *nameTag_;
    osg::Vec3 nameTagPosition_;
    bool isBBSet_;
    bool showParamEqu_;
    bool showCoordEqu_;
    bool showNormEqu_;
    bool labelsShown_;
    int menuLanguage_;
    double scale_;
    int numDrawPoints_;
    osg::Vec3 oldNormalDirection_;
    osg::Vec3 oldDirection1_;
    osg::Vec3 oldDirection2_;

    // methods
    /// computes the angel (in degrees) between 2 directions
    double computeAngle(osg::Vec3 direction1, osg::Vec3 direction2);
    /// sorts the vector of the found intersection points with boundingbox
    /// computes point2 in vector mode (orthogonal to normal direction)
    osg::Vec3 computePoint2(osg::Vec3 normal);
    /// computes point3 in vector mode (orthogonal to normal direction)
    osg::Vec3 computePoint3(osg::Vec3 normal);
    /// computes and sets normal from dir1 and dir2
    /// with the same length and direction as the old one
    void updateNormal();
    ///initialises the plane
    int init();
    /// string with name and parametric, coordinate and normal equation
    string computeNameTagText();
    /// intersects plane with otherPlane, computing intersection line
    /// (2 points) and intersection angle (in degrees)
    /// no testing for skew and parallel
    bool isIntersection(coVRPlane *otherPlane, osg::Vec3 *isectLinePoint1, osg::Vec3 *isectLinePoint2, double *angle);
    /// materials for color
    void makeColor();
    /// pinboard tag with the name and equations
    void makeNameTag();
    /// updates position for pinboard name tag and equationtext
    void updateNameTag(bool planeChanged);
    /// computes the drawables for the plane
    int updatePlane();
};

#endif
