/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2009 Visenso  **
 **                                                                        **
 ** Description: coVRDistance                                              **
 **              Draws two perpendicular points and the connecting line    **
 **               with a nameTag and distance                              **
 **               only within the bounding box (needs to be set)           **
 **                                                                        **
 ** Author: M. Theilacker                                                  **
 **                                                                        **
 ** History:                                                               **
 **     5.2009 initial version                                             **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef _COVRDISTANCE_H
#define _COVRDISTANCE_H

#include "coVRPoint.h"

#include <osg/Vec3>
#include <osg/BoundingBox>
#include <osg/MatrixTransform>

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coCheckboxMenuItem.h>

class coVRDistance
{
public:
    // constructor destructor
    coVRDistance(osg::Vec3 point1, osg::Vec3 point2, int mode, double lineRadius = 0.08);
    ~coVRDistance();

    enum
    {
        ONLY_LINE = 0,
        POINT_LINE = 1,
        POINT_LINE_POINT = 2,
    };

    // methods of class
    static void setBoundingBox(osg::BoundingBox *boundingBox);

    // methods
    /// adds the visiblility checkbox to the parent menu
    int addToMenu(coRowMenu *parentMenu_, int posInMenu);
    double getDistance();
    bool isVisible();
    /// takes care of menu events from parent class
    void menuEvent(coMenuItem *menuItem);
    /// preparation for each frame
    void preFrame();
    void removeFromMenu();
    void setMode(int mode);
    void setVisible(bool visible);
    /// updates menu, the points, the lines and name tag
    void update(osg::Vec3 point1, osg::Vec3 point2);
    void hideLabels(bool hide);

private:
    // variables of class
    static osg::BoundingBox *_boundingBox_;

    // variables
    osg::ref_ptr<osg::MatrixTransform> node_;
    int mode_;
    coVRPoint *point1_;
    coVRPoint *point2_;
    coVRLabel *nameTag_;
    osg::Cylinder *line_;
    osg::Geode *lineGeode_;
    osg::ShapeDrawable *lineDraw_;
    double lineRadius_;
    coRowMenu *parentMenu_;
    coCheckboxMenuItem *visibleCheckbox_;
    bool isVisible_;
    bool isBBSet_;
    bool labelsShown_;
    int menuLanguage_;

    // methods
    /// string with name and distance
    string computeNameTagText();
    /// string with checkbox text and distance
    string computeCheckboxText();
    /// pinboard tag with the distance
    void makeNameTag();
    /// updates the cylinder for the connecting line
    int updateLine();
    /// updates position for pinboard name tag and equationtext
    void updateNameTag();
};

#endif
