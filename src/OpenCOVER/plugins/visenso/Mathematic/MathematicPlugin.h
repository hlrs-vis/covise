/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2009 Visenso  **
 **                                                                        **
 ** Description: MathematicPlugin                                          **
 **              for VR4Schule mathematics                                 **
 **                                                                        **
 ** Author: M. Theilacker                                                  **
 **                                                                        **
 ** History:                                                               **
 **     4.2009 initial version                                             **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef _MATH_PLUGIN_H
#define _MATH_PLUGIN_H

#include <vector>
#include <string>

#include <osg/BoundingBox>

#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>

#include <OpenVRUI/osg/OSGVruiMatrix.h>

#include <OpenVRUI/sginterface/vruiMatrix.h>

#include <config/CoviseConfig.h>

#include "coVRCoordinateAxis.h"
#include "coVRDistance.h"
#include "coVRPoint.h"
#include "coVRLine.h"
#include "coVRPlane.h"

namespace vrui
{
class coRowMenu;
class coButtonMenuItem;
class coVRModule;
}

using namespace vrui;
using namespace opencover;

class MathematicPlugin : public opencover::coVRPlugin, public vrui::coMenuListener
{
public:
    // variables of class
    static MathematicPlugin *plugin;

    // constructor destructor
    MathematicPlugin();
    virtual ~MathematicPlugin();

    // methods of class
    /// rounds all 3 values of osg::Vec to .0 or .5
    static osg::Vec3 roundVec2(osg::Vec3 vec);
    /// rounds double to .0 or .5
    static double round2(double num);
    /// rounds all 3 values of osg::Vec to .0 till .9
    static osg::Vec3 roundVec10(osg::Vec3 vec);
    /// rounds double to .0 till .9
    static double round10(double num);
    /// computes the angle (in degrees) between 2 directions
    static double computeAngle(osg::Vec3 direction1, osg::Vec3 direction2);

    // methods
    /// init
    virtual bool init();
    /// react on menu events
    virtual void menuEvent(coMenuItem *menuItem);

    /// preparation for each frame
    void preFrame();
    void showBoundingBox(bool show);

    // variables
    vrui::coRowMenu *mathematicsMenu_;

private:
    // variables
    double boundary_;
    coCheckboxMenuItem *showAxisMenuItem_;
    coCheckboxMenuItem *hideLabelsMenuItem_;
    coButtonMenuItem *addPointMenuItem_;
    coButtonMenuItem *addLineMenuItem_;
    coButtonMenuItem *addPlaneMenuItem_;
    vector<coButtonMenuItem *> deleteButtonsVec_;
    int deleteButtonNum_;
    coLabelMenuItem *stateLabel_;
    coLabelMenuItem *sepMainMenu_;
    int mainMenuSepPos_;
    coVRCoordinateAxis *axis_;
    vector<coVRPoint *> pointsVec_;
    vector<coVRLine *> linesVec_;
    vector<coVRPlane *> planesVec_;
    /// line intersections [0]g0g1 [1]g1g2 [2]g2g0
    vector<coVRPoint *> isectPoints_;
    /// plane intersections [0]E0E1
    vector<coVRLine *> isectLines_;
    /// line states [0]g0g1 [1]g1g2 [2]g2g0
    /// plane line states [0]E0g0
    /// plane states [0]E0E1
    vector<int> statesVec_;
    vector<double> anglesVec_;
    /// info of gemetry [i]type [i+1]numInVec
    vector<string> geometryInfoVec_;
    /// line perpendiculars, distance between points, points and lines/planes or planes
    vector<coVRDistance *> distancesVec_;
    osg::BoundingBox *boundingBox_;
    osg::Geode *boxGeode_;
    int geometryCount_;

    // methods
    void addPoint();
    void addLine();
    void addPlane();
    /// text for the menu
    string computeLineIsectText(int line);
    /// text for the menu
    string computeLineText(int line1, int line2, int lineState);
    /// text for the menu
    string computePlaneIsectText(int plane);
    string computePlaneLineIsectText(int plane);
    string computePlaneLineText(int plane, int state);
    string computePlaneText(int plane1, int plane2, int planeState);
    void drawBoundingBox(osg::BoundingBox *box);
    void makeLineDistance2Point(int line1, int point);
    void makeLineDistance2Line(int line1, int line2);
    void makeLineIntersections(int line1, int line2);
    void makePlaneDistance2Line(int plane, int line);
    void makePlaneDistance2Plane(int plane1, int plane2);
    void makePlaneDistance2Point(int plane, int point);
    void makePlaneLineIntersections(int plane, int line);
    void makePlaneIntersections(int plane1, int plane2);
    void makePointDistance2Point(int point1, int point2);
    void makeMathematicsMenu();
    void testLines(int line1, int line2);
    void testPlaneLine(int plane, int line);
    void testPlanes(int plane1, int plane2);
    void updateStateLabel();
    void hideAllLabels(bool hide);
};

#endif
