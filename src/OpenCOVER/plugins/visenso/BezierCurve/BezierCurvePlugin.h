/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2010 Visenso  **
 **                                                                        **
 ** Description: BezierCurvePlugin                                              **
 **              for Cyberclassroom mathematics                            **
 **                                                                        **
 ** Author: T.Milbich                                                      **
 **                                                                        **
 ** History:                                                               **
 **     12.2010 initial version                                            **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef _BEZIERCURVEPLUGIN_H
#define _BEZIERCURVEPLUGIN_H

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include "cover/coVRLabel.h"

#include <cover/coVRPlugin.h>
#include <config/CoviseConfig.h>

#include <osg/Vec3>
#include <osg/Geometry>
#include <osg/Array>
#include <osg/Material>

#include "BezierCurveVisualizer.h"
#include "Point.h"

using namespace opencover;
using namespace vrui;

class BezierCurvePlugin : public coVRPlugin, public coMenuListener
{
public:
    static BezierCurvePlugin *plugin;

    BezierCurvePlugin();
    virtual ~BezierCurvePlugin();
    virtual bool destroy();
    virtual bool init();
    void preFrame();

    virtual void guiToRenderMsg(const char *msg);

private:
    // variables
    osg::ref_ptr<osg::MatrixTransform> node_;
    osg::Geometry *plane_;
    osg::Geode *planeGeode_;
    osg::StateSet *stateSet_;
    osg::Material *material_;
    osg::Vec4 color_;

    BezierCurveVisualizer *curve;
    BezierCurveVisualizer::Computation computation;
    std::vector<Point *> controlPoints;
    float parameterValueAnimation;
    float parameterValueStep;
    bool showCasteljauAnimation;
    bool showCasteljauStep;
    bool showTangents;
    bool firstLoad;

    float scale;
    int presentationStepCounter;

    coRowMenu *objectsMenu;
    coButtonMenuItem *menuItemObjectsAddPoint;
    coButtonMenuItem *menuItemObjectsRemovePoint;
    coCheckboxMenuItem *menuItemAnimateCasteljau;
    coCheckboxMenuItem *menuItemShowCasteljau;
    coButtonMenuItem *menuItemdegreeElevation;
    coButtonMenuItem *menuItemdegreeReductionForest;
    coButtonMenuItem *menuItemdegreeReductionFarin;
    coCheckboxMenuItem *menuItemshowTangents;
    coSliderMenuItem *menuItemParameterValue;
    coLabelMenuItem *menuItemSeparator1;
    coLabelMenuItem *menuItemSeparator2;

    coVRLabel *casteljauLabel;

    void makeCurve();
    void initializeCurve();
    void addNewPoint();
    void addNewPoint(Point *newPoint);
    void casteljauAnimation();
    void removePoint();
    void createMenu();
    void menuEvent(coMenuItem *menuItem);
    void casteljauStep();
    void setParameterValueStep(float pv);
    void setParameterValueAnimation(float pv);
    void elevateDegreeOfCurve();
    void reduceDegreeOfCurveForest();
    void reduceDegreeOfCurveFarin();
    //      void message(int , int , const void *iData);
    void removeMenuEntries();
    void changeStatus();

    void setMenuVisible(bool visible);
};

#endif
