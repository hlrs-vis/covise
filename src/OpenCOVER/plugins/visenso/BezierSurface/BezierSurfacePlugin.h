/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2010 Visenso  **
 **                                                                        **
 ** Description: BezierPlugin                                              **
 **              for Cyberclassroom mathematics                            **
 **                                                                        **
 ** Author: T.Milbich                                                      **
 **                                                                        **
 ** History:                                                               **
 **     12.2010 initial version                                            **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef _BEZIERSURFACEPLUGIN_H
#define _BEZIERSURFACEPLUGIN_H

#include <cover/coVRPlugin.h>
#include "BezierSurfaceVisualizer.h"
#include "Point.h"
#include <string>

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include "cover/coVRLabel.h"

#include <osg/Vec3>
#include <osg/Geometry>
#include <osg/Array>
#include <osg/Matrix>
#include <osg/Material>

using namespace opencover;
using namespace vrui;

class BezierSurfacePlugin : public coVRPlugin, public coMenuListener
{
public:
    enum Reduction
    {
        FARIN,
        FOREST
    };

    static BezierSurfacePlugin *plugin;

    BezierSurfacePlugin();
    virtual ~BezierSurfacePlugin();
    virtual bool destroy();
    virtual bool init();
    void preFrame();

    virtual void guiToRenderMsg(const char *msg);

private:
    osg::ref_ptr<osg::MatrixTransform> node_;
    osg::Geometry *plane_;
    osg::Geode *planeGeode_;
    osg::StateSet *stateSet_;
    osg::Material *material_;
    osg::Vec4 color_;

    BezierSurfaceVisualizer *surface;
    std::vector<Point *> controlPoints;
    int n;
    int m;
    int presentationStepCounter;
    double parameterValueAnimationU;
    double parameterValueAnimationV;
    double parameterValueStepV;
    double parameterValueStepU;
    bool showCasteljauStep;
    bool showControlPolygon;
    bool showInteractors;
    bool showCasteljauAnimation;
    bool firstLoad;
    float scale;

    coRowMenu *objectsMenu;
    coButtonMenuItem *menuItemdegreeElevationU;
    coButtonMenuItem *menuItemdegreeElevationV;
    //coButtonMenuItem *menuItemHideCasteljau;
    //coButtonMenuItem *menuItemShowControlPolygon;
    //coButtonMenuItem *menuItemShowInteractors;
    //coButtonMenuItem *menuItemAnimateCasteljau;
    coCheckboxMenuItem *menuItemShowControlPolygon;
    coCheckboxMenuItem *menuItemShowInteractors;
    coCheckboxMenuItem *menuItemAnimateCasteljau;
    coCheckboxMenuItem *menuItemShowCasteljau;
    coButtonMenuItem *menuItemdegreeReductionForestU;
    coButtonMenuItem *menuItemdegreeReductionForestV;
    coButtonMenuItem *menuItemdegreeReductionFarinU;
    coButtonMenuItem *menuItemdegreeReductionFarinV;
    coSliderMenuItem *menuItemParameterValueU;
    coSliderMenuItem *menuItemParameterValueV;
    coLabelMenuItem *menuItemSeparator1;
    coLabelMenuItem *menuItemSeparator2;
    coLabelMenuItem *menuItemSeparator3;

    coVRLabel *labelU;
    coVRLabel *labelV;
    coVRLabel *labelStatus;
    coVRLabel *casteljauLabel;

    void makeSurface();
    void createMenu();
    void menuEvent(coMenuItem *menuItem);
    void elevateDegreeOfSurface(char direction);
    void reduceDegreeOfSurface(char direction, Reduction reduction);
    void casteljauStep();
    void casteljauAnimation();
    void removeMenuEntries();
    void changeStatus();
    //      void message(int , int , const void *iData);
    void setMenuVisible(bool visible);
};

#endif
