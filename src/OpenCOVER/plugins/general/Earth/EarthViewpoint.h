/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EarthViewpoints_H
#define EarthViewpoints_H 1

#include <cover/coTabletUI.h>

#include <osgEarth/Map>
#include <osgEarth/Viewpoint>
#include <OpenVRUI/coUpdateManager.h>

using namespace osgEarth;
using namespace opencover;

//---------------------------------------------------------------------------
class EarthViewpoint : public coTUIListener, public vrui::coUpdateable
{

public:
    EarthViewpoint(coTUIFrame *parent, Viewpoint *viewpoint, int ViewpointNumber);

    virtual void tabletPressEvent(coTUIElement *);
    void setViewpoint();

protected:
    int viewpointNumber;
    Viewpoint *viewpoint;
    coTUIFrame *parentFrame;
    coTUIButton *viewpointButton;
    coTUIToggleButton *animateButton;
    osg::Vec3d new_center;

    virtual bool update();
    bool firstTime;
    osg::Matrix toUpright;
    double startTime;
    void setScale();
    void computeToUpright();
    /*coTUIEditFloatField *xf;
	coTUIEditFloatField *yf;
	coTUIEditFloatField *zf;*/
};

#endif // EarthViewpoints_H
