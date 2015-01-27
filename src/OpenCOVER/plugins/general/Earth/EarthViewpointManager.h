/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EarthViewpointManager_H
#define EarthViewpointManager_H 1

#include <cover/coTabletUI.h>

#include <osgEarth/Map>
#include <osgEarth/Viewpoint>
#include "EarthViewpoint.h"
#include <list>

using namespace osgEarth;
using namespace opencover;

//---------------------------------------------------------------------------
class EarthViewpointManager : public coTUIListener
{

public:
    EarthViewpointManager(coTUITab *et);
    void addViewpoint(Viewpoint *);
    float getRPM()
    {
        return rpmField->getValue();
    };

protected:
    coTUITab *earthTab;
    osg::ref_ptr<osgEarth::Map> map;

    typedef std::list<EarthViewpoint *> ViewpointList;
    ViewpointList viewpoints;
    virtual void tabletPressEvent(coTUIElement *);

    coTUIFrame *ViewpointsFrame;
    coTUIFrame *ViewpointOptionsFrame;
    coTUILabel *rpmLabel;
    coTUIEditFloatField *rpmField;
    coTUIButton *printButton;
};

#endif // EarthViewpointManager_H
