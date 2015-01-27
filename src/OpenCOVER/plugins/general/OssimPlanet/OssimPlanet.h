/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _OSSIM_PLANET_PLUGIN_H_
#define _OSSIM_PLANET_PLUGIN_H_

#include "coMenu.h"
#include <osg/Group>
#include <cover/coVRPlugin.h>
namespace covise
{
class coFrame;
class coPanel;
class coButtonMenuItem;
class coButton;
class coPotiItem;
class coLabelItem;
}
using namespace covise;
using namespace opencover;

/** Plugin to load ossimPlanet data
  @author Jurgen Schulze (jschulze@ucsd.edu)
*/
class OssimPlanet : public coVRPlugin, public coMenuListener, public coButtonActor, public coValuePotiActor
{
protected:
    coSubMenuItem *_ossimPlanetMenuItem;
    coPanel *_panel;
    coFrame *_frame;
    osgDB::DatabasePager *_databasePager;
    osg::ref_ptr<ossimPlanet> _planet;
    coRowMenu *_ossimPlanetMenu;
    coCheckboxMenuItem *_showPlanetItem;
    coCheckboxMenuItem *_hudModeItem;
    coCheckboxMenuItem *_wireframeModeItem;
    coPotiMenuItem *_elevationDial;
    coPotiMenuItem *_splitMetricDial;
    coCheckboxMenuItem *_flatlandModeItem;
    coSubMenuItem *_layersMenuItem;
    coRowMenu *_layersMenu;
    coCheckboxMenuItem *_bordersItem;

    void menuEvent(coMenuItem *);
    void menuReleaseEvent(coMenuItem *);
    void potiValueChanged(float oldvalue, float newvalue, coValuePoti *poti, int context);
    void createPlanet();
    void setWireFrameMode(bool);

public:
    OssimPlanet();
    ~OssimPlanet();
    bool init();
    void buttonEvent(coButton *);
    void preFrame();
};

#endif

// EOF
