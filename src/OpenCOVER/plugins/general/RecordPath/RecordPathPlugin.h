/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _RecordPath_PLUGIN_H
#define _RecordPath_PLUGIN_H
/****************************************************************************\
 **                                                            (C)2005 HLRS  **
 **                                                                          **
 ** Description: RecordPath Plugin (records viewpoints and viewing directions and targets)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                 **
 **                                                                          **
 ** History:  								                                 **
 ** April-05  v1	    				       		                         **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include "cover/coTabletUI.h"
#include <osg/Geode>
#include <osg/ref_ptr>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/LineWidth>
#include <PluginUtil/coSphere.h>

class RecordPathPlugin : public coVRPlugin, public coTUIListener
{
public:
    RecordPathPlugin();
    virtual ~RecordPathPlugin();
    bool init();

    // this will be called in PreFrame
    void preFrame();
    coTUITab *PathTab;
    coTUIToggleButton *record;
    coTUIButton *stop;
    coTUIButton *play;
    coTUIButton *reset;
    coTUIButton *saveButton;
    coTUIToggleButton *viewPath;
    coTUIToggleButton *viewlookAt;
    coTUIToggleButton *viewDirections;
    coTUILabel *numSamples;
    coTUILabel *recordRateLabel;
    coTUIEditIntField *recordRateTUI;
    coTUILabel *lengthLabel;
    coTUIEditFloatField *lengthEdit;
    coTUILabel *radiusLabel;
    coTUIEditFloatField *radiusEdit;
    coTUIFileBrowserButton *fileNameBrowser;
    coTUIComboBox *renderMethod;
    osg::ref_ptr<osg::StateSet> geoState;
    osg::ref_ptr<osg::Material> linemtl;
    osg::ref_ptr<osg::LineWidth> lineWidth;

    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletReleaseEvent(coTUIElement *tUIItem);
    virtual void tabletEvent(coTUIElement *tUIItem);

private:
    bool playing;
    int frameNumber;
    double recordRate;
    float *positions;
    float *lookat[3];
    float length;
    const char **objectName;
    char *filename;

    void save();

    osg::ref_ptr<osg::Geometry> geom;
    osg::ref_ptr<osg::Geode> geode;

    osg::ref_ptr<osg::Geode> dirGeode;
    osg::ref_ptr<osg::Geometry> dirGeom;

    osg::ref_ptr<osg::Geode> destGeode;
    osg::ref_ptr<osg::Geometry> destGeom;

    osg::ref_ptr<osg::Geode> lookAtGeode;
    osg::ref_ptr<coSphere> lookAtSpheres;
};
#endif
