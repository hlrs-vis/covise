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
#include <cover/coVRFileManager.h>
using namespace covise;
using namespace opencover;

#include "cover/coTabletUI.h"
#include <osg/Geode>
#include <osg/ref_ptr>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/LineWidth>
#include <PluginUtil/coSphere.h>
#include <QStringList>
#include <QMap>

class CNCPlugin : public coVRPlugin, public coTUIListener
{
public:
    CNCPlugin();
    virtual ~CNCPlugin();
    static CNCPlugin *instance();
    bool init();

    int loadGCode(const char *filename, osg::Group *loadParent);
    static int sloadGCode(const char *filename, osg::Group *loadParent, const char *covise_key);
    static int unloadGCode(const char *filename, const char *covise_key);

    void straightFeed(double x, double y, double z, double a, double b, double c, double feedRate);

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
    coTUIComboBox *mapChoice;
    QStringList mapNames;
    QMap<QString, int> mapSize;
    QMap<QString, float *> mapValues;

    int currentMap;
    osg::ref_ptr<osg::StateSet> geoState;
    osg::ref_ptr<osg::Material> linemtl;
    osg::ref_ptr<osg::LineWidth> lineWidth;
    void setTimestep(int t);

    osg::Vec4 getColor(float pos);
    void deleteColorMap(const QString &name);

    virtual void tabletPressEvent(coTUIElement *tUIItem);
    virtual void tabletReleaseEvent(coTUIElement *tUIItem);
    virtual void tabletEvent(coTUIElement *tUIItem);

private:
    bool playing;
    int frameNumber;
    double recordRate;
    //float *positions;
    float *lookat[3];
    float length;
    const char **objectName;
    char *filename;
    osg::Group *parentNode;
    osg::Vec3Array *vert;
    osg::Vec4Array *color;
    osg::DrawArrayLengths *primitives;

    static CNCPlugin *thePlugin;

    void save();

    osg::ref_ptr<osg::Geometry> geom;
    osg::ref_ptr<osg::Geode> geode;
};
#endif
