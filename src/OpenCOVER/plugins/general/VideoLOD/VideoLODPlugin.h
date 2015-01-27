/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// Description:
//
// Author:
//
// Creation Date:
//
// **************************************************************************

#ifndef _VIDEOLOD_PLUGIN_H_
#define _VIDEOLOD_PLUGIN_H_

#include "VideoLOD.h"
#include "coMenu.h"
#include <osg/Geometry>
#include "coMenu.h"
#include <Interaction.h>
#include <OpenVRUI/coNavInteraction.h>
#include <OpenVRUI/coMouseButtonInteraction.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <osg/Texture2D>
#include <osg/Geometry>
#include <osg/Vec3>
#include <osg/Vec2>
#include <osg/Vec4>
#include <osg/StateSet>
#include <osg/StateAttribute>
#include <osg/PrimitiveSet>
#include <osg/Vec2>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Matrix>
#include <osg/Switch>

#include <iostream>
#include <fstream> // file I/O
class coFrame;
class coPanel;
class coButtonMenuItem;
class coPopupHandle;
class coButton;
class coPotiItem;
class coLabelItem;

class VideoLODPlugin : public coVRPlugin, public coMenuListener, public coValuePotiActor
{
    osg::Node *node;
    osg::Node *root;
    coSubMenuItem *VideoLODPluginMenuItem;
    coSubMenuItem *LoadMenuItem;
    coRowMenu *VideoLODPluginMenu;
    //coCheckboxMenuItem* enable;
    coPanel *panel;
    coFrame *frame;
    coRowMenu *load;
    coCheckboxMenuItem *tiled;
    coCheckboxMenuItem *loop;
    coPotiMenuItem *scalepoti;
    coButtonMenuItem *play;
    vector<coButtonMenuItem *> filelist;
    coButtonMenuItem *pause;
    coButtonMenuItem *stop;
    coButtonMenuItem *ff;
    coButtonMenuItem *rw;
    coButtonMenuItem *stats;
    coButtonMenuItem *set;
    coSliderMenuItem *playback;
    coButtonMenuItem *removelast;
    coButtonMenuItem *clear;

    string format;
    char *number;

    int frame_num;
    int firstrun;
    int counter;

    void runtime_init();
    void menuEvent(coMenuItem *);

protected:
    struct fileloadinfo
    {
        string path;
        string prefix;
        string suffix;
        int vnums;
        int framestart;
        int frameend;
        int h;
        int w;
    };
    int offset;
    vector<osg::Node *> loadednodes;
    vector<fileloadinfo *> addlist;
    vector<fileloadinfo *> fileinfolist;
    int playmode;
    void potiValueChanged(float oldvalue, float newvalue, coValuePoti *poti, int context);
    vector<osg::VideoLOD *> vlod;

public:
    //osg::VideoLOD * nlod;
    //osg::VideoLOD * nlod2;
    VideoLODPlugin();
    ~VideoLODPlugin();
    bool init();
    void buttonEvent(coButton *);
    void preFrame();
};

#endif

// EOF
