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

#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <cover/coVRMSController.h>
#include "coRectButtonGeometry.h"
#include <OpenVRUI/coValuePoti.h>
#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coFrame.h>
#include <config/CoviseConfig.h>
#include <string.h>

// OSG:
#include <osg/Node>
#include <osgDB/ReadFile>
#ifndef WIN32
#include <sys/time.h>
#endif
// Local:
#include "VideoLODPlugin.h"
#include "VideoLOD.h"

using namespace osg;
using namespace std;
using namespace cui;
using namespace osgDB;

static const string FILES("COVER.Plugin.VideoLOD.Files");

/// Constructor
VideoLODPlugin::VideoLODPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool VideoLODPlugin::init()
{
    cerr << "VideoLODPlugin init\n";

    offset = 0;

    //create panel
    panel = new coPanel(new coFlatPanelGeometry(coUIElement::BLACK));

    // Create main menu button
    VideoLODPluginMenuItem = new coSubMenuItem("VideoLOD");
    VideoLODPluginMenuItem->setMenuListener(this);
    cover->getMenu()->add(VideoLODPluginMenuItem);

    VideoLODPluginMenu = new coRowMenu("VideoLOD Menu");

    LoadMenuItem = new coSubMenuItem("Load");
    LoadMenuItem->setMenuListener(this);
    VideoLODPluginMenu->add(LoadMenuItem);

    //enable = new coCheckboxMenuItem("Enable", false);
    //enable->setMenuListener(this);
    //VideoLODPluginMenu->add(enable);

    load = new coRowMenu("Load");
    //load->setMenuListener(this);
    LoadMenuItem->setMenu(load);

    tiled = new coCheckboxMenuItem("Tiled Wall", false);
    tiled->setMenuListener(this);
    VideoLODPluginMenu->add(tiled);

    loop = new coCheckboxMenuItem("Loop", true);
    loop->setMenuListener(this);
    VideoLODPluginMenu->add(loop);

    /*mi = new coButtonMenuItem("Mystic India");
  mi->setMenuListener(this);
  load->add(mi);

  galaxyhd = new coButtonMenuItem("GalaxyHD");
  galaxyhd->setMenuListener(this);
  load->add(galaxyhd);

  galaxy4k = new coButtonMenuItem("Galaxy 4k");
  galaxy4k->setMenuListener(this);
  load->add(galaxy4k);

  galaxy4k2 = new coButtonMenuItem("long");
  galaxy4k2->setMenuListener(this);
  load->add(galaxy4k2);*/

    play = new coButtonMenuItem("Play");
    play->setMenuListener(this);
    VideoLODPluginMenu->add(play);

    pause = new coButtonMenuItem("Pause");
    pause->setMenuListener(this);
    VideoLODPluginMenu->add(pause);

    stop = new coButtonMenuItem("Stop");
    stop->setMenuListener(this);
    VideoLODPluginMenu->add(stop);

    ff = new coButtonMenuItem(" FF >>");
    ff->setMenuListener(this);
    VideoLODPluginMenu->add(ff);

    rw = new coButtonMenuItem("RW <<");
    rw->setMenuListener(this);
    VideoLODPluginMenu->add(rw);

    stats = new coButtonMenuItem("Stats");
    stats->setMenuListener(this);
    VideoLODPluginMenu->add(stats);

    scalepoti = new coPotiMenuItem("Frame Rate", 0, 100, 0);
    VideoLODPluginMenu->add(scalepoti);
    scalepoti->setMenuListener(this);

    set = new coButtonMenuItem("Set");
    set->setMenuListener(this);
    VideoLODPluginMenu->add(set);

    playback = new coSliderMenuItem("Playback", 0.0, 100.0, 0.0);
    playback->setMenuListener(this);
    VideoLODPluginMenu->add(playback);

    removelast = new coButtonMenuItem("Remove Last");
    removelast->setMenuListener(this);
    VideoLODPluginMenu->add(removelast);

    clear = new coButtonMenuItem("Clear All");
    clear->setMenuListener(this);
    VideoLODPluginMenu->add(clear);

    VideoLODPluginMenuItem->setMenu(VideoLODPluginMenu);

    /*MatrixTransform* mtnode = new MatrixTransform();
  Matrix mat;
  mat.makeTranslate(0.0f, -1080.0f, 0.00f);
  mtnode->setMatrix(mat);*/

    //this->nlod = new VideoLOD();
    //this->nlod2 = new VideoLOD();
    //mtnode->addChild(nlod);
    //cover->getObjectsRoot()->addChild(mtnode);
    //cover->getObjectsRoot()->addChild(nlod2);

    //cerr << "Scene: " << (cover->getScene())->className() << " parents: " << (cover->getScene())->getNumParents() << "\n";
    //cerr << "Scene: " << ((cover->getScene())->getParent(0))->className() << " parents: " << ((cover->getScene())->getParent(0))->getNumParents() << "\n";

    coCoviseConfig::ScopeEntries e = coCoviseConfig::getScopeEntries(FILES.c_str());
    const char **entries = e.getValue();
    if (entries)
    {
        while (*entries)
        {
            cerr << *entries << "\n";

            const char *fileName = *entries;

            if (fileName)
            {
                coButtonMenuItem *temp = new coButtonMenuItem(fileName);
                temp->setMenuListener(this);
                filelist.push_back(temp);
                struct fileloadinfo *info = new struct fileloadinfo;
                info->path = string(coCoviseConfig::getEntry("path", (FILES + string(".") + string(fileName)).c_str()));
                info->prefix = string(coCoviseConfig::getEntry("prefix", (FILES + string(".") + string(fileName)).c_str()));
                info->suffix = string(coCoviseConfig::getEntry("suffix", (FILES + string(".") + string(fileName)).c_str()));
                info->vnums = coCoviseConfig::getInt("vnums", (FILES + string(".") + string(fileName)).c_str(), 1);
                info->framestart = coCoviseConfig::getInt("framestart", (FILES + string(".") + string(fileName)).c_str(), 0);
                info->frameend = coCoviseConfig::getInt("frameend", (FILES + string(".") + string(fileName)).c_str(), 1);
                info->h = coCoviseConfig::getInt("h", (FILES + string(".") + string(fileName)).c_str(), 1080);
                info->w = coCoviseConfig::getInt("w", (FILES + string(".") + string(fileName)).c_str(), 1920);
                fileinfolist.push_back(info);
                cerr << info->path << " " << info->prefix << " " << info->suffix << " " << info->vnums << " " << info->framestart << " " << info->frameend << " " << info->h << " " << info->w << "\n";
            }

            entries++;
            entries++;
        }
    }

    for (int k = 0; k < filelist.size(); k++)
    {
        load->add(filelist.at(k));
    }

    //cerr << coCoviseConfig::getInt("w", "COVER.Plugin.VideoLOD.Files.Testfile", -1) << " " << coCoviseConfig::getInt("h", "COVER.Plugin.VideoLOD.Files.Testfile", -1) << " " << coCoviseConfig::getEntry("str", "COVER.Plugin.VideoLOD.Files.Testfile") << "\n";
    cerr << "VideoLODPlugin init done.\n";

    return true;
}

void VideoLODPlugin::runtime_init()
{
    //this->nlod = new VideoLOD();
}

/// Destructor
VideoLODPlugin::~VideoLODPlugin()
{
    offset = 0;
    for (int i = 0; i < vlod.size(); i++)
    {
        cover->getObjectsRoot()->removeChild(loadednodes.at(i));
    }
    vlod.clear();
    loadednodes.clear();
    delete panel;
    //delete mi;
    delete stats;
    delete LoadMenuItem;
    //delete galaxyhd;
    //delete galaxy4k;
    delete load;
    delete play;
    delete loop;
    delete tiled;
    delete pause;
    delete stop;
    delete ff;
    delete rw;
    delete clear;
    delete removelast;
    delete set;
    delete scalepoti;
    delete VideoLODPluginMenuItem;
    for (int i = 0; i < filelist.size(); i++)
    {
        delete filelist.at(i);
    }
    for (int i = 0; i < fileinfolist.size(); i++)
    {
        delete fileinfolist.at(i);
    }
    //delete enable;
}

void VideoLODPlugin::menuEvent(coMenuItem *menuItem)
{

    for (int i = 0; i < filelist.size(); i++)
    {
        if (filelist.at(i) == menuItem)
        {
            struct fileloadinfo *li = fileinfolist.at(i);
            if (vlod.size() > 0)
            {
                offset += li->h;
            }
            MatrixTransform *mtnode = new MatrixTransform();
            Matrix mat;
            mat.makeTranslate(0.0f, (float)(-(offset)), 0.00f);
            mtnode->setMatrix(mat);
            VideoLOD *temp = new VideoLOD();
            vlod.push_back(temp);
            mtnode->addChild(temp);
            loadednodes.push_back(mtnode);
            cover->getObjectsRoot()->addChild(mtnode);

            vlod.back()->setDataSet(li->path, li->prefix, li->suffix, li->framestart, li->frameend, li->vnums);
            //offset += li->h;

            if (tiled->getState() && coVRMSController::instance()->isMaster())
            {
                vlod.back()->setMaster(true);
                //nlod2->setMaster(true);
            }

            addlist.push_back(li);

            return;
        }
    }

    if (vlod.size() == 0)
    {
        return;
    }

    if (menuItem == clear)
    {
        offset = 0;
        for (int i = 0; i < vlod.size(); i++)
        {
            cover->getObjectsRoot()->removeChild(loadednodes.at(i));
        }
        vlod.clear();
        loadednodes.clear();
        return;
    }

    if (menuItem == removelast)
    {
        struct fileloadinfo *li = addlist.back();
        if (offset != 0)
            offset -= li->h;

        cover->getObjectsRoot()->removeChild(loadednodes.back());
        loadednodes.pop_back();
        vlod.pop_back();
        addlist.pop_back();

        return;
    }

    vlod.back()->setLoop(loop->getState());
    //cerr << "menuEvent\n";
    // listen for initPDB frame to open close
    if (menuItem == set)
    {
        vlod.back()->setFrameRate((int)scalepoti->getValue());
    }

    if (menuItem == playback)
    {
        vlod.back()->seek(playback->getValue() / 100.0);
    }

    //if(menuItem == mi || menuItem == galaxyhd || menuItem == galaxy4k || menuItem == galaxy4k2)
    //{
    // cerr << "Load\n";
    //cerr << "in menuEvent\n";
    /*if(1)
    {

      if(menuItem == mi)
      {
        ((VideoLOD*)nlod)->setDataSet(string("/home/jschulze/svn/trunk/covise/src/renderer/OpenCOVER/plugins/Image/images/"), string("mi_cropped.0000"), string(""), 49, 152, 3);
        ((VideoLOD*)nlod2)->setDataSet(string("/home/jschulze/svn/trunk/covise/src/renderer/OpenCOVER/plugins/Image/images/"), string("mi_cropped.0000"), string(""), 49, 152, 3);
        file->path = string("/home/jschulze/svn/trunk/covise/src/renderer/OpenCOVER/plugins/Image/images/");
        file->prefix = string("mi_cropped.0000");
        file->vnums = 3;
        file->framestart = 49;
        file->frameend = 152;*/
    /*}
      else if(menuItem == galaxyhd)
      {
        //((VideoLOD*)nlod)->setDataSet(string("/home/jschulze/data/galaxy/osga/"), string("hdgc8-17_4k.10"), string(""), 0, 99, 2);
        //((VideoLOD*)nlod2)->setDataSet(string("/home/jschulze/data/galaxy/osga/"), string("hdgc8-17_4k.10"), string(""), 0, 99, 2);
        ((VideoLOD*)nlod)->setDataSet(string("/home/jschulze/data/galaxy/osga/"), string("hdgc8-17_4k.10"), string(""), 0, 99, 2);
        //((VideoLOD*)nlod)->setDataSet(string("/home/jschulze/data/galaxy/osgamt/"), string("hdgc8-17_4k.10"), string(""), 0, 99, 2);
        //((VideoLOD*)nlod2)->setDataSet(string("/home/jschulze/data/galaxy/osgacom/"), string("hdgc8-17_4k.10"), string(""), 0, 99, 2);
        file->path = string("/home/jschulze/data/galaxy/osga/");
        file->prefix = string("hdgc8-17_4k.10");
        file->vnums = 2;
        file->framestart = 0;
        file->frameend = 99;*/
    /*}
      else if(menuItem == galaxy4k)
      {
        //((VideoLOD*)nlod)->setDataSet(string("/home/jschulze/data/galaxy/osga4k/"), string("gc8-17_4k.10"), string(""), 0, 99, 2);
        //((VideoLOD*)nlod2)->setDataSet(string("/home/jschulze/data/galaxy/osga4k/"), string("gc8-17_4k.10"), string(""), 0, 99, 2);
        ((VideoLOD*)nlod)->setDataSet(string("/home/jschulze/data/galaxy/osga4kc/"), string("gc8-17_4k.10"), string(""), 0, 99, 2);
        //((VideoLOD*)nlod2)->setDataSet(string("/home/jschulze/data/galaxy/osga4kc/"), string("gc8-17_4k.10"), string(""), 0, 99, 2);
        file->path = string("/home/jschulze/data/galaxy/osga4k/");
        file->prefix = string("gc8-17_4k.10");
        file->vnums = 2;
        file->framestart = 0;
        file->frameend = 99;*/
    /*}
      else if(menuItem == galaxy4k2)
      {
        //((VideoLOD*)nlod)->setDataSet(string("/home/jschulze/data/galaxy/osga4k2/"), string("gc8-17_4k.10"), string(""), 0, 99, 2);
        //((VideoLOD*)nlod2)->setDataSet(string("/home/jschulze/data/galaxy/osga4k2/"), string("gc8-17_4k.10"), string(""), 0, 99, 2);
        ((VideoLOD*)nlod)->setDataSet(string("/tmp/data/osga/"), string("ptest"), string(""), 1, 7637, 6);
//        ((VideoLOD*)nlod)->setDataSet(string("/tmp/data/osgast/"), string("ptest"), string(""), 1, 203, 6);
        //((VideoLOD*)nlod2)->setDataSet(string("/tmp/data/osga/"), string("ptest"), string(""), 1, 7637, 6);
        file->path = string("/home/jschulze/data/galaxy/osga4k2/");
        file->prefix = string("gc8-17_4k.10");
        file->vnums = 2;
        file->framestart = 0;
        file->frameend = 99;*/
    /*}

      nlod->setPauseType(VideoLOD::WHAT_YOU_SEE);


      //cerr << "numoftiles: " << ((VideoLOD*)node)->numoftiles << " maxlevel: " << ((VideoLOD*)node)->maxlevel << " tilearea: " << ((VideoLOD*)node)->tilearea << " sync: " << ((VideoLOD*)node)->sync << " treenodes: " << ((VideoLOD*)node)->treenodes.size() << "\n";



      if(tiled->getState() && coVRMSController::instance()->isMaster())
      {
        nlod->setMaster(true);
        //nlod2->setMaster(true);
      }



    }
    else
    {
      menuEvent(stop);
      menuEvent(menuItem);
    }*/
    //}

    if (menuItem == play)
    {
        cerr << "Play\n";
        vlod.back()->setPlayMode(VideoLOD::PLAY);
        //((VideoLOD*)nlod2)->setPlayMode(VideoLOD::PLAY);
    }

    if (menuItem == pause)
    {
        cerr << "Pause\n";
        vlod.back()->setPlayMode(VideoLOD::PAUSE);
        //((VideoLOD*)nlod2)->setPlayMode(VideoLOD::PAUSE);
    }

    if (menuItem == ff)
    {
        cerr << "FF\n";
        vlod.back()->setPlayMode(VideoLOD::FF);
        //((VideoLOD*)nlod2)->setPlayMode(VideoLOD::FF);
    }

    if (menuItem == rw)
    {
        cerr << "RW\n";
        vlod.back()->setPlayMode(VideoLOD::RW);
        //((VideoLOD*)nlod2)->setPlayMode(VideoLOD::RW);
    }

    if (menuItem == stop)
    {
        cerr << "Stop\n";
        vlod.back()->stop();
        //((VideoLOD*)nlod2)->stop();
    }

    if (menuItem == stats)
    {

        //nlod->setFrameRate((int)scalepoti->getValue());
        //nlod->seek((scalepoti->getValue()) / 100.0);
        //nlod->seek(((int)scalepoti->getValue()) * 70);
        vlod.back()->printout();
        //((VideoLOD*)nlod2)->printout();
    }
}

// need to define because abstract
void VideoLODPlugin::potiValueChanged(float, float, coValuePoti *, int)
{
}

// load new structure listener
void VideoLODPlugin::buttonEvent(coButton *)
{
}

/// Called before each frame
void VideoLODPlugin::preFrame()
{
    for (int i = 0; i < vlod.size(); i++)
    {
        vlod.at(i)->advanceFrame();
    }
    //((VideoLOD*)nlod2)->advanceFrame();
    if (vlod.size() > 0)
    {
        playback->setValue(vlod.back()->getPos() * 100.0);
    }
    else
    {
        playback->setValue(0.0);
    }
    //cerr << "pos: " << ((VideoLOD*)nlod)->getPos() *100.0 << "\n";
    //nlod->setScale();
    //nlod2->setScale();
}

COVERPLUGIN(VideoLODPlugin)
