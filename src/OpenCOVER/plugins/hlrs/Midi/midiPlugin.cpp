/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef WIN32
#include <winsock2.h>
#include <windows.h>
#include <direct.h>
#endif
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <PluginUtil/coSphere.h>
#include <cover/coVRTui.h>
#include <config/CoviseConfig.h>
#include "midiPlugin.h"

#include <osg/GL>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/PrimitiveSet>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/CullFace>
#include <osg/Light>
#include <osg/LightSource>
#include <osg/Depth>
#include <osgDB/ReadFile>
#include <osg/Program>
#include <osg/Shader>
#include <osg/Point>
#include <osg/ShadeModel>
#include <osg/BlendFunc>
#include <osg/AlphaFunc>

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>

#include <osgGA/GUIEventAdapter>

#ifndef WIN32
// for chdir
#include <unistd.h>
#endif

#ifdef _WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

MidiPlugin *MidiPlugin::plugin = NULL;

FILE *outfile = stderr;


static FileHandler handlers[] = {
    { NULL,
      MidiPlugin::loadMidi,
      MidiPlugin::loadMidi,
      MidiPlugin::unloadMidi,
      "MID" },
    { NULL,
      MidiPlugin::loadMidi,
      MidiPlugin::loadMidi,
      MidiPlugin::unloadMidi,
      "mid" }
};

int MidiPlugin::loadMidi(const char *filename, osg::Group *parent, const char *)
{

    cerr << "Read the file " << filename << endl;
    return plugin->loadFile(filename, parent);
}

int MidiPlugin::unloadMidi(const char *filename, const char *)
{

    cerr << "unload the file " << filename << endl;
    return plugin->unloadFile();
}

int MidiPlugin::loadFile(const char *filename, osg::Group *parent)
{

      midifile.read(filename);

   int numTracks = midifile.getTrackCount();
   cout << "TPQ: " << midifile.getTicksPerQuarterNote() << endl;
   if (numTracks > 1) {
      cout << "TRACKS: " << numTracks << endl;
   }
   for (int track=1; track < numTracks; track++) {
      tracks.push_back(new Track(track));
     /* if (tracks > 1) {
         cout << "\nTrack " << track << endl;
      }
      for (int event=0; event < midifile[track].size(); event++) {
         cout << dec << midifile[track][event].tick;
         cout << '\t' << hex;
         for (int i=0; i<midifile[track][event].size(); i++) {
            cout << (int)midifile[track][event][i] << ' ';
         }
         cout << endl;
      }*/
   }
    return 1;
}

int MidiPlugin::unloadFile()
{

    return 0;
}

//------------------------------------------------------------------------------
void MidiPlugin::key(int type, int keySym, int mod)
{
    if (type == osgGA::GUIEventAdapter::KEYDOWN)
    {
        //fprintf(stdout,"--- coVRKey called (KeyPress, keySym=%d, mod=%d)\n",
        //	keySym,mod);
        return;
        //}else{
        //fprintf(stdout,"--- coVRKey called (KeyRelease, keySym=%d)\n",keySym);
    }

    switch (keySym)
    {

    case ('r'): /* r: reset animation */
        break;

    }
}

//-----------------------------------------------------------------------------
MidiPlugin::MidiPlugin()
{
    plugin = this;
    MIDITab = NULL;
}
osg::Geode *MidiPlugin::createGeometry(int i)
{
    osg::Geode *geode;

    osg::Sphere *mySphere = new osg::Sphere(osg::Vec3(0, 0, 0), 200.0);
    osg::ShapeDrawable *mySphereDrawable = new osg::ShapeDrawable(mySphere, hint.get());
        mySphereDrawable->setColor(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    geode = new osg::Geode();
    geode->addDrawable(mySphereDrawable);
    geode->setStateSet(shadedStateSet.get());
    return geode;
}

bool MidiPlugin::init()
{
    coVRFileManager::instance()->registerFileHandler(&handlers[0]);
    coVRFileManager::instance()->registerFileHandler(&handlers[1]);
    //----------------------------------------------------------------------------

    MIDIRoot = new osg::Group;
    MIDIRoot->setName("MIDIRoot");
    cover->getObjectsRoot()->addChild(MIDIRoot.get());
    
    globalmtl = new osg::Material;
    globalmtl->ref();
    globalmtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    globalmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0));
    globalmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    globalmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    globalmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
    globalmtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

    shadedStateSet = new osg::StateSet();
    shadedStateSet->ref();
    shadedStateSet->setAttributeAndModes(globalmtl.get(), osg::StateAttribute::ON);
    shadedStateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    shadedStateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
    //for transparency, we need a transparent bin
    shadedStateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    shadedStateSet->setNestRenderBins(false);
    shadeModel = new osg::ShadeModel;
    shadeModel->setMode(osg::ShadeModel::SMOOTH);
    shadedStateSet->setAttributeAndModes(shadeModel, osg::StateAttribute::ON);

    noteGeometries.resize(80);
    
    hint = new osg::TessellationHints();
    hint->setDetailRatio(0.1);
    
    for(int i=0;i<80;i++)
       noteGeometries[i]=NULL;
    noteGeometries[27] = createGeometry(27);
    noteGeometries[28] = createGeometry(28);
    
    noteGeometries[29] = createGeometry(29);
    noteGeometries[30] = createGeometry(30);
    
    noteGeometries[31] = createGeometry(31);
    noteGeometries[32] = createGeometry(32);
    
    noteGeometries[36] = createGeometry(36);
    
    noteGeometries[38] = createGeometry(38);
    
    noteGeometries[40] = createGeometry(40);
    
    noteGeometries[41] = createGeometry(41);
    noteGeometries[39] = createGeometry(39);
    
    noteGeometries[43] = createGeometry(43);
    noteGeometries[58] = createGeometry(58);
    
    noteGeometries[46] = createGeometry(46);
    noteGeometries[44] = createGeometry(44);
    
    noteGeometries[48] = createGeometry(48);
    
    noteGeometries[49] = createGeometry(49);
    noteGeometries[55] = createGeometry(55);
    
    noteGeometries[51] = createGeometry(51);
    noteGeometries[59] = createGeometry(59);
    
    noteGeometries[53] = createGeometry(53);
    
    noteGeometries[57] = createGeometry(57);
    noteGeometries[52] = createGeometry(52);

    MIDItab_create();
    return true;
}

//------------------------------------------------------------------------------
MidiPlugin::~MidiPlugin()
{
}

bool MidiPlugin::destroy()
{

    coVRFileManager::instance()->unregisterFileHandler(&handlers[0]);
    coVRFileManager::instance()->unregisterFileHandler(&handlers[1]);

    cover->getObjectsRoot()->removeChild(MIDIRoot.get());

    MIDItab_delete();

    return true;
}

//------------------------------------------------------------------------------
void MidiPlugin::preFrame()
{
}

//------------------------------------------------------------------------------
void MidiPlugin::postFrame()
{
    // we do not need to care about animation (auto or step) here,
    // because it's in the main program
}

//----------------------------------------------------------------------------
/*
osg::Geode *MidiPlugin::createBallGeometry(int ballId)
{

    //------------------------
    // ! function not used
    //------------------------
    //fprintf(outfile,
    // "--- MidiPlugin::createBallGeometry (%d) called\n",ballId);

    osg::Geode *geode;
    float color3[3];
    int col;

    osg::Sphere *mySphere = new osg::Sphere(osg::Vec3(0, 0, 0), (float)geo.ballsradius[ballId]);
    osg::ShapeDrawable *mySphereDrawable = new osg::ShapeDrawable(mySphere, hint.get());
    col = geo.ballscolor[ballId];
    if (col < 0)
    {
        ballcolor(color3, str.balldyncolor[str.act_step][ballId]);
        mySphereDrawable->setColor(osg::Vec4(color3[0], color3[1], color3[2], 1.0f));
    }
    else
    {
        mySphereDrawable->setColor(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    }
    geode = new osg::Geode();
    geode->addDrawable(mySphereDrawable);
    geode->setStateSet(shadedStateSet.get());

    return (geode);
}*/
//--------------------------------------------------------------------
void MidiPlugin::setTimestep(int t)
{
}
//--------------------------------------------------------------------
void MidiPlugin::MIDItab_create(void)
{
    MIDITab = new coTUITab("MIDI", coVRTui::instance()->mainFolder->getID());
    MIDITab->setPos(0, 0);

    infoLabel = new coTUILabel("MIDI Version 1.0", MIDITab->getID());
    infoLabel->setPos(0, 0);
}

//--------------------------------------------------------------------
void MidiPlugin::MIDItab_delete(void)
{
    if (MIDITab)
    {
        delete infoLabel;

        delete MIDITab;
    }
}

//--------------------------------------------------------------------

COVERPLUGIN(MidiPlugin)


Track::Track(int tn)
{
}
Track::~Track()
{
}
void Track::update()
{
}
void Track::setVisible(bool state)
{
}
