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
    startTime = 0;
}
osg::Geode *MidiPlugin::createGeometry(int i)
{
    osg::Geode *geode;

    osg::Sphere *mySphere = new osg::Sphere(osg::Vec3(0, 0, 0), 20.0);
    osg::ShapeDrawable *mySphereDrawable = new osg::ShapeDrawable(mySphere, hint.get());
        mySphereDrawable->setColor(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    geode = new osg::Geode();
    geode->addDrawable(mySphereDrawable);
    geode->setStateSet(shadedStateSet.get());
    return geode;
}

NoteInfo::NoteInfo(int nN)
{
    noteNumber = nN;
    geometry = MidiPlugin::plugin->createGeometry(nN);
    MidiPlugin::plugin->nIs.push_back(this);
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

    noteInfos.resize(80);
    
    hint = new osg::TessellationHints();
    hint->setDetailRatio(1.0);
    
    for(int i=0;i<80;i++)
       noteInfos[i]=NULL;
    noteInfos[27] = new NoteInfo(27);
    noteInfos[28] = new NoteInfo(28);
    
    noteInfos[29] = new NoteInfo(29);
    noteInfos[30] = new NoteInfo(30);
    
    noteInfos[31] = new NoteInfo(31);
    noteInfos[32] = new NoteInfo(32);
    
    noteInfos[36] = new NoteInfo(36);
    
    noteInfos[38] = new NoteInfo(38);
    
    noteInfos[40] = new NoteInfo(40);
    
    noteInfos[41] = new NoteInfo(41);
    noteInfos[39] = new NoteInfo(39);
    
    noteInfos[43] = new NoteInfo(43);
    noteInfos[58] = new NoteInfo(58);
    
    noteInfos[46] = new NoteInfo(46);
    noteInfos[44] = new NoteInfo(44);
    
    noteInfos[48] = new NoteInfo(48);
    
    noteInfos[49] = new NoteInfo(49);
    noteInfos[55] = new NoteInfo(55);
    
    noteInfos[51] = new NoteInfo(51);
    noteInfos[59] = new NoteInfo(59);
    
    noteInfos[53] = new NoteInfo(53);
    
    noteInfos[57] = new NoteInfo(57);
    noteInfos[52] = new NoteInfo(52);

    
    for(int i=0;i<nIs.size();i++)
    {
        float angle = ((float)i/nIs.size())*2.0*M_PI/4.0*3.0;
        float radius = 300.0+(float)i/nIs.size()*800.0;
        nIs[i]->initialPosition.set(sin(angle)*radius,cos(angle)*radius,0);
        nIs[i]->initialVelocity.set(sin(angle)*10.0,cos(angle)*10.0,1000);
    }

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
    if(tracks.size() > 0)
    {
        if(startTime == 0.0)
        {
            tracks[30]->reset();
            startTime = cover->frameTime();
            tracks[30]->setVisible(true);
        }
        tracks[30]->update();
    }
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
    TrackRoot = new osg::Group();
    trackNumber = tn;
}
Track::~Track()
{
}
void Track::reset()
{
    eventNumber = 0;
}
void Track::update()
{
    double speed = MidiPlugin::plugin->midifile.getTicksPerQuarterNote();
    double time = cover->frameTime() - MidiPlugin::plugin->startTime;
    MidiEvent me;
    do {
        if(eventNumber < MidiPlugin::plugin->midifile[trackNumber].size())
        {
            me = MidiPlugin::plugin->midifile[trackNumber][eventNumber];
            me.getDurationInSeconds();
            if(me.isNoteOn())
            {
                //fprintf(stderr,"new note %d\n",me.getKeyNumber());
                if(MidiPlugin::plugin->noteInfos[me.getKeyNumber()]!=NULL)
                {
                notes.push_back(new Note(me.getKeyNumber(),this));
                }
                else
                {
                    
                //fprintf(stderr,"unknown note %d\n",me.getKeyNumber());
                }
            }
            eventNumber++;
        }
    } while(eventNumber < MidiPlugin::plugin->midifile[trackNumber].size() && (me.tick/speed) < time);
    for(std::list<Note *>::iterator it = notes.begin(); it != notes.end();it++)
    {
        (*it)->integrate(cover->frameDuration());
    }
}
void Track::setVisible(bool state)
{
    if(state)
    {
        MidiPlugin::plugin->MIDIRoot->addChild(TrackRoot);
    }
    else
    {
        MidiPlugin::plugin->MIDIRoot->removeChild(TrackRoot);
    }
}

Note::Note(int number, Track *t)
{
    NoteInfo *ni = MidiPlugin::plugin->noteInfos[number];
    transform = new osg::MatrixTransform();
    transform->setMatrix(osg::Matrix::translate(ni->initialPosition));
    transform->addChild(ni->geometry);
    t->TrackRoot->addChild(transform.get());
    velo = ni->initialVelocity; 
}
Note::~Note()
{
}
void Note::integrate(double time)
{
    osg::Matrix nm=transform->getMatrix();
    osg::Vec3 pos = nm.getTrans();
    osg::Vec3 a = osg::Vec3(0,0,-500.0);
    velo = velo + a*time;
    pos += (velo ) * time;
    nm.setTrans(pos);
    transform->setMatrix(nm);
}
    