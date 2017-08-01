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

void playerUnavailableCB()
{
    MidiPlugin::plugin->player = NULL;
}


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
      tempo = 60.0;

   int numTracks = midifile.getTrackCount();
   cout << "TPQ: " << midifile.getTicksPerQuarterNote() << endl;
      char text[1000];
      snprintf(text,1000,"numTracks = %d",numTracks);
      infoLabel->setLabel(text);
   if (numTracks > 1) {
   for (int i=0; i<midifile.getNumEvents(0); i++) {

      // check for tempo indication
      if (midifile[0][i][0] == 0xff &&
                 midifile[0][i][1] == 0x51) {
         setTempo(i);
      }
   }
   fprintf(stderr,"Tempo %f\n",tempo);
   
   
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
    player = NULL;

    MIDITab = NULL;
    startTime = 0;
}
osg::Geode *MidiPlugin::createGeometry(int i)
{
    osg::Geode *geode;

    osg::Sphere *mySphere = new osg::Sphere(osg::Vec3(0, 0, 0), 20.0);
    osg::ShapeDrawable *mySphereDrawable = new osg::ShapeDrawable(mySphere, hint.get());
    if(noteInfos[i]!=NULL)
    {
        mySphereDrawable->setColor(noteInfos[i]->color);
    }
    geode = new osg::Geode();
    geode->addDrawable(mySphereDrawable);
    geode->setStateSet(shadedStateSet.get());
    return geode;
}

NoteInfo::NoteInfo(int nN)
{
    noteNumber = nN;
    MidiPlugin::plugin->nIs.push_back(this);
}
void NoteInfo::createGeom()
{
    geometry = MidiPlugin::plugin->createGeometry(noteNumber);
}

bool MidiPlugin::init()
{
currentTrack = 0;
    coVRFileManager::instance()->registerFileHandler(&handlers[0]);
    coVRFileManager::instance()->registerFileHandler(&handlers[1]);
    //----------------------------------------------------------------------------
  /*  if (player == NULL)
    {
        player = cover->usePlayer(playerUnavailableCB);
        if (player == NULL)
        {
            cover->unusePlayer(playerUnavailableCB);
            cover->addPlugin("Vrml97");
            player = cover->usePlayer(playerUnavailableCB);
            if (player == NULL)
            {
                cerr << "sorry, no VRML, no Sound support " << endl;
            }
        }
    }*/

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

    noteInfos.resize(180);
    
    hint = new osg::TessellationHints();
    hint->setDetailRatio(1.0);
    
    for(int i=0;i<180;i++)
       noteInfos[i]=NULL;
   /* noteInfos[27] = new NoteInfo(27);
    noteInfos[27]->color = osg::Vec4(1,0,1,1);
    noteInfos[28] = new NoteInfo(28);
    noteInfos[28]->color = osg::Vec4(1,0,0.5,1);
    
    noteInfos[29] = new NoteInfo(29);
    noteInfos[29]->color = osg::Vec4(1,0,0,1);
    noteInfos[30] = new NoteInfo(30);
    noteInfos[30]->color = osg::Vec4(1,0,0,1);
    
    noteInfos[31] = new NoteInfo(31);
    noteInfos[31]->color = osg::Vec4(1,0.2,0.2,1);
    noteInfos[32] = new NoteInfo(32);
    noteInfos[32]->color = osg::Vec4(1,0.2,0.2,1);
    
    noteInfos[36] = new NoteInfo(36);
    noteInfos[36]->color = osg::Vec4(1,1,0,1);
    
    noteInfos[38] = new NoteInfo(38);
    noteInfos[38]->color = osg::Vec4(1,1,0.2,1);
    
    noteInfos[40] = new NoteInfo(40);
    noteInfos[40]->color = osg::Vec4(1,1,0.4,1);
    
    noteInfos[41] = new NoteInfo(41);
    noteInfos[41]->color = osg::Vec4(0.4,0.4,1,1);
    noteInfos[39] = new NoteInfo(39);
    noteInfos[39]->color = osg::Vec4(0.4,0.4,1,1);
    
    noteInfos[43] = new NoteInfo(43);
    noteInfos[43]->color = osg::Vec4(0.2,0.2,1,1);
    noteInfos[58] = new NoteInfo(58);
    noteInfos[58]->color = osg::Vec4(0.2,0.2,1,1);
    
    noteInfos[46] = new NoteInfo(46);
    noteInfos[46]->color = osg::Vec4(0,0,1,1);
    noteInfos[44] = new NoteInfo(44);
    noteInfos[44]->color = osg::Vec4(0,0,1,1);
    
    noteInfos[48] = new NoteInfo(48);
    noteInfos[48]->color = osg::Vec4(0,0,0.8,1);
    
    noteInfos[49] = new NoteInfo(49);
    noteInfos[49]->color = osg::Vec4(1,1,0.3,1);
    noteInfos[55] = new NoteInfo(55);
    noteInfos[55]->color = osg::Vec4(1,1,0.3,1);
    
    noteInfos[51] = new NoteInfo(51);
    noteInfos[51]->color = osg::Vec4(1,1,0.1,1);
    noteInfos[59] = new NoteInfo(59);
    noteInfos[59]->color = osg::Vec4(1,1,0.1,1);
    
    noteInfos[53] = new NoteInfo(53);
    noteInfos[53]->color = osg::Vec4(0.8,0.8,0.0,1);
    
    noteInfos[57] = new NoteInfo(57);
    noteInfos[57]->color = osg::Vec4(0.9,0.9,0.9,1);
    noteInfos[52] = new NoteInfo(52);
    noteInfos[52]->color = osg::Vec4(0.9,0.9,0.9,1);*/
    
    
    noteInfos[36] = new NoteInfo(36); //kick
    noteInfos[36]->color = osg::Vec4(1,0,1,1);
    noteInfos[43] = new NoteInfo(43);//tom3
    noteInfos[43]->color = osg::Vec4(1,0,0.5,1);
    noteInfos[45] = new NoteInfo(45);//tom2
    noteInfos[45]->color = osg::Vec4(1,0,0,1);
    noteInfos[48] = new NoteInfo(48);//tom1
    noteInfos[48]->color = osg::Vec4(1,0,0,1);
    noteInfos[38] = new NoteInfo(38); //snare
    noteInfos[38]->color = osg::Vec4(1,0.2,0.2,1);
    noteInfos[46] = new NoteInfo(46);//hi-Hat open
    noteInfos[46]->color = osg::Vec4(1,0.2,0.2,1);
    noteInfos[42] = new NoteInfo(42);//hi-Hat closed
    noteInfos[42]->color = osg::Vec4(1,1,0,1);
    noteInfos[44] = new NoteInfo(44);//hi-Hat Stomp
    noteInfos[44]->color = osg::Vec4(1,1,0.2,1);
    noteInfos[49] = new NoteInfo(49);//crash
    noteInfos[49]->color = osg::Vec4(1,1,0.4,1);
    
    noteInfos[51] = new NoteInfo(51);//ride
    noteInfos[51]->color = osg::Vec4(0.4,0.4,1,1);
   /* noteInfos[39] = new NoteInfo(39);
    noteInfos[39]->color = osg::Vec4(0.4,0.4,1,1);
    
    noteInfos[43] = new NoteInfo(43);
    noteInfos[43]->color = osg::Vec4(0.2,0.2,1,1);
    noteInfos[58] = new NoteInfo(58);
    noteInfos[58]->color = osg::Vec4(0.2,0.2,1,1);
    
    noteInfos[46] = new NoteInfo(46);
    noteInfos[46]->color = osg::Vec4(0,0,1,1);
    noteInfos[44] = new NoteInfo(44);
    noteInfos[44]->color = osg::Vec4(0,0,1,1);
    
    noteInfos[48] = new NoteInfo(48);
    noteInfos[48]->color = osg::Vec4(0,0,0.8,1);
    
    noteInfos[49] = new NoteInfo(49);
    noteInfos[49]->color = osg::Vec4(1,1,0.3,1);
    noteInfos[55] = new NoteInfo(55);
    noteInfos[55]->color = osg::Vec4(1,1,0.3,1);
    
    noteInfos[51] = new NoteInfo(51);
    noteInfos[51]->color = osg::Vec4(1,1,0.1,1);
    noteInfos[59] = new NoteInfo(59);
    noteInfos[59]->color = osg::Vec4(1,1,0.1,1);
    
    noteInfos[53] = new NoteInfo(53);
    noteInfos[53]->color = osg::Vec4(0.8,0.8,0.0,1);
    
    noteInfos[57] = new NoteInfo(57);
    noteInfos[57]->color = osg::Vec4(0.9,0.9,0.9,1);
    noteInfos[52] = new NoteInfo(52);
    noteInfos[52]->color = osg::Vec4(0.9,0.9,0.9,1);*/

    
    for(int i=0;i<nIs.size();i++)
    {
    nIs[i]->createGeom();
        float angle = ((float)i/nIs.size())*2.0*M_PI/4.0*4.0;
        //float radius = 300.0+(float)i/nIs.size()*800.0;
        float radius = 800.0;
        nIs[i]->initialPosition.set(sin(angle)*radius,cos(angle)*radius,0);
        nIs[i]->initialVelocity.set(sin(angle)*100.0,cos(angle)*100.0,1000);
    }

    midi1fd = -1;
    if(coVRMSController::instance()->isMaster())
    {
        midi1fd = open("/dev/midi1",O_RDONLY | O_NONBLOCK);
        fprintf(stderr,"open /dev/midi1 %d",midi1fd);
    }
    lTrack = NULL;
    lTrack = new Track(tracks.size());
    tracks.push_back(lTrack);
    MIDItab_create();
    return true;
}

//------------------------------------------------------------------------------
MidiPlugin::~MidiPlugin()
{
delete lTrack;
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
    //fprintf(stderr,"tracks %d\n",tracks.size());
    if(tracks.size() > 0)
    {
        if(startTime == 0.0)
        {
            tracks[currentTrack]->reset();
            startTime = cover->frameTime();
            tracks[currentTrack]->setVisible(true);
        }
        tracks[currentTrack]->update();
    }
}

void MidiPlugin::tabletEvent(coTUIElement *elem)
{
    if(elem == trackNumber)
    {
    
            tracks[currentTrack]->setVisible(false);
	    
        currentTrack = trackNumber->getValue()-1;
	if(currentTrack>=0)
	{
            tracks[currentTrack]->reset();
            startTime = cover->frameTime();
            tracks[currentTrack]->setVisible(true);
	    }
	    else
	    {
	    currentTrack=0;
	    }
    }
}

void MidiPlugin::setTempo(int index) {
   double newtempo = 0.0;
   static int count = 0;
   count++;

   MidiEvent& mididata = midifile[0][index];

   int microseconds = 0;
   microseconds = microseconds | (mididata[3] << 16);
   microseconds = microseconds | (mididata[4] << 8);
   microseconds = microseconds | (mididata[5] << 0);

   newtempo = 60.0 / microseconds * 1000000.0;
   if (count <= 1) {
      tempo = newtempo;
   } else if (tempo != newtempo) {
      cout << "; WARNING: change of tempo from " << tempo
           << " to " << newtempo << " ignored" << endl;
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
    
    trackNumber = new coTUIEditIntField("trackNumber", MIDITab->getID());
    trackNumber->setPos(0, 1);
    trackNumber->setValue(currentTrack);
    trackNumber->setEventListener(this);
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
    char soundName[200];
    snprintf(soundName,200,"RENDERS/S%d.wav",tn);
    //trackAudio = new vrml::Audio(soundName);
    trackAudio = NULL;
    trackSource = NULL;
    if (trackAudio!=NULL && MidiPlugin::plugin->player)
    {
        trackSource = MidiPlugin::plugin->player->newSource(trackAudio);
        if (trackSource)
        {
            trackSource->setLoop(false);
	    trackSource->setPitch(1);
            trackSource->stop();
            trackSource->setIntensity(1.0);
            trackSource->setSpatialize(false);
            trackSource->setVelocity(0,0,0);
	    
        }
    }
}
Track::~Track()
{
}
void Track::reset()
{
    eventNumber = 0;
    if(trackSource!=NULL)
    {
            trackSource->play();
    }
    for(std::list<Note *>::iterator it = notes.begin(); it != notes.end();it++)
    {
    delete *it;
    }
    notes.clear();
}
void Track::update()
{
    double speed = MidiPlugin::plugin->midifile.getTicksPerQuarterNote();
    double time = cover->frameTime() - MidiPlugin::plugin->startTime;
    MidiEvent me;
    if(this == MidiPlugin::plugin->lTrack)
    {
        char buf[1000];
        int numRead = 0;
        if(MidiPlugin::plugin->midi1fd>0)
        {
            numRead = read(MidiPlugin::plugin->midi1fd,buf,2);
        }
	if(numRead > 0)
	{
	    if(buf[0]!=-2)
	    {
	        if(buf[0]==-112 || buf[0]==-103)
		{
		    int key = buf[1];
                    me.setP0(buf[0]);
                    me.setP1(buf[1]);
                    int numRead = read(MidiPlugin::plugin->midi1fd,buf,2);
		    int value = buf[0];
                    me.setP2(buf[0]);
		    if(value > 0)
                    {
                        // key press
			me.setKeyNumber(key);
                        me.setVelocity(value);
		    }
		    else
		    {
		        // key release
		    }
		}
		else
		{
		    fprintf(stderr,"%d %d\n",(int)buf[0],(int)buf[1]);
		}
	    }
	}
        if (coVRMSController::instance()->isMaster())
        {
	    char buf[3];
	    buf[0] = me.getP0();
	    buf[1] = me.getP1();
	    buf[2] = me.getP2();
            coVRMSController::instance()->sendSlaves((char *)buf, 3);
	    
        }
        else
        {
	    char buf[3];
            coVRMSController::instance()->readMaster((char *)buf, 3);
                    me.setP0(buf[0]);
                    me.setP1(buf[1]);
                    me.setP2(buf[2]);
        }
        if(me.isNote() && me.getVelocity()>0)
        {
            notes.push_back(new Note(me,this));
        }
    }
    else
    {
        if(eventNumber < MidiPlugin::plugin->midifile[trackNumber].size())
        {
            me = MidiPlugin::plugin->midifile[trackNumber][eventNumber];
	    }
    while((eventNumber < MidiPlugin::plugin->midifile[trackNumber].size()) && ((me.tick*60.0/MidiPlugin::plugin->tempo/speed) < time))
    {
            me = MidiPlugin::plugin->midifile[trackNumber][eventNumber];
            me.getDurationInSeconds();
            if(me.isNoteOn())
            {
                //fprintf(stderr,"new note %d\n",me.getKeyNumber());
                if(MidiPlugin::plugin->noteInfos[me.getKeyNumber()]!=NULL)
                {
                notes.push_back(new Note(me,this));
                }
                else
                {

                    //fprintf(stderr,"unknown note %d\n",me.getKeyNumber());
                }
            }
            eventNumber++;
    } 
    }
    
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
    if(trackSource!=NULL)
    {
            trackSource->play();
    }
    }
    else
    {
        MidiPlugin::plugin->MIDIRoot->removeChild(TrackRoot);
    if(trackSource!=NULL)
    {
            trackSource->stop();
    }
    }
}

Note::Note(MidiEvent &me, Track *t)
{
    event = me;
    track = t;
    NoteInfo *ni = MidiPlugin::plugin->noteInfos[me.getKeyNumber()];
    transform = new osg::MatrixTransform();
    float s = event.getVelocity()/10.0;
    if(ni==NULL)
    {
        fprintf(stderr,"no NoteInfo for Key %d\n",me.getKeyNumber());
        ni = MidiPlugin::plugin->noteInfos[27];
        event.setKeyNumber(27);
    }
    transform->setMatrix(osg::Matrix::scale(s,s,s) * osg::Matrix::translate(ni->initialPosition));
    transform->addChild(ni->geometry);
    velo = ni->initialVelocity*event.getVelocity()/100.0;
    velo[2] = (event.getVelocity()-32) *20;
    t->TrackRoot->addChild(transform.get());
    
}
Note::~Note()
{
    track->TrackRoot->removeChild(transform.get());
}
void Note::integrate(double time)
{
    osg::Matrix nm=transform->getMatrix();
    osg::Vec3 pos = nm.getTrans();
    osg::Vec3 spiral;
    spiral[0] = pos[1];
    spiral[1] = -pos[0];
    spiral[2] = 0.0;
    spiral *= 0.1;
    osg::Vec3 a = osg::Vec3(0,0,-300.0);
    a = pos;
    a.normalize();
    a *= 300;
    velo = velo + a*time;
    pos += (velo + spiral) * time;
    nm.setTrans(pos);
    transform->setMatrix(nm);
}
    
