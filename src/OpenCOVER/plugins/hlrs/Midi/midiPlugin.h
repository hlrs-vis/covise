/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MIDI_PLUGIN_H
#define _MIDI_PLUGIN_H


#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <cover/coTabletUI.h>
#include <OpenVRUI/coPopupHandle.h>
#include <osg/Material>
#include <osg/StateSet>
#include <osg/Array>
#include <PluginUtil/coSphere.h>
#include <cover/coVRPlugin.h>
#include <MidiFile.h>
#include <osg/ShapeDrawable>
#include <osg/ShadeModel>
#include <vrml97/vrml/Player.h>

using namespace opencover;
using namespace covise;

namespace covise
{
class coSubMenuItem;
class coRowMenu;
class coCheckboxMenuItem;
class coCheckboxGroup;
class coButtonMenuItem;
class coSliderMenuItem;
}
class Track;
class Note
{
public:
    Note(MidiEvent &me, Track *t);
    ~Note();
    Track *track;
    void integrate(double time);
    osg::ref_ptr<osg::MatrixTransform> transform;
    osg::Vec3 velo;
    MidiEvent event;
};
class Track
{
public:
    int eventNumber;
     Track(int tn);
     ~Track();
     std::list<Note *> notes;
     //std::list<Note *>::iterator lastNoteIt;
     
     osg::ref_ptr<osg::Group> TrackRoot;
     void update();
     void reset();
     void setVisible(bool state);
     int trackNumber;
     void addNote(Note*);
     vrml::Player::Source *trackSource;
     vrml::Audio *trackAudio;
     osg::Geode *createLinesGeometry();
     
     osg::ref_ptr<osg::Geode> geometryLines;
     osg::Vec3Array *lineVert = new osg::Vec3Array;
     osg::Vec4Array *lineColor = new osg::Vec4Array;
     osg::DrawArrayLengths *linePrimitives;
     private:
     int lastNum;
     int lastPrimitive;
    double oldTime=0.0;
};

class NoteInfo
{
    public:
        NoteInfo(int nN);
        ~NoteInfo();
	void createGeom();
    osg::ref_ptr<osg::Geode> geometry;
    osg::Vec3 initialPosition;
    osg::Vec3 initialVelocity;
    osg::Vec4 color;
    int noteNumber;
};

class MidiPlugin : public coVRPlugin, public coTUIListener
{
private:
    



public:
    
    double  tempo;
    //tabletUI
    coTUITab *MIDITab;
    coTUILabel *infoLabel;
    coTUIEditIntField *trackNumber;
    coTUIButton *reset;
    std::vector<Track *> tracks;
    std::vector<NoteInfo *> noteInfos;
    static MidiPlugin *plugin;
    vrml::Player *player;
    //scenegraph
    osg::ref_ptr<osg::Group> MIDIRoot;
    std::vector<NoteInfo *> nIs;
    MidiFile midifile;
    double startTime;
    int currentTrack;

    static int unloadMidi(const char *filename, const char *);
    static int loadMidi(const char *filename, osg::Group *parent, const char *);
    int loadFile(const char *filename, osg::Group *parent);
    int unloadFile();
    void setTempo(int index);

    void setTimestep(int t);
    int midi1fd;
    Track *lTrack;

    // constructor
    MidiPlugin();

    // destructor
    virtual ~MidiPlugin();
    osg::Geode *createGeometry(int i);
    osg::ref_ptr<osg::TessellationHints> hint;
    osg::ref_ptr<osg::StateSet> shadedStateSet;
    osg::ref_ptr<osg::StateSet> lineStateSet;
    
    osg::ref_ptr<osg::ShadeModel> shadeModel;
    osg::ref_ptr<osg::Material> globalmtl;
    

    bool init();
    bool destroy();

    // loop
    bool update();
    void preFrame();
    void postFrame();

    void key(int type, int keySym, int mod);

    //tabletUI
    void MIDItab_create();
    void MIDItab_delete();
    
    //tabletUI
    void tabletEvent(coTUIElement *);

};
#endif
