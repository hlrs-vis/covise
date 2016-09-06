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
    Note(int number, Track *t);
    ~Note();
    void integrate(double time);
    osg::ref_ptr<osg::MatrixTransform> transform;
    osg::Vec3 velo;
};
class Track
{
public:
    int eventNumber;
     Track(int tn);
     ~Track();
     std::list<Note *> notes;
     
     osg::ref_ptr<osg::Group> TrackRoot;
     void update();
     void reset();
     void setVisible(bool state);
     int trackNumber;
};

class NoteInfo
{
    public:
        NoteInfo(int nN);
        ~NoteInfo();
    osg::ref_ptr<osg::Geode> geometry;
    osg::Vec3 initialPosition;
    osg::Vec3 initialVelocity;
    int noteNumber;
};

class MidiPlugin : public coVRPlugin, public coTUIListener
{
private:
    



public:
    
    //tabletUI
    coTUITab *MIDITab;
    coTUILabel *infoLabel;
    std::vector<Track *> tracks;
    std::vector<NoteInfo *> noteInfos;
    static MidiPlugin *plugin;
    //scenegraph
    osg::ref_ptr<osg::Group> MIDIRoot;
    std::vector<NoteInfo *> nIs;
    MidiFile midifile;
    double startTime;

    static int unloadMidi(const char *filename, const char *);
    static int loadMidi(const char *filename, osg::Group *parent, const char *);
    int loadFile(const char *filename, osg::Group *parent);
    int unloadFile();

    void setTimestep(int t);

    // constructor
    MidiPlugin();

    // destructor
    virtual ~MidiPlugin();
    osg::Geode *createGeometry(int i);
    osg::ref_ptr<osg::TessellationHints> hint;
    osg::ref_ptr<osg::StateSet> shadedStateSet;
    osg::ref_ptr<osg::ShadeModel> shadeModel;
    osg::ref_ptr<osg::Material> globalmtl;
    

    bool init();
    bool destroy();

    // loop
    void preFrame();
    void postFrame();

    void key(int type, int keySym, int mod);

    //tabletUI
    void MIDItab_create();
    void MIDItab_delete();
    
    //tabletUI
    //void tabletEvent(coTUIElement *);

};
#endif
