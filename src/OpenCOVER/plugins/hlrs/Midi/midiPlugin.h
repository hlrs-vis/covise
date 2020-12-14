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
#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Label.h>
#include <cover/ui/Slider.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Owner.h>
#include <net/udpMessage.h>
#include <SDL_audio.h>
#include <SDL.h>
#include <fftw3.h>

#define BINSIZE 1024
const int numChannels = 16;
const int numKeys = 127;

#include <UDPComm.h>
namespace smf
{
    // just for compiling with MidiFile inside and outside of smf namespace
}

namespace covise
{
    class coSubMenuItem;
    class coRowMenu;
    class coCheckboxMenuItem;
    class coCheckboxGroup;
    class coButtonMenuItem;
    class coSliderMenuItem;
}

class AudioInStream;

using namespace opencover;
using namespace covise;
using namespace smf;

class UDPMidiMessage
{
public:
	UDPMidiMessage(int t, unsigned char s, unsigned char p1, unsigned char p2) { mTime = t; mStatus = s; mParam1 = p1; mParam2 = p2; };
	int mTime;
	unsigned char mStatus;
	unsigned char mParam1;
	unsigned char mParam2;
};

class controllerMapping
{
	controllerMapping();
	~controllerMapping();

	enum controlTypes
	{
		undefinedController = -1,
		frequencyRadius1,
		frequencyRadius2,
		amplitudeRadius1,
		amplitudeRadius2,
		objectScale,
		audioSpeed,
		frequencyFactor,
		amplitudeFactor,
		speedFactor,
		acceleration,
		radialAcceleration,
		spiralSpeed,
		numControlTypes
	};
	const char* controllerName[numControlTypes + 2] = {
		"UNDEFINED",
		"frequencyRadius1",
		"frequencyRadius2",
		"amplitudeRadius1",
		"amplitudeRadius2",
		"objectScale",
		"audioSpeed",
		"frequencyFactor",
		"amplitudeFactor",
		"speedFactor",
		"acceleration",
		"radialAcceleration",
		"spiralSpeed",
		"numControllerTypes",
	};
	struct controllerInfo
	{
		float min;
		float max;
		controlTypes type;
	};
	struct controlInfo
	{
		float min;
		float max;
	};
	controlInfo controlInfos[numControlTypes + 2];

	controllerInfo controlID[numKeys][numChannels];
};
class NoteInfo
{
public:
	NoteInfo(int nN);
	~NoteInfo();
	void createGeom();
	osg::ref_ptr<osg::Node> geometry;
	osg::Vec3 initialPosition;
	osg::Vec3 initialVelocity;
	osg::Vec4 color;
	std::string modelName;
	float modelScale = 1.0;
	int noteNumber;
};
class MidiInstrument
{
public:
	MidiInstrument(std::string name, int id);
	~MidiInstrument();
	std::string type;
	int channel = 0;
	std::vector<NoteInfo*> noteInfos;
};
class MidiDevice
{
public:
	MidiDevice(std::string name, int id);
	~MidiDevice();
	std::string name;
	int ID;
	int instrumentNumber;
	MidiInstrument* instrument=nullptr;
};

class WaveSurface
{
public:
	WaveSurface(osg::Group *parent, AudioInStream* stream, int width);
	~WaveSurface();

	enum surfaceType {SurfacePlane,SurfaceCylinder,SurfaceSphere};

	virtual bool update();
	void setType(surfaceType st);
	float radius1;
	float radius2;
	float yStep;
        float amplitudeFactor=-10.0;
        float frequencyFactor=-1;
protected:
	surfaceType st;
	void moveOneLine();
	void createNormals();
	int depth = 300;
	int width = 50;
	osg::ref_ptr<osg::Geode> geode;
	osg::Vec3Array *vert;
	osg::Vec3Array *normals;
	osg::Vec2Array *texCoord;
	AudioInStream* stream;
	static osg::ref_ptr <osg::Material>globalDefaultMaterial;
};


class FrequencySurface : public WaveSurface
{
public:
	FrequencySurface(osg::Group *parent, AudioInStream* stream);
	~FrequencySurface();

	virtual bool update();
private:

	double *lastMagnitudes;
};
class AmplitudeSurface : public WaveSurface
{
public:
	AmplitudeSurface(osg::Group *parent, AudioInStream* stream);
	~AmplitudeSurface();

	virtual bool update();
private:

	double *lastAmplitudes;
};

class AudioInStream
{
public:
	AudioInStream(std::string deviceName="");
	~AudioInStream();
	void update();
	double *ddata;
	double *magnitudes;
	int inputSize;
	int outputSize;
private:
	SDL_AudioDeviceID dev;
	void readData(Uint8 * stream, int len);
	static void readData(void *userdata, Uint8 * stream,int len);
	int gBufferBytePosition=0;
	int bytesPerSample;
	int gBufferByteSize;
	int gBufferByteMaxPosition;
	Uint8 *gRecordingBuffer;
	SDL_AudioSpec want, have;

	fftw_complex *ifft_result;
	fftw_plan plan;

};
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
	int vertNum;
	float noteScale;
	void setInactive(bool state);
	bool inactive = true;
	osg::Vec3 spin;
	osg::Vec3 rot;
};
class Track
{
public:

	enum deviceType {Drum,Keyboard,NumTypes};
    int eventNumber;
    Track(int tn,bool life=false);
    ~Track();
    std::list<Note *> notes;
    //std::list<Note *>::iterator lastNoteIt;

	void clearStore();
    osg::ref_ptr<osg::MatrixTransform> TrackRoot;
    void update();
    void reset();
    void setVisible(bool state);
    int trackNumber;
    void addNote(Note*);
	void endNote(MidiEvent& me);
	void setRotation(osg::Vec3 &rotationSpeed);
    vrml::Player::Source *trackSource;
    vrml::Audio *trackAudio;
	MidiInstrument *instrument=nullptr;
    osg::Geode *createLinesGeometry();

    osg::ref_ptr<osg::Geode> geometryLines;
    osg::Vec3Array *lineVert = new osg::Vec3Array;
    osg::Vec4Array *lineColor = new osg::Vec4Array;
    osg::DrawArrayLengths *linePrimitives;
	void handleEvent(MidiEvent& me);
	void store();
private:
    bool life;
    int lastNum;
    int lastPrimitive;
    double oldTime = 0.0;
    int streamNum;
    enum deviceType devType=Track::Drum;
	osg::Vec3 rotationSpeed;
};

class MidiPlugin : public coVRPlugin, public coTUIListener, public ui::Owner
{
private:


	int gRecordingDeviceCount;
	std::list<AudioInStream *>audioStreams;
	std::list<WaveSurface *>waveSurfaces;

public:

    UDPComm *udp= nullptr;
    static const size_t NUMMidiStreams = 16;
    double  tempo;
	std::vector<std::unique_ptr<MidiInstrument>> instruments;
	std::vector<std::unique_ptr<MidiDevice>> devices;
    std::vector<Track *> tracks;
    std::list<MidiEvent> eventqueue[NUMMidiStreams];
    static MidiPlugin *instance();
    vrml::Player *player;
    //scenegraph
    osg::ref_ptr<osg::Group> MIDIRoot;
    osg::ref_ptr<osg::MatrixTransform> MIDITrans[NUMMidiStreams];
    MidiFile midifile;
    double startTime;
	float speedFactor = 1.0;
    int currentTrack;
	void handleController(MidiEvent& me);
	void clearStore();
    static int unloadMidi(const char *filename, const char *);
    static int loadMidi(const char *filename, osg::Group *parent, const char *);
    int loadFile(const char *filename, osg::Group *parent);
    int unloadFile();
    void setTempo(int index);

    void setTimestep(int t);
    Track *lTrack[NUMMidiStreams];
	std::list<Track *>storedTracks;

    void addEvent(MidiEvent &me, int MidiStream);

    // constructor
    MidiPlugin();

    // destructor
    virtual ~MidiPlugin();
    osg::Node *createGeometry(int i);
    osg::ref_ptr<osg::TessellationHints> hint;
    osg::ref_ptr<osg::StateSet> shadedStateSet;
    osg::ref_ptr<osg::StateSet> lineStateSet;

    osg::ref_ptr<osg::ShadeModel> shadeModel;
    osg::ref_ptr<osg::Material> globalmtl;

#ifdef WIN32
    HMIDIOUT hMidiDeviceOut = NULL;
    HMIDIIN hMidiDevice[NUMMidiStreams];
#else
#endif
    int midifd[NUMMidiStreams];
    int midiOutfd;

    bool openMidiIn(int streamNum, int device);
    bool openMidiOut(int device);

    bool init();
    bool destroy();

    // loop
    bool update();
    void preFrame();
    void postFrame();

	void UDPmessage(UdpMessage* msg);

    void key(int type, int keySym, int mod);

    void MIDItab_create();
    void MIDItab_delete();
    FrequencySurface *frequencySurface;
    AmplitudeSurface *amplitudeSurface;

    ui::Menu *MIDITab = nullptr;
	ui::Button* reset = nullptr;
	ui::Button* clearStoreButton = nullptr;
	
    ui::Slider *radius1Slider = nullptr;
    ui::Slider *radius2Slider = nullptr;
    ui::Slider *yStepSlider = nullptr;
    ui::Slider *amplitudeFactor = nullptr;
    ui::Slider *frequencyFactor = nullptr;
    ui::Slider *accelSlider = nullptr;
    ui::Slider *raccelSlider = nullptr;
    ui::Slider *spiralSpeedSlider = nullptr;
	ui::Slider* sphereScaleSlider = nullptr;
	
    ui::EditField *trackNumber = nullptr;
    ui::SelectionList *inputDevice[NUMMidiStreams];
    ui::SelectionList *outputDevice = nullptr;
    ui::Label *infoLabel = nullptr;
    float acceleration=-300;
    float rAcceleration=0.2;
    float spiralSpeed=0.1;
	float sphereScale = 1.0;
private:

    static MidiPlugin *plugin;


};
#endif
