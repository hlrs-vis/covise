/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CHAT_PLUGIN_H
#define _CHAT_PLUGIN_H

#include <vrb/client/VRBClient.h>

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <cover/coTabletUI.h>
#include <OpenVRUI/coPopupHandle.h>
#include <cover/coVRPlugin.h>
#include <vrml97/vrml/Player.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Label.h>
#include <cover/ui/Slider.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Owner.h>
#include <QIODevice>
#include <QtMultimedia/QAudioOutput>
#include <QtMultimedia/QAudioInput>
#include <QtMultimedia/QAudioFormat>
#include <speex/speex.h>

namespace covise
{
    class coSubMenuItem;
    class coRowMenu;
    class coCheckboxMenuItem;
    class coCheckboxGroup;
    class coButtonMenuItem;
    class coSliderMenuItem;
}


using namespace opencover;
using namespace covise;

class ChatPlugin : public coVRPlugin, public ui::Owner
{
private:


public:
	static const int frameSize = 160;
    static ChatPlugin *instance();

    // constructor
    ChatPlugin();

    // destructor
    virtual ~ChatPlugin();
    bool init();
    bool destroy();

    // loop
    bool update();
    void preFrame();
    void postFrame();
	virtual void UDPmessage(UdpMessage* msg);

    void key(int type, int keySym, int mod);

    void CHATtab_create();
    void CHATtab_delete();


    ui::Menu *CHATTab = nullptr;
    ui::Button *reset = nullptr;
    ui::Slider *recVolSlider = nullptr;
    ui::Slider *outVolSlider = nullptr;
    ui::Label *infoLabel = nullptr;
	QIODevice* inStream;
	QIODevice* outStream;


	QByteArray audioBuffer;

	//Speex
	float inputData[frameSize];
	float outputData[frameSize];
	short outputDataShort[frameSize];
	char cbits[1024];
	/*Holds the state of the encoder*/
	void* encoderState;
	/*Holds the state of the decoder*/
	void* decoderState;
	/*Holds bits so they can be read and written to by the Speex routines*/
    SpeexBits bits;

private:
	QAudioInput* input;
	QAudioOutput* output;
    static ChatPlugin *plugin;
	void initSpeex();

	int encodeSpeex();

	int decodeSpeex(const void* data, int length);

};
#endif
