/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef WIN32
#include <SDKDDKVer.h>
#include <winsock2.h>
#include <windows.h>
#include <direct.h>
#include <conio.h>
#include <mmsystem.h>
#endif
#include <stdio.h>
#include "cover/OpenCOVER.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <cover/coVRTui.h>
#include <config/CoviseConfig.h>
#include "chatPlugin.h"
#include <net/udpMessage.h>
#include <net/udp_message_types.h>

#include <speex/speex.h>


#ifndef WIN32
 // for chdir
#include <unistd.h>
#endif

#ifdef _WINDOWS
#include <direct.h>
#include <mmeapi.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

ChatPlugin *ChatPlugin::plugin = NULL;

//------------------------------------------------------------------------------
void ChatPlugin::key(int type, int keySym, int mod)
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
ChatPlugin::ChatPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("ChatPlugin", cover->ui)
{
	CHATTab = NULL;

}
ChatPlugin * ChatPlugin::instance()
{
	return plugin;
}

bool ChatPlugin::init()
{
	QAudioFormat format;
	format.setSampleRate(128000);
	format.setChannelCount(1);
	format.setSampleSize(16);
	format.setCodec("audio/pcm");
	format.setByteOrder(QAudioFormat::LittleEndian);
	format.setSampleType(QAudioFormat::UnSignedInt);

	//If format isn't supported find the nearest supported
	QAudioDeviceInfo info(QAudioDeviceInfo::defaultInputDevice());
	if (!info.isFormatSupported(format))
		format = info.nearestFormat(format);
	input = nullptr;
	output = nullptr;
	inStream = nullptr;
	outStream = nullptr;

	if (coVRMSController::instance()->isMaster())
	{
		int CHATPort = coCoviseConfig::getInt("InPort", "COVER.Plugin.CHAT", 0);
		int CHATPortOut = coCoviseConfig::getInt("OutPort", "COVER.Plugin.CHAT", 1);
		input = new QAudioInput(format);
		output = new QAudioOutput(format);
		inStream = input->start();
		outStream = output->start();
	}
	CHATtab_create();
	initSpeex();
	return true;
}

bool ChatPlugin::destroy()
{
	return true;
}

//------------------------------------------------------------------------------
ChatPlugin::~ChatPlugin()
{
	/*Destroy the encoder state*/
	speex_encoder_destroy(encoderState);
	/*Destroy the bit-packing struct*/
	speex_bits_destroy(&bits);
}

bool ChatPlugin::update()
{
	if(inStream)
	{
		covise::UdpMessage audioMessage;
		audioMessage.type = covise::udp_msg_type::AUDIO_STREAM;
		//while (inStream->bytesAvailable() > bufferSize)
		do{
			audioBuffer = inStream->read(frameSize);
			int encodedLength = encodeSpeex();
			audioMessage.data = DataHandle((char*)cbits, encodedLength,false);
			cover->sendVrbMessage(&audioMessage);
		} while (audioBuffer.length() > 0);
		audioMessage.data = DataHandle(nullptr, 0, false);;
	}
	return true;
}

//------------------------------------------------------------------------------
void ChatPlugin::preFrame()
{


}
//------------------------------------------------------------------------------
void ChatPlugin::postFrame()
{
	// we do not need to care about animation (auto or step) here,
	// because it's in the main program
}
void ChatPlugin::UDPmessage(covise::UdpMessage* msg)
{
	if (msg->type == covise::udp_msg_type::AUDIO_STREAM)
	{
		if(outStream)
		{
			int outSize = decodeSpeex(msg->data.data(), msg->data.length());
			outStream->write((const char*)outputDataShort, outSize);
		}
	}
}
//--------------------------------------------------------------------
void ChatPlugin::CHATtab_create(void)
{

	CHATTab = new ui::Menu("CHAT", this);
	reset = new ui::Button(CHATTab, "Reset");
	reset->setText("Reset");
	reset->setCallback([this](bool) {
		
	});
	recVolSlider = new ui::Slider(CHATTab, "RecVolume");
	recVolSlider->setText("RecVolume");
	recVolSlider->setBounds(1, 100);
	recVolSlider->setValue(40.0);
	recVolSlider->setCallback([this](float value, bool) {
		//frequencySurface->radius1 = value;
		//amplitudeSurface->radius1 = value;
	});
	outVolSlider = new ui::Slider(CHATTab, "OutVolume");
	outVolSlider->setText("OutVolume");
	outVolSlider->setBounds(1, 100);
	outVolSlider->setValue(20.0);
	outVolSlider->setCallback([this](float value, bool) {
		//frequencySurface->radius2 = value;
		//amplitudeSurface->radius2 = value;
	});

	infoLabel = new ui::Label(CHATTab, "CHAT Version 1.0");


}


//--------------------------------------------------------------------
void ChatPlugin::CHATtab_delete(void)
{
	if (CHATTab)
	{
		delete infoLabel;
		delete recVolSlider;
		delete outVolSlider;

		delete CHATTab;
	}
}

void ChatPlugin::initSpeex()
{
	/*Create a new encoder state in narrowband mode*/
	encoderState = speex_encoder_init(&speex_nb_mode);

	/*Set the quality to 8 (15 kbps)*/
	int tmp = 8;
	speex_encoder_ctl(encoderState, SPEEX_SET_QUALITY, &tmp);
	/*Initialization of the structure that holds the bits*/
	speex_bits_init(&bits);

	/*Create a new decoder state in narrowband mode*/
	decoderState = speex_decoder_init(&speex_nb_mode);
	
	/*Set the perceptual enhancement on*/
	tmp = 1;
	speex_decoder_ctl(decoderState, SPEEX_SET_ENH, &tmp);
}
int ChatPlugin::encodeSpeex()
{
	const short* in = (const short*)audioBuffer.constData();
	for (int i = 0; i < frameSize; i++)
		inputData[i] = (float)in[i];
	/*Flush all the bits in the struct so we can encode a new frame*/
	speex_bits_reset(&bits);
	/*Encode the frame*/
	speex_encode(encoderState, inputData, &bits);
	return speex_bits_write(&bits, cbits, 1024);
}

int ChatPlugin::decodeSpeex(const void* data, int length)
{
	/*Copy the data into the bit-stream struct*/
	speex_bits_read_from(&bits, (char *)data, length);
	
	/*Decode the data*/
	speex_decode(decoderState, &bits, outputData);
	/*Copy from float to short (16 bits) for output*/
	
	for (int i = 0; i < frameSize; i++)
		outputDataShort[i] = outputData[i];

	return frameSize*sizeof(short);
}

//--------------------------------------------------------------------

COVERPLUGIN(ChatPlugin)
