#include "remoteSound.h"
#include "remoteSoundClient.h"
#include "../../sys/carSound/remoteSoundMessages.h"

#include <net/covise_connect.h>
#include <net/covise_host.h>
#include <net/message_types.h>
#include <net/tokenbuffer.h>

using namespace remoteSound;
Sound::Sound(Client *c , const std::string fn)
{
    fileName = fn;
    client = c;
    soundID = 0;

    rsd.msgType = SoundMessages::TypeRemoteSound;
    rsd.soundID = soundID;
    rsdd.msgType = SoundMessages::TypeRemoteSoundDelay;
    rsdd.soundID = soundID;
    resend();
    client->sounds.push_back(this);
}
Sound::~Sound()
{
    covise::TokenBuffer tb;
    tb << (int)SoundMessages::SOUND_DELETE_SOUND;
    tb << soundID;
    client->send(tb);
    client->sounds.remove(this);
}
void Sound::resend()
{
    if(client->isConnected())
    {
    covise::TokenBuffer tb;
    tb << (int)SoundMessages::SOUND_NEW_SOUND;
    tb << fileName;
    client->send(tb);
    covise::Message* m = client->receiveMessage();
    covise::TokenBuffer rtb(m);
    int messageType = 0;
    rtb >> messageType;
    if (messageType != (int)SoundMessages::SOUND_SOUND_ID)
    {
        fprintf(stderr, "wrong message in reply to NEW_SOUND\n");
    }
    rtb >> soundID;
    if (soundID < 0)
    {
        // not in cache: send file
        m = client->receiveMessage();
        covise::TokenBuffer rtb(m);
        rtb >> soundID;
    }
    if (playing)
        play();
    else
        stop();
    setLoop(loop, loopCount);
    setVolume(volume);
    setPitch(pitch);
    }
}

void Sound::play()
{
    rsd.action = (unsigned char)remoteSoundActions::Start;
    client->send(&rsd, sizeof(rsd));
    playing = true;
}

void Sound::stop()
{
    rsd.action = (unsigned char)remoteSoundActions::Stop;
    client->send(&rsd, sizeof(rsd));
    playing = false;
}

void remoteSound::Sound::setLoop(bool state, int count)///loop count values -1 = endless
{
    if (state)
    {
        rsd.action = (unsigned char)remoteSoundActions::enableLoop; 
        rsd.value = (float)count;
    }
    else
    {
        rsd.action = (unsigned char)remoteSoundActions::disableLoop; //don't loop
        rsd.value = (float)0.0;
    }
    client->send(&rsd, sizeof(rsd));
    loop = state;
    loopCount = count;

}

void remoteSound::Sound::setVolume(float v)
{
    rsd.action = (unsigned char)remoteSoundActions::Volume;
    rsd.value = v;
    client->send(&rsd, sizeof(rsd));
    volume = v;
}

void remoteSound::Sound::setPitch(float p)
{
    rsd.action = (unsigned char)remoteSoundActions::Pitch; 
    rsd.value = p;
    client->send(&rsd, sizeof(rsd));
    pitch = p;
}

void remoteSound::Sound::setDelay(unsigned long long dspclock_start, unsigned long long dspclock_end, bool stopchannels)
{
    rsdd.startValue = dspclock_start;
    rsdd.endValue = dspclock_end;
    rsdd.stopChannel = stopchannels;
    client->send(&rsd, sizeof(rsd));
}
