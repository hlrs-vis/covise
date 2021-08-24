#pragma once
#include <util/coExport.h>
#include <string>
#include "../../sys/carSound/remoteSoundMessages.h"
namespace remoteSound
{
    class Client;
    class RSEXPORT Sound
    {
    public:
        Sound(Client *c, const std::string filename);
        ~Sound();
        void play();
        void stop();
        void setLoop(bool doLoop, int count=-1);
        void setVolume(float);
        void setPitch(float);
        void setDelay(unsigned long long dspclock_start, unsigned long long dspclock_end, bool stopchannels=true);
        void resend();
    private:
        std::string fileName;
        Client* client;
        int soundID;
        bool loop=false;
        int loopCount=-1;
        float volume=1.0;
        float pitch=1.0;
        bool playing = false;;
        RemoteSoundData rsd;
        RemoteSoundDelayData rsdd;
    };
}