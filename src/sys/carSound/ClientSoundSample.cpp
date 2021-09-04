/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "ClientSoundSample.h"
#include "soundClient.h"
#include <iostream>
#include <strstream>
#include <QMessageBox>
#include "mainWindow.h"
#include <net/tokenbuffer.h>
#include <net/message_types.h>
#include "remoteSoundMessages.h"
#include <boost/filesystem.hpp>

ClientSoundSample::ClientSoundSample(const std::string& name, size_t fileSize, time_t fileTime, soundClient *c)
{
    namespace fs = boost::filesystem;
    FMOD_RESULT result;
	sound = NULL;
	channel = NULL;
    fileName = name;
    client = c;
    stopIcon = QIcon(":icons/media-playback-stop.svg");
    playIcon = QIcon(":icons/media-playback-start.svg");
    pauseIcon = QIcon(":icons/media-playback-pause.svg");
    backwardIcon = QIcon(":icons/media-skip-backward.svg");
    forwardIcon = QIcon(":icons/media-skip-forward.svg");

    ID = IDCounter++;


    myItem = new QTreeWidgetItem(mainWindow::instance()->soundTable);
    myItem->setText(SoundColumns::CSoundID, QString::number(ID));
    myItem->setText(SoundColumns::CClient, QString::number(client->ID));
    myItem->setText(SoundColumns::CFileName,fileName.c_str());
    myItem->setIcon(SoundColumns::CState, stopIcon);

    fs::path p(fileName);
    size_t localFileSize = 0;
    time_t localFileTime = 0;
    if (fs::exists(p))
    {
        localFileSize = fs::file_size(p);
        localFileTime = fs::last_write_time(p);
    }
    if (fileSize == localFileSize)
    {
        covise::TokenBuffer tb;
        tb << (int)SoundMessages::SOUND_SOUND_ID;
        tb << ID;
        client->send(tb);
    }
    else
    {
        cacheFileName = createCacheFileName(fileName);
        fs::path cp(cacheFileName);
        size_t localFileSize = 0;
        time_t localFileTime = 0;
        if (fs::exists(cp))
        {
            localFileSize = fs::file_size(cp);
            localFileTime = fs::last_write_time(cp);
        }
        if (fileSize == localFileSize && localFileTime == fileTime)
        {
            covise::TokenBuffer tb;
            tb << (int)SoundMessages::SOUND_SOUND_ID;
            tb << ID;
            client->send(tb);
        }
        else
        {
            covise::TokenBuffer tb;
            tb << (int)SoundMessages::SOUND_SOUND_ID;
            tb << -1; // report back that we need this file
            client->send(tb);

            covise::Message* m = client->receiveMessage();

            if (m->type == covise::COVISE_MESSAGE_CLOSE_SOCKET || m->type == covise::COVISE_MESSAGE_SOCKET_CLOSED || m->type == covise::COVISE_MESSAGE_QUIT)
            {
                mainWindow::instance()->removeClient(this->client);
                return;
            }
            else if (m->type == covise::COVISE_MESSAGE_SOUND)
            {
                covise::TokenBuffer tb(m);
                int type;
                std::string fn;
                size_t fs;
                time_t ft;
                tb >> type;
                if(type == SoundMessages::SOUND_SOUND_FILE)
                {
                    tb >> fn;
                    tb >> fs;
                    tb >> ft;

                    const char* fileBuf = tb.getBinary(fs);
                    fileName = cacheFileName;
#ifdef WIN32
                    int fd = open(cacheFileName.c_str(), O_RDWR | O_BINARY | O_CREAT);
#else
                    int fd = open(cacheFileName.c_str(), O_RDWR | O_CREAT);
#endif
                    if (fd == -1)
                    {
                        fprintf(stderr, "could not write to %s", cacheFileName.c_str());
                        covise::TokenBuffer tb;
                        tb << (int)SoundMessages::SOUND_SOUND_ID;
                        tb << -1;
                        client->send(tb);
                        return;
                    }
                    size_t sr = write(fd, fileBuf, fileSize);
                    close(fd);
                    fs::last_write_time(cp, ft);

                }
            }
            else
            {
                cerr << "wrong message in new ClientSoundSample" << endl;
                covise::TokenBuffer tb;
                tb << (int)SoundMessages::SOUND_SOUND_ID;
                tb << -1; // failed
                client->send(tb);
            }
        }
    }

    result = mainWindow::instance()->system->createSound(fileName.c_str(), FMOD_DEFAULT | FMOD_LOOP_NORMAL, 0, &sound); // FMOD_DEFAULT uses the defaults.  These are the same as FMOD_LOOP_OFF | FMOD_2D | FMOD_HARDWARE.
    //ERRCHECK(result);
    result = mainWindow::instance()->system->playSound(sound, NULL, true, &channel);
    //ERRCHECK(result);
    covise::TokenBuffer tb;
    tb << (int)SoundMessages::SOUND_SOUND_ID;
    tb << ID;
    client->send(tb);
}

ClientSoundSample::~ClientSoundSample()
{
    delete myItem;
    channel->setPaused(true);
    sound->release();
    if (mainWindow::instance()->currentSound == this)
        mainWindow::instance()->currentSound = nullptr;
}

void ClientSoundSample::start()
{
    bool isp;
    channel->isPlaying(&isp);
    if (!isp)
    {
		mainWindow::instance()->system->playSound(sound, NULL, true, &channel);
        if (looping)
            channel->setLoopCount(-1);
        else
            channel->setLoopCount(0);
    }
    playing = true;
    FMOD_RESULT result;
    result = channel->setPosition(0, FMOD_TIMEUNIT_PCM);
    channel->setPaused(!playing); // This is where the sound really starts.
    myItem->setIcon(SoundColumns::CState, playIcon);
    //ERRCHECK(result);
};
void ClientSoundSample::continuePlaying()
{
    playing = true;
    FMOD_RESULT result;
    result = channel->setPaused(!playing); // This is where the sound really starts.
    myItem->setIcon(SoundColumns::CState, playIcon);
    //ERRCHECK(result);
};
void ClientSoundSample::stop()
{
    playing = false;
    FMOD_RESULT result;
    result = channel->setPaused(!playing); // This is where the sound really starts.
    channel->setPosition(0, FMOD_TIMEUNIT_PCM);
    myItem->setIcon(SoundColumns::CState, stopIcon);
    //ERRCHECK(result);
};
void ClientSoundSample::rewind(){};
void ClientSoundSample::loop(bool l,int count)
{
    looping = l;
    if (l)
        channel->setLoopCount(count);
    else
        channel->setLoopCount(0);
};

void ClientSoundSample::volume(float v)
{
    channel->setVolume(v);
    myItem->setText(SoundColumns::CVolume, QString::number(v));
}
void ClientSoundSample::pitch(float p)
{
    channel->setPitch(p);
    myItem->setText(SoundColumns::CPitch, QString::number(p));
}

void ClientSoundSample::setDelay(unsigned long long dspclock_start, unsigned long long dspclock_end, bool stopchannels)
{
    channel->setDelay(dspclock_start, dspclock_end, stopchannels);
}

std::string ClientSoundSample::createCacheFileName(const std::string& fileName)
{
    std::string cn = fileName;
    size_t sl = cn.length();
    for (size_t i=0;i<sl;i++)
    {
        if (cn[i] == ':')
            cn[i] = '_';
        else if (cn[i] == '\\')
            cn[i] = '_';
        else if (cn[i] == '/')
            cn[i] = '_';
    }
    return cn;
}

int ClientSoundSample::IDCounter = 10;