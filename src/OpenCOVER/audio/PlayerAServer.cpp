/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Audio.h"
#include "PlayerAServer.h"
#include <sys/types.h>
#ifdef WIN32
#include <errno.h>
#include <winsock2.h>
#include <util/unixcompat.h>
#include <windows.h>
#else
#include <sys/socket.h>
#include <netdb.h>
#include <errno.h>
#include <unistd.h>
#include <netinet/in.h>
#endif
#include <fcntl.h>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <filesystem>

using std::cerr;
using std::endl;
using namespace opencover::audio;

#define MAX_BUFLEN 1024

namespace
{

int errno_sys()
{
#ifdef _WIN32
    return WSAGetLastError();
#else
    return errno;
#endif
}

#ifdef _WIN32
int close(int fd)
{
    return closesocket(fd);
}
#endif

}

PlayerAServer::PlayerAServer(const Listener *listener, const std::string &host, int port, bool isMaster)
    : Player(listener, isMaster)
    , asFd(-1)
    , asHost(host)
    , asPort(port)
{
    connect();
}

#ifdef CERR
#undef CERR
#endif
#define CERR std::cout

void PlayerAServer::connect()
{
    cerr << "AServer::connect()" << endl;
    if (!isMaster)
    {
        asFd = -1;
        cerr << "AServer::connect() not Master" << endl;
        return;
    }
    cerr << "AServer::connect()2" << endl;
    if (asHost.empty() || asPort <= 0)
    {
        CERR << "audio server not configured" << endl;
        return;
    }

    CERR << "connecting to audio server \"" << asHost.c_str() << "\" at port " << asPort << "...";

    struct hostent *host = gethostbyname(asHost.c_str());
    if (!host)
    {
        asFd = -1;
        return;
    }

    asFd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (asFd == -1)
    {
        CERR << "opening socket failed" << endl;
        return;
    }

    if (fcntl(asFd, F_SETFL, O_NONBLOCK) == -1)
    {
        CERR << "making socket non-blocking failed" << endl;
        close(asFd);
        asFd = -1;
        return;
    }

    struct sockaddr_in hostaddr;
    memset(&hostaddr, 0, sizeof(hostaddr));
    hostaddr.sin_family = AF_INET;
    hostaddr.sin_port = htons((unsigned short)asPort);
#ifdef WIN32
    char char_address[4];
    char_address[0] = *host->h_addr_list[0];
    char_address[1] = *(host->h_addr_list[0] + 1);
    char_address[2] = *(host->h_addr_list[0] + 2);
    char_address[3] = *(host->h_addr_list[0] + 3);
    hostaddr.sin_addr.s_addr = *((uint32_t *)char_address);
#else
    hostaddr.sin_addr.s_addr = *(in_addr_t *)host->h_addr;
#endif

    if (::connect(asFd, (struct sockaddr *)&hostaddr, sizeof(hostaddr)) == -1)
    {
        auto error = errno_sys();
        if (error != EINPROGRESS
#ifdef WIN32
            || error != WSAEISCONN // this is a weird Windows specific necessity!
#endif
        )
        {
            CERR << "connecting socket failed: " << strerror(errno_sys()) << endl;
            close(asFd);
            asFd = -1;
            return;
        }
    }

    fd_set fdset;
    FD_ZERO(&fdset);
    FD_SET(asFd, &fdset);
    struct timeval tv;
    tv.tv_sec = 3;
    tv.tv_usec = 0;

    if (select(asFd + 1, NULL, &fdset, NULL, &tv) == 1)
    {
        int so_error;
        socklen_t len = sizeof so_error;

        getsockopt(asFd, SOL_SOCKET, SO_ERROR, &so_error, &len);

        if (so_error != 0)
        {
            CERR << "connecting socket failed: " << strerror(errno_sys()) << endl;
            close(asFd);
            asFd = -1;
            return;
        }

        if (fcntl(asFd, F_SETFL, 0) == -1)
        {
            CERR << "making socket blocking again failed" << endl;
            close(asFd);
            asFd = -1;
            return;
        }
    }

    cerr << " ok." << endl;

    // fcntl(asFd, F_SETFL, O_NONBLOCK);
}

PlayerAServer::~PlayerAServer()
{
    if (asFd != -1)
        close(asFd);
    asFd = -1;
}

bool PlayerAServer::isConnected() const
{
    return asFd != -1;
}

int PlayerAServer::sendCommand(std::string_view cmd) const
{
    std::string buf = std::string(cmd) + "\n";
    std::cout << "OOOOO " << cmd << std::endl;
    return sendData(buf.c_str(), buf.length());
}

int PlayerAServer::sendData(const char *data, int size) const
{
    if (asFd == -1)
    {
        return 0;
    }

    char *buf = NULL;

    int written = 0;
    do
    {
        // CERR << "jetzt" << endl;
        fd_set writefds;
        FD_ZERO(&writefds);
        FD_SET(asFd, &writefds);
#if 0
       struct timeval tv;
       tv.tv_sec = 10;
       tv.tv_usec = 0;

       int ret = select(asFd+1, NULL, &writefds, NULL, &tv);
       if( ret < 0 )
       {
           CERR << "select failed: " << strerror(errno_sys()) << endl;
           break;
       }
       if( ret==0 )
       {
           CERR << "timed out" << endl;
           break;
       }
#endif

#ifdef _WIN32
        int n = send(asFd, data + written, size - written, 0);
#else
        int n = write(asFd, data + written, size - written);
#endif
        if (n < 0)
        {
            int errCode = errno_sys();
            // fprintf(stderr,"errCode %d \n", errCode);
            if ((errCode == EAGAIN) || (errCode == EINTR) ||
#ifndef WIN32
                (errCode == EWOULDBLOCK) ||
#endif
                (errCode == 0))
            {
                n = 0;
                // fprintf(stderr,"EAGAIN or EWOULDBLOCK or 0\n");
            }
            else
                break;
        }
        // fprintf(stderr, "n=%d\n", n);
        written += n;
    } while (written < size);

    if (written < size)
    {
        CERR << "write failed: " << strerror(errno_sys()) << endl;
    }

    if (written != size)
    {
        close(asFd);
        // CERR << "asFd reset1: " << asFd<< endl;
        asFd = -1;
        return -1;
    }

    return 0;
}

int PlayerAServer::read_answer(char *buf, int maxsize) const
{
    // CERR << "asFd: " << asFd << endl;
    if (asFd == -1)
        return -1;

    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(asFd, &readfds);
    struct timeval tv;
    tv.tv_sec = 10;
    tv.tv_usec = 0;
    int ret = select(asFd + 1, &readfds, NULL, NULL, &tv);
    if (ret < 0)
    {
        CERR << "select failed: " << strerror(errno_sys()) << endl;
        close(asFd);
        asFd = -1;
        return -1;
    }
    if (ret == 0)
    {
        CERR << "timed out" << endl;
        close(asFd);
        asFd = -1;
        return -1;
    }
    int nread = 0;
    while (nread < maxsize)
    {

#ifdef _WIN32
        int n = recv(asFd, buf + nread, 1, 0);
#else
        int n = read(asFd, buf + nread, 1);
#endif

        if (n < 0)
        {
            close(asFd);
            asFd = -1;
            return -1;
        }
        if (buf[nread] == '\n')
        {
            buf[nread] = '\0';
            return 0;
        }
        nread++;
    }

    if (nread < maxsize)
    {
        buf[nread] = '\0';
        return 0;
    }

    return -1;
}

PlayerAServer::Source::Source(Player *player, const Audio *audio)
    : opencover::audio::Source(player, audio)
    , asHandle(-1)
    , odirection(0.0)
    , ointensity(0.0)
    , opitch(1.0)
    , ov(0.0, 0.0, 0.0)
{
    loadAudio(audio);
}

#ifdef _WIN32
#define snprintf _snprintf
#endif

void PlayerAServer::Source::loadAudio(const Audio *audio)
{
    const PlayerAServer *player = (PlayerAServer *)this->player;

    if (!player)
        return;

    std::filesystem::path path(audio->url());

    char msg[MAX_BUFLEN];
    snprintf(msg, sizeof(msg), "GHDL %s", path.filename().c_str());
    if (player->sendCommand(msg) < 0)
    {
        CERR << "writing GHDL command failed" << endl;
        asHandle = -1;
    }
    else
    {
        if (player->read_answer(msg, sizeof(msg)) < 0)
        {
            CERR << "reading answer to GHDL failed" << endl;
            asHandle = -1;
        }
        else
        {
            // fprintf(stderr, "read: %s\n", msg);
            asHandle = atol(msg);
        }
    }

    if (asHandle < 0)
    {
        // Audio file is not yet cached on the server, send it.

        size_t file_size = std::filesystem::file_size(path);
        snprintf(msg, sizeof(msg), "PTFI %s %lu\r", path.filename().c_str(), file_size);
        if (player->sendCommand(msg) < 0)
        {
            CERR << "writing command failed" << endl;
        }

        // Send file contents
        FILE *fd = fopen(path.string().c_str(), "rb");
        char buf[4096];
        for (size_t i = 0; i < file_size; i += 4096)
        {
            size_t s = fread(buf, 1, 4096, fd);
            if (player->sendData(buf, s) < 0)
            {
                CERR << "writing data failed" << endl;
            }
        }

        // Try again to get handle
        snprintf(msg, sizeof(msg), "GHDL %s", path.filename().c_str());
        if (player->sendCommand(msg) != 0)
        {
            CERR << "writing command failed" << endl;
            asHandle = -1;
            return;
        }
        if (player->read_answer(msg, sizeof(msg)) < 0)
        {
            CERR << "reading handle failed" << endl;
            asHandle = -1;
            return;
        }
        // fprintf(stderr, "read: %s\n", msg);
        asHandle = atol(msg);
    }

    if (asHandle >= 0)
    {
        snprintf(msg, sizeof(msg), "SSVO %d 0.0", asHandle);
        player->sendCommand(msg);

        snprintf(msg, sizeof(msg), "SSPO %d 0.0 0.0 0.0", asHandle);
        player->sendCommand(msg);
    }
}

void PlayerAServer::Source::setAudio(const Audio *audio)
{
    char msg[MAX_BUFLEN];

    if (asHandle >= 0)
    {
        snprintf(msg, sizeof(msg), "RHDL %d", asHandle);
        const PlayerAServer *player = (PlayerAServer *)this->player;
        player->sendCommand(msg);
    }

    loadAudio(audio);
}

PlayerAServer::Source::~Source()
{
    // CERR << "destroy source " << asHandle << endl;

    char msg[MAX_BUFLEN];

    stop();

    if (asHandle >= 0)
    {
        snprintf(msg, sizeof(msg), "RHDL %d", asHandle);
        const PlayerAServer *player = (PlayerAServer *)this->player;
        player->sendCommand(msg);
    }
}

void PlayerAServer::Source::play(double start)
{
    char msg[MAX_BUFLEN];

    if (asHandle >= 0)
    {
        snprintf(msg, sizeof(msg), "PLAY %d %f", asHandle, start);
        const PlayerAServer *player = (PlayerAServer *)this->player;
        if (player->sendCommand(msg) < 0)
        {
            CERR << "playing handle " << asHandle << " failed" << endl;
        }
    }

    opencover::audio::Source::play(start);
}

void PlayerAServer::Source::play()
{
    opencover::audio::Source::play();
}

void PlayerAServer::Source::stop()
{
    opencover::audio::Source::stop();

    char msg[MAX_BUFLEN];

    if (asHandle >= 0)
    {
        snprintf(msg, sizeof(msg), "STOP %d", asHandle);
        const PlayerAServer *player = (PlayerAServer *)this->player;
        player->sendCommand(msg);
    }
}

void PlayerAServer::Source::update(const Player *genericPlayer)
{
    opencover::audio::Source::update(genericPlayer);

    // dynamic_cast causes problems on gcc2 systems
    // const PlayerAServer *player = dynamic_cast<const PlayerAServer *>(genericPlayer);
    const PlayerAServer *player = (const PlayerAServer *)(genericPlayer);
    if (!player || asHandle < 0 || !isPlaying())
    {
        return;
    }

    char msg[MAX_BUFLEN];
    auto rel_src = glm::normalize(x - player->listener->getPosition());
    // I don't think this is correct, the source needs to have a direction too
    float cos_angle = glm::dot(rel_src, glm::vec3(0.0, 1.0, 0.0));
    float direction = acos(cos_angle);

    if (spatialize)
    {
        if (odirection != direction || ointensity != intensity)
        {
            snprintf(msg, sizeof(msg), "SSDV %d %f %f", asHandle, direction, intensity);
            player->sendCommand(msg);
            ointensity = intensity;
            odirection = direction;
        }
    }
    else
    {
        if (ointensity != intensity)
        {
            snprintf(msg, sizeof(msg), "SSVO %d %f", asHandle, intensity);
            player->sendCommand(msg);
        }
    }

    if (v != ov)
    {
        // snprintf(msg, sizeof(msg), "SSVE %d %f %f %f", asHandle, v.x, v.y, v.z);
        snprintf(msg, sizeof(msg), "SSVE %d %f", asHandle, glm::length(v));
        player->sendCommand(msg);
        ov = v;
    }

    if (opitch != pitch)
    {
        snprintf(msg, sizeof(msg), "SPIT %d %f", asHandle, pitch);
        player->sendCommand(msg);
        opitch = pitch;
    }
}

void PlayerAServer::Source::setLoop(bool loop)
{
    opencover::audio::Source::setLoop(loop);

    char msg[MAX_BUFLEN];

    if (asHandle >= 0)
    {
        snprintf(msg, sizeof(msg), "SSLP %d %d", asHandle, loop ? 1 : 0);
        const PlayerAServer *player = (PlayerAServer *)this->player;
        player->sendCommand(msg);
    }
}

std::unique_ptr<opencover::audio::Source>
PlayerAServer::makeSource(const Audio *audio)
{
    return std::make_unique<PlayerAServer::Source>(this, audio);
}
