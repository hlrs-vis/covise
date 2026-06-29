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

inline int errno_sys()
{
#ifdef _WIN32
    return WSAGetLastError();
#else
    return errno;
#endif
}

PlayerAServer::PlayerAServer(const Listener *listener, const std::string &host, int port)
    : Player(listener)
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

    asFd = (int)socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (asFd == -1)
    {
        CERR << "opening socket failed" << endl;
        return;
    }

    struct sockaddr_in hostaddr;
    memset(&hostaddr, 0, sizeof(hostaddr));
    hostaddr.sin_family = AF_INET;
    int sourceport = 31431;
    hostaddr.sin_port = htons(sourceport);
    hostaddr.sin_addr.s_addr = INADDR_ANY;
    while (bind(asFd, (struct sockaddr *)&hostaddr, sizeof(hostaddr)) < 0)
    {
#ifndef _WIN32
        if (errno == EADDRINUSE)
#else
        if (GetLastError() == WSAEADDRINUSE)
#endif
        {
            sourceport++;
            hostaddr.sin_port = htons(sourceport);
        }
        else
        {

#ifdef WIN32
            closesocket(asFd);
#else
            close(asFd);
#endif
            CERR << "binding socket failed: " << strerror(errno_sys()) << endl;
            asFd = -1;
            return;
        }
    }

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
#ifdef WIN32
        int error = WSAGetLastError();
        // cerr << "error:" <<error << endl;
        if (error != WSAEISCONN) // this is a weird Windows specific necessity!
        {
//  closesocket(asFd);
#else
        close(asFd);

        CERR << "connecting socket failed: " << strerror(errno_sys()) << endl;
        asFd = -1;
        return;
#endif
#ifdef WIN32
        }
#endif
    }

    cerr << " ok." << endl;

    // fcntl(asFd, F_SETFL, O_NONBLOCK);
}

PlayerAServer::~PlayerAServer()
{
    if (asFd != -1)
#ifdef WIN32
        closesocket(asFd);
#else
        close(asFd);
#endif
    asFd = -1;
}

int PlayerAServer::send_cmd(const char *cmd) const
{
    // fprintf(stderr, "AudioServer: %s\n", cmd);
    size_t len = strlen(cmd);
    char *buf = new char[len + 1];
    memcpy(buf, cmd, len);
    buf[len] = '\n';
    return send_data(buf, (int)len + 1, false);
}

int PlayerAServer::send_data(const char *data, int size, bool swapped) const
{
    if (asFd == -1)
    {
        return 0;
    }

    char *buf = NULL;
    if (swapped)
    {
        if (size % 2)
        {
            CERR << "odd: odd number of bytes to swap" << endl;
            return -1;
        }
        buf = new char[size];
        for (int i = 0; i < size; i += 2)
        {
            buf[i] = data[i + 1];
            buf[i + 1] = data[i];
        }
        data = buf;
    }

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

    if (swapped)
    {
        delete[] buf;
    }

    if (written < size)
    {
        CERR << "write failed: " << strerror(errno_sys()) << endl;
    }

    if (written != size)
    {
#ifdef WIN32
        closesocket(asFd);
#else
        close(asFd);
#endif
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
#ifdef WIN32
        closesocket(asFd);
#else
        close(asFd);
#endif
        asFd = -1;
        return -1;
    }
    if (ret == 0)
    {
        CERR << "timed out" << endl;
#ifdef WIN32
        closesocket(asFd);
#else
        close(asFd);
#endif
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
#ifdef WIN32
            closesocket(asFd);
#else
            close(asFd);
#endif
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
    if (player->send_cmd(msg) < 0)
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
        if (player->send_cmd(msg) < 0)
        {
            CERR << "writing command failed" << endl;
        }

        // Send file contents
        FILE *fd = fopen(path.string().c_str(), "rb");
        char buf[4096];
        for (size_t i = 0; i < file_size; i += 4096)
        {
            size_t s = fread(buf, 1, 4096, fd);
            if (player->send_data(buf, s) < 0)
            {
                CERR << "writing data failed" << endl;
            }
        }

        // Try again to get handle
        snprintf(msg, sizeof(msg), "GHDL %s", path.filename().c_str());
        if (player->send_cmd(msg) != 0)
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
        player->send_cmd(msg);

        snprintf(msg, sizeof(msg), "SSPO %d 0.0 0.0 0.0", asHandle);
        player->send_cmd(msg);
    }
}

void PlayerAServer::Source::setAudio(const Audio *audio)
{
    char msg[MAX_BUFLEN];

    if (asHandle >= 0)
    {
        snprintf(msg, sizeof(msg), "RHDL %d", asHandle);
        const PlayerAServer *player = (PlayerAServer *)this->player;
        player->send_cmd(msg);
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
        player->send_cmd(msg);
    }
}

void PlayerAServer::Source::play(double start)
{
    char msg[MAX_BUFLEN];

    if (asHandle >= 0)
    {
        snprintf(msg, sizeof(msg), "PLAY %d %f", asHandle, start);
        const PlayerAServer *player = (PlayerAServer *)this->player;
        if (player->send_cmd(msg) < 0)
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
        player->send_cmd(msg);
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
            player->send_cmd(msg);
            ointensity = intensity;
            odirection = direction;
        }
    }
    else
    {
        if (ointensity != intensity)
        {
            snprintf(msg, sizeof(msg), "SSVO %d %f", asHandle, intensity);
            player->send_cmd(msg);
        }
    }

    if (v != ov)
    {
        // snprintf(msg, sizeof(msg), "SSVE %d %f %f %f", asHandle, v.x, v.y, v.z);
        snprintf(msg, sizeof(msg), "SSVE %d %f", asHandle, glm::length(v));
        player->send_cmd(msg);
        ov = v;
    }

    if (opitch != pitch)
    {
        snprintf(msg, sizeof(msg), "SSPI %d %f", asHandle, pitch);
        player->send_cmd(msg);
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
        player->send_cmd(msg);
    }
}

std::unique_ptr<opencover::audio::Source>
PlayerAServer::makeSource(const Audio *audio)
{
    return std::make_unique<PlayerAServer::Source>(this, audio);
}
