/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <vrml97/vrml/Audio.h>
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

using std::cerr;
using std::endl;
using namespace vrml;

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

void
PlayerAServer::connect()
{
    CERR << "host: " << asHost.c_str() << ", port: " << asPort << endl;

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
        //cerr << "error:" <<error << endl;
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

    //fcntl(asFd, F_SETFL, O_NONBLOCK);
}

PlayerAServer::~PlayerAServer()
{
    //CERR << "destruction of " << sources.size() << " sources" << endl;

    // these would be deleted in the Player destructor, too,
    // but then the asSocket is already closed
    for (std::vector<Player::Source *>::iterator it = sources.begin();
         it != sources.end(); it++)
    {
        delete *it;
        *it = 0;
    }
    sources.resize(0);

    //send_cmd("QUIT");

    if (asFd != -1)
#ifdef WIN32
        closesocket(asFd);
#else
        close(asFd);
#endif
    asFd = -1;
}

int
PlayerAServer::send_cmd(const char *cmd) const
{
    //fprintf(stderr, "AudioServer: %s\n", cmd);
    size_t len = strlen(cmd);
    char *buf = new char[len + 1];
    memcpy(buf, cmd, len);
    buf[len] = '\n';
    return send_data(buf, (int)len + 1, false);
}

int
PlayerAServer::send_data(const char *data, int size, bool swapped) const
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
        //CERR << "jetzt" << endl;
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
            //fprintf(stderr,"errCode %d \n", errCode);
            if ((errCode == EAGAIN) || (errCode == EINTR) ||
#ifndef WIN32
                (errCode == EWOULDBLOCK) ||
#endif
                (errCode == 0))
            {
                n = 0;
                //fprintf(stderr,"EAGAIN or EWOULDBLOCK or 0\n");
            }
            else
                break;
        }
        //fprintf(stderr, "n=%d\n", n);
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
        //CERR << "asFd reset1: " << asFd<< endl;
        asFd = -1;
        return -1;
    }

    return 0;
}

int
PlayerAServer::read_answer(char *buf, int maxsize) const
{
    //CERR << "asFd: " << asFd << endl;
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

void
PlayerAServer::restart()
{
    for (unsigned i = 0; i < sources.size(); i++)
    {
        if (sources[i])
        {
            sources[i]->stopForRestart();
        }
    }

    if (asFd != -1)
    {
        send_cmd("QUIT");
#ifdef WIN32
        closesocket(asFd);
#else
        close(asFd);
#endif
        asFd = -1;
    }

    connect();

    for (unsigned i = 0; i < sources.size(); i++)
    {
        if (sources[i])
        {
            sources[i]->restart();
        }
    }
}

PlayerAServer::Source::Source(const Audio *audio, PlayerAServer *player)
    : Player::Source(audio)
    , asHandle(-1)
    , odirection(0.0)
    , ointensity(0.0)
    , opitch(1.0)
    , ov(0.0, 0.0, 0.0)
    , player(player)
{
    loadAudio(audio);
}

#ifdef _WIN32
#define snprintf _snprintf
#endif

void
PlayerAServer::Source::loadAudio(const Audio *audio)
{
    //CERR << "url" << audio->url() << endl;
    if (!player)
        return;

    char msg[MAX_BUFLEN];
    char filename[MAX_BUFLEN - 100];
    if (audio->url() && strlen(audio->url()) > 0)
    {
        strcpy(filename, audio->url());
    }
    else
    {
        snprintf(filename, sizeof(filename), "PlayerAServer-%d-%08lx.wav",
#ifdef _WIN32
                 0,
#else
                 getpid(),
#endif
                 reinterpret_cast<unsigned long>(audio));
    }
    snprintf(msg, sizeof(msg), "GHDL %s", filename);
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
            //fprintf(stderr, "read: %s\n", msg);
            asHandle = atol(msg);
        }
    }

    if (asHandle < 0 && audio->numBytes() > 0)
    {
        // data is not yet cached on the server, send it
        Audio::WaveHeader header;
        audio->createWaveHeader(&header);
        int numbytes = audio->numBytes();
        snprintf(msg, sizeof(msg), "PTFI %s %lu\r", filename, (unsigned long)(numbytes + sizeof(header)));
        if (player->send_cmd(msg) < 0)
        {
            CERR << "writing command failed" << endl;
        }
        if (player->send_data((char *)&header, sizeof(header)) < 0)
        {
            CERR << "writing header failed" << endl;
        }
#ifdef BYTESWAP
        // BYTESWAP means little endian, audio server is running on little endian
        bool swap = false;
#else
        bool swap = true;
#endif
        if (player->send_data((char *)audio->samples(), numbytes, swap) < 0)
        {
            CERR << "writing sample data (" << numbytes << " bytes) failed" << endl;
        }

        // try again to get handle
        snprintf(msg, sizeof(msg), "GHDL %s", filename);
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
        //fprintf(stderr, "read: %s\n", msg);
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

void
PlayerAServer::Source::setAudio(const Audio *audio)
{
    char msg[MAX_BUFLEN];

    if (asHandle >= 0)
    {
        snprintf(msg, sizeof(msg), "RHDL %d", asHandle);
        player->send_cmd(msg);
    }

    loadAudio(audio);
}

PlayerAServer::Source::~Source()
{
    //CERR << "destroy source " << asHandle << endl;

    char msg[MAX_BUFLEN];

    stop();

    if (asHandle >= 0)
    {
        snprintf(msg, sizeof(msg), "RHDL %d", asHandle);
        player->send_cmd(msg);
    }
}

void
PlayerAServer::Source::play(double start)
{
    char msg[MAX_BUFLEN];

    if (asHandle >= 0)
    {
        snprintf(msg, sizeof(msg), "PLAY %d %f", asHandle, start);
        if (player->send_cmd(msg) < 0)
        {
            CERR << "playing handle " << asHandle << " failed" << endl;
        }
    }

    Player::Source::play(start);
}

void
PlayerAServer::Source::play()
{
    Player::Source::play();
}

void
PlayerAServer::Source::stop()
{
    Player::Source::stop();

    char msg[MAX_BUFLEN];

    if (asHandle >= 0)
    {
        snprintf(msg, sizeof(msg), "STOP %d", asHandle);
        player->send_cmd(msg);
    }
}

void
PlayerAServer::Source::stopForRestart()
{
    Player::Source::stop();

    if (asHandle >= 0)
    {
        char msg[MAX_BUFLEN];
        snprintf(msg, sizeof(msg), "STOP %d", asHandle);
        player->send_cmd(msg);

        snprintf(msg, sizeof(msg), "RHDL %d", asHandle);
        player->send_cmd(msg);
    }
}

void
PlayerAServer::Source::restart()
{
    loadAudio(audio);
}

void
PlayerAServer::update()
{
    Player::update();

    for (unsigned i = 0; i < sources.size(); i++)
    {
        if (sources[i])
        {
            sources[i]->update(this);
        }
    }
}

int
PlayerAServer::Source::update(const Player *genericPlayer, char *buf, int bufsize)
{
    Player::Source::update(genericPlayer, buf, bufsize);

    // dynamic_cast causes problems on gcc2 systems
    //const PlayerAServer *player = dynamic_cast<const PlayerAServer *>(genericPlayer);
    const PlayerAServer *player = (const PlayerAServer *)(genericPlayer);
    if (!player)
        return -1;

    if (asHandle < 0)
        return -1;

    if (!isPlaying())
        return -1;

    char msg[MAX_BUFLEN];
    vec rel_src = x.sub(player->getListenerPositionWC());
    float cos_angle = rel_src.normalize().dot(vec(0.0, 1.0, 0.0));
    float direction = acos(cos_angle);
    //if(rel_src.x < 0.0)
    //direction = (float)(1.0*M_PI - direction);
    //direction *= 180.0/M_PI;

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

    if (v.x != ov.x || v.y != ov.y || v.z != ov.z)
    {
        //snprintf(msg, sizeof(msg), "SSVE %d %f %f %f", asHandle, v.x, v.y, v.z);
        snprintf(msg, sizeof(msg), "SSVE %d %f", asHandle, v.length());
        player->send_cmd(msg);
        ov = v;
    }

    if (opitch != pitch)
    {
        snprintf(msg, sizeof(msg), "SSPI %d %f", asHandle, pitch);
        player->send_cmd(msg);
        opitch = pitch;
    }

    return 0;
}

void
PlayerAServer::Source::setLoop(bool loop)
{
    Player::Source::setLoop(loop);

    char msg[MAX_BUFLEN];

    if (asHandle >= 0)
    {
        snprintf(msg, sizeof(msg), "SSLP %d %d", asHandle, loop ? 1 : 0);
        player->send_cmd(msg);
    }
}

Player::Source *
PlayerAServer::newSource(const Audio *audio)
{
    Source *src = new Source(audio, this);
    int handle = addSource(src);
    if (-1 == handle)
    {
        delete src;
        src = 0;
    }

    return src;
}

void
PlayerAServer::setEAXEnvironment(int env)
{
    char msg[MAX_BUFLEN];
    snprintf(msg, sizeof(msg), "SEEN %d", env);
    send_cmd(msg);
}
