/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "UDPClient.h"

//#define DUMP
#include <sstream>
#include <cstring>

unsigned int UDPClient::m_iCurrentPacketNumber = 1;

bool UDPClient::setup(std::string _remote, std::string _port)
{
    struct hostent *host;
#ifdef _WIN32
    if (WSAStartup(MAKEWORD(2, 0), &wsaData) != 0) /* Load Winsock 2.0 DLL */
    {
        fprintf(stderr, "WSAStartup() failed");
        return false;
    }
#endif

    int port = atoi(_port.c_str());
#ifdef _WIN32
    if ((m_sRemote.sin_addr.S_un.S_addr = inet_addr(_remote.c_str())) == -1)
    {
#else
    if (!inet_aton(_remote.c_str(), &m_sRemote.sin_addr))
    {
#endif
        host = gethostbyname(_remote.c_str());
        if (!host)
        {
#ifdef _WIN32
            printf("gethostbyname() failed: %d\n", WSAGetLastError());
#else
            herror("gethostbyname() failed");
#endif
            return false;
        }
        m_sRemote.sin_addr = *(struct in_addr *)host->h_addr;
    }
    m_iSocket = socket(PF_INET, SOCK_DGRAM, 0);
    if (m_iSocket == -1)
    {
        perror("socket() failed");
        return false;
    }

    printf("connecting to %s:%s...\n", inet_ntoa(m_sRemote.sin_addr), _port.c_str());
    fflush(stdout);

    m_sRemote.sin_port = htons(port);
    m_sRemote.sin_family = AF_INET;
    memset(&m_sRemote.sin_zero, 0, sizeof(m_sRemote.sin_zero));

    return true;
}

bool UDPClient::sendMessage(const std::string &str, bool includePacketNumber)
{
    std::string message;
    if (includePacketNumber)
    {
        std::stringstream sstr;
        m_iWaitingForPacket = m_iCurrentPacketNumber;
        sstr << m_iCurrentPacketNumber;

        message = sstr.str() + " " + str;
        ++m_iCurrentPacketNumber;
    }
    else
    {
        message = str;
    }
#ifdef DUMP
    printf("sendData %s packetNumber %d (include %d)\n", str.c_str(), m_iCurrentPacketNumber - 1, includePacketNumber);
#endif

    int pkLen;
    pkLen = sendto(m_iSocket, message.c_str(), message.length(), 0, (struct sockaddr *)&m_sRemote, sizeof(m_sRemote));

    if (pkLen == -1)
    {
        perror("send() failed");
        return false;
    }
    if (pkLen != message.length())
    {
        printf("sent out wrong number of bytes\n");
    }
    else
    {
#ifdef DUMP
        printf("sent out %d number of bytes to %s \n", pkLen, inet_ntoa(m_sRemote.sin_addr));
#endif
    }
    return true;
}

/* sends message over UDP-protocol to remote server */
bool UDPClient::sendData(SendCommandEnum command, bool includePacketNumber)
{
    switch (command)
    {
    case createDebugOutput:
        return sendMessage("createDebugOutput", includePacketNumber);
    }
    return false;
}
/* request data over UDP-protocol from remote server */
bool UDPClient::requestData(RequestCommandEnum command, bool includePacketNumber)
{
    switch (command)
    {
    case MarkerPos_filtered:
        return sendMessage("request MarkerPos_filtered", includePacketNumber);
    case MarkerPos_unfiltered:
        return sendMessage("request MarkerPos_unfiltered", includePacketNumber);
    case Target_filtered:
        return sendMessage("request Target_filtered", includePacketNumber);
    case Target_unfiltered:
        return sendMessage("request Target_unfiltered", includePacketNumber);
    case All:
        return sendMessage("request All", includePacketNumber);
    }
    return false;
}
/* receive data from remote server over UDP-protocol */

bool UDPClient::receiveData(bool &areOld, long timeout_us)
{
#ifdef WIN32
    LARGE_INTEGER start, freq, current;
#else
    struct timeval start, current;
#endif
    int timeout_s = timeout_us / 1000000;
    struct timeval timeout = { timeout_s, static_cast<int>(timeout_us - 1000000 * timeout_s) };

    long int timeDiff;
    bool receivedData = false;
    bool timeOut = false;
#ifdef DUMP
    printf("receiving data\n");
#endif
    fd_set fds_read;
    // wait for answer

    char buffer[BUF_SIZ];
    int bytes;

    std::string sPacketNumber, sLength;

    struct sockaddr_in addr;
#ifdef _WIN32
    int addr_len;
#else
    socklen_t addr_len;
#endif

#ifdef WIN32
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);
#else
    gettimeofday(&start, NULL);
#endif

    FD_ZERO(&fds_read);
    FD_SET(m_iSocket, &fds_read);
    do
    {
#ifdef WIN32
        select(1, &fds_read, NULL, NULL, &timeout);
#else
        select(m_iSocket + 1, &fds_read, NULL, NULL, &timeout);
#endif

        if (!FD_ISSET(m_iSocket, &fds_read))
        {
// answer is lost
#ifdef DUMP
            printf("no data available -> packet lost\n");
#endif
            receivedData = false;
            timeOut = true;
        }

/* Erinnerung: wenn recv() als Rueckgabe 0 liefert, dann
        * wurde die Verbindung geschlossen. Daher hier auf > 0 und
        * nicht auf == -1 testen.
        */
#ifdef DUMP
        printf("receiving data\n");
#endif
        addr_len = sizeof(addr);

        bytes = recvfrom(m_iSocket, buffer, sizeof(buffer) - 1, 0, (struct sockaddr *)&addr, &addr_len);
#ifdef DUMP
        printf("%d bytes received\n", bytes);
#endif
        if (bytes > 0)
        {
            buffer[bytes] = '\0';
            m_sRecvData = buffer;
            sLength = m_sRecvData.substr(0, m_sRecvData.find(' '));
            m_sRecvData = m_sRecvData.substr(m_sRecvData.find(' ') + 1);
            sPacketNumber = m_sRecvData.substr(0, m_sRecvData.find(' '));
            if (atoi(sPacketNumber.c_str()) != 0)
            {
                m_sRecvData = sLength + m_sRecvData.substr(m_sRecvData.find(' '));
            }
            else
            {
                m_sRecvData = sLength + " " + m_sRecvData;
            }
#ifdef DUMP
            printf("received: %s\n", buffer);
            printf("Length: %s PacketNumber: %s\n", sLength.c_str(), sPacketNumber.c_str());
            printf("received: %s\n", m_sRecvData.c_str());
#endif

            if (atoi(sPacketNumber.c_str()) == m_iWaitingForPacket)
            {
                areOld = false;
                receivedData = true;
            }
            else
            {
                areOld = true;
            }
        }
        else if (bytes == -1)
        {
#ifdef DUMP
            printf("waiting...\n");
#endif
            receivedData = false;
        }
#ifdef WIN32
        QueryPerformanceCounter(&current);
        timeDiff = (long)(((current.QuadPart - start.QuadPart) * 1000000) / freq.QuadPart);
#else
        gettimeofday(&current, NULL);
        timeDiff = (current.tv_sec - start.tv_sec) * 1000000 + (current.tv_usec - start.tv_usec);
#endif
        timeout.tv_usec = timeout_us - timeDiff;
        if (timeout.tv_usec < 0)
        {
            timeOut = true;
        }
    } while (!receivedData && !timeOut);
    return receivedData;
}
void UDPClient::close()
{
#ifdef _WIN32
    closesocket(m_iSocket);
    WSACleanup(); /* Cleanup Winsock */
#else
    ::close(m_iSocket); /* Close client socket */
#endif
}
#ifdef WITH_TEST
main(int argc, char *argv[])
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s <remote> <port>\n", argv[0]);
        return 1;
    }
    UDPClient client;
    bool ok, old;
    if (client.setup(argv[1], argv[2]))
    {
        printf("Setup\n");
        do
        {
#ifdef _WIN32
#else
            usleep(10000);
#endif
            ok = client.requestData(All, true);
            ok &= client.receiveData(old);
        } while (ok);
    }
}
#endif
