/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/common.h>
#include <net/covise_socket.h>
#include <net/covise_connect.h>
#include <net/tokenbuffer.h>
#include <net/udpMessage.h>
#include <net/covise_host.h>
#include <config/CoviseConfig.h>
#include "coVRSlave.h"
#include <cover/coVRMSController.h>

#ifdef HAS_MPI
#include <mpi.h>
#ifndef CO_MPI_SEND
#define CO_MPI_SEND MPI_Ssend
#endif
#endif

using namespace opencover;
using namespace covise;

// Enable in kernel.pro and recompile
#ifdef DEBUG_MESSAGES
extern int debugMessageCounter;
#endif

coVRSlave::coVRSlave(int ID)
    : myID(ID)
{
}

coVRSlave::~coVRSlave()
{
}

int coVRSlave::readMessage(Message *msg)
{
    char read_buf[4 * sizeof(int)];
    int *read_buf_int;
    int headerSize = 4 * sizeof(int);
    int toRead;
    int bytesRead = 0;
    int ret = read(read_buf, headerSize);
    if (ret < headerSize)
        return ret;
    read_buf_int = (int *)read_buf;
    msg->sender = read_buf_int[0];
    msg->send_type = read_buf_int[1];
    msg->type = read_buf_int[2];
    msg->data = DataHandle(read_buf_int[3]);
    while (bytesRead < msg->data.length())
    {
        toRead = msg->data.length() - bytesRead;
        if (toRead > READ_BUFFER_SIZE)
            toRead = READ_BUFFER_SIZE;
        ret = read(msg->data.accessData() + bytesRead, toRead);
        if (ret < toRead)
            return ret;
        bytesRead += toRead;
    }

    return bytesRead;
}

int coVRSlave::sendMessage(const Message *msg)
{
    char write_buf[WRITE_BUFFER_SIZE];
    int *write_buf_int;
    int headerSize = 4 * sizeof(int);
    int len = msg->data.length() + headerSize;
    int toWrite;
    int written = 0;
    toWrite = len;
    if (toWrite > WRITE_BUFFER_SIZE)
        toWrite = WRITE_BUFFER_SIZE;
    write_buf_int = (int *)write_buf;
    write_buf_int[0] = msg->sender;
    write_buf_int[1] = msg->send_type;
    write_buf_int[2] = msg->type;
    write_buf_int[3] = msg->data.length();
    if (toWrite > WRITE_BUFFER_SIZE)
        toWrite = WRITE_BUFFER_SIZE;
    memcpy(write_buf + headerSize, msg->data.data(), toWrite - headerSize);
#ifdef DEBUG_MESSAGES
    int tmpLen;
    send((char *)&headerSize, sizeof(headerSize));
    send((char *)&debugMessageCounter, sizeof(debugMessageCounter));
    read((char *)&tmpLen, sizeof(tmpLen));
#endif
    send(write_buf, toWrite);
    written += toWrite;
    while (written < len)
    {
        toWrite = len - written;
        if (toWrite > WRITE_BUFFER_SIZE)
            toWrite = WRITE_BUFFER_SIZE;
        int numSent = send(msg->data.data() + written - headerSize, toWrite);
        if (numSent < toWrite)
        {
            cerr << "numSent = " << numSent << " toWrite = " << toWrite << endl;
            if (numSent < 0)
            {
                return numSent;
            }
        }
        written += numSent;
    }
    return len;
}

void coVRSlave::sendMessage(const covise::UdpMessage* msg)
{
	char *write_buf = new char[UDP_MESSAGE_HEADER_SIZE + msg->data.length()];
	int* write_buf_int;
	write_buf_int = (int*)write_buf;
	write_buf_int[0] = msg->type;
	write_buf_int[1] = msg->sender;

	if (UDP_MESSAGE_HEADER_SIZE + msg->data.length() > WRITE_BUFFER_SIZE)
	{
		cerr << "udp message of type " << msg->type << " is too long!" << endl;
	}
	write_buf_int[2] = msg->data.length();
	//cerr << "sending udp msg to slave " << getID() << " type = " << write_buf_int[0] << " sender = " << write_buf_int[1] << " length = " << write_buf_int[2] << endl;
	memcpy(write_buf + UDP_MESSAGE_HEADER_SIZE, msg->data.data(), msg->data.length());
	send(write_buf, UDP_MESSAGE_HEADER_SIZE + msg->data.length());
	delete[]write_buf;
}

int coVRSlave::readMessage(covise::UdpMessage* msg)
{
	char read_buf[UDP_MESSAGE_HEADER_SIZE];
	int* read_buf_int = (int*)read_buf;
	int ret = read(read_buf, UDP_MESSAGE_HEADER_SIZE);
	if (ret < UDP_MESSAGE_HEADER_SIZE)
		return -1;
	msg->type = (udp_msg_type)read_buf_int[0];
	msg->sender = read_buf_int[1];
	if (read_buf_int[2] > WRITE_BUFFER_SIZE - UDP_MESSAGE_HEADER_SIZE)
	{
		cerr << "udp message of type " << msg->type << " was too long to read;" << endl;
		return 0;
	}
    msg->data = DataHandle(read_buf_int[2]);
	ret = read(msg->data.accessData(), msg->data.length());
	return ret;
}

coVRTcpSlave::coVRTcpSlave(int ID)
    : coVRSlave(ID)
{
    socket = new Socket(&port);
    socket->listen();
    socketDraw = new Socket(&port2);
    socketDraw->listen();
    //std::cerr << "coVRTcpSlave(" << ID << "): fds: app=" << port << ", draw=" << port2 << std::endl;
}

coVRTcpSlave::~coVRTcpSlave()
{
}

int coVRTcpSlave::read(void *c, int n)
{
    int ret;
    do
    {
        ret = socket->Read(c, n);

    } while ((ret < 0) && ((errno == EAGAIN) || (errno == EINTR)));
    return ret;
}

int coVRTcpSlave::readDraw(void *c, int n)
{
    int ret;
    do
    {
        ret = socketDraw->Read(c, n);

    } while ((ret < 0) && ((errno == EAGAIN) || (errno == EINTR)));
    return ret;
}

int coVRTcpSlave::sendDraw(const void *c, int n)
{
    int ret;
    do
    {
        ret = socketDraw->write(c, n);

    } while ((ret <= 0) && ((errno == EAGAIN) || (errno == EINTR)));
    return ret;
}

int coVRTcpSlave::send(const void *c, int n)
{
    int ret;
    do
    {
        ret = socket->write(c, n);

    } while ((ret <= 0) && ((errno == EAGAIN) || (errno == EINTR)));
    return ret;
}

void coVRTcpSlave::start()
{
    //start Client on remote host
    cerr << " Entering start method of coVRSlave.cpp" << endl;
    char cEntry[2000];
    char co[2000];
    static char hostname[256] = "";
    if (hostname[0] == '\0')
    {
        strcpy(hostname, "localhost");
        gethostname(hostname, sizeof(hostname));
    }
    std::string hn(hostname);

    std::string mi = coCoviseConfig::getEntry("COVER.MultiPC.MasterInterface");
    if (mi.empty())
    {
        mi = hn;
    }
    sprintf(cEntry, "COVER.MultiPC.Startup:%d", myID - 1);
    string command = coCoviseConfig::getEntry(cEntry);
    if (command.empty())
    {
        cerr << cEntry << " not found in config file" << endl;
        return;
    }
    if (strstr(command.c_str(), "startOpenCover"))
    {
        //connect to remote deamon and trigger startup of cover
        int remPort = 0;
        cerr << "Using remote daemon startup!" << endl;
        remPort = coCoviseConfig::getInt("port", "System.RemoteDaemon.Server", 31090);

        sprintf(cEntry, "COVER.MultiPC.Host:%d", myID - 1);
        std::string strHost = coCoviseConfig::getEntry(cEntry);
        if (strHost.empty())
        {
            cerr << cEntry << " not found in config file" << endl;
            return;
        }
        else
        {
            cerr << "Hostname is: " << strHost << endl;
        }
        Host *objHost = new Host(strHost.c_str());
        // verify myID value
        SimpleClientConnection *clientConn = new SimpleClientConnection(objHost, remPort);
        if (!clientConn)
        {
            cerr << "Creation of ClientConnection failed!" << endl;
            return;
        }
        else
        {
            cerr << "ClientConnection created!" << endl;
        }
        if (!(clientConn->is_connected()))
        {
            cerr << "Connection to RemoteDaemon on " << strHost << " failed!" << endl;
            return;
        }
        else
        {
            cerr << "Connection to RemoteDaemon on " << strHost << " established!" << endl;
        }

        // create command to send to remote daemon
        sprintf(co, "%s -c %d %s %d %s\n", command.c_str(), myID, mi.c_str(), port, hn.c_str());
        cerr << "Sending RemoteDaemon the message: " << co << endl;

        clientConn->getSocket()->write(co, (int)strlen(co));

        cerr << "Message sent!" << endl;
        cerr << "Closing connection objects!" << endl;

        delete objHost;
        delete clientConn;

        cerr << "Leaving Start-Method of coVRSlave " << endl;
    }
    else if (strncmp(command.c_str(), "covRemote", 9) == 0)
    {
        //connect to remote deamon and trigger startup of cover
        int remPort = 0;
        cerr << "Using covRemote startup!" << endl;
        remPort = coCoviseConfig::getInt("port", "System.covRemote.Server", 31809);
        std::string strHost = coCoviseConfig::getEntry("System.covRemote.Host");
        if (strHost.empty())
        {
            cerr << "covRemote.Host not found in covise.config file using localhost" << endl;
            strHost = "localhost";
            return;
        }
        else
        {
            cerr << "Hostname is: " << strHost << endl;
        }
        Host *objHost = new Host(strHost.c_str());
        // verify myID value
        for (int loop = 0; loop < 20; loop++)
        {
            SimpleClientConnection *clientConn = new SimpleClientConnection(objHost, remPort);
            if (!clientConn)
            {
                cerr << "Creation of ClientConnection failed!" << endl;
                return;
            }
            else
            {
                cerr << "ClientConnection created!  (try=" << loop << ")" << endl;
            }
            if (!(clientConn->is_connected()))
            {
                cerr << "Connection to covRemote on " << strHost << " failed!" << endl;
                return;
            }
            else
            {
                cerr << "Connection to covRemote on " << strHost << " established!" << endl;
            }

// create command to send to remote daemon
#ifndef NDEBUG
            sprintf(co, "OpenCOVER_debug %s -c %d %s %d %s\n", command.c_str() + 20, myID, mi.c_str(), port, hn.c_str());
#else
            sprintf(co, "%s -c %d %s %d %s\n", command.c_str() + 9, myID, mi.c_str(), port, hn.c_str());
#endif

            cerr << "Sending coVRemote the message: " << co << endl;
            int val = 0;
            val = clientConn->getSocket()->write(co, (int)strlen(co));
            //fprintf(stderr,"write val = %d\n",val);
            if (val > 0)
                break;
            delete clientConn;
            clientConn = NULL;
        }
        delete objHost;
    }
    else
    {
        cerr << "Using default ssh remote startup" << endl;
#ifdef WIN32
        sprintf(co, "%s -c %d %s %d %s", command.c_str(), myID, mi.c_str(), port, hn.c_str());
#else
		sprintf(co, "%s -c %d %s %d %s&", command.c_str(), myID, mi.c_str(), port, hn.c_str());
#endif
        cerr << "DEF starting: " << co << endl;
        if (system(co) == -1)
        {
            cerr << "coVRSlave::start: exec " << co << " failed" << endl;
        }
            cerr << "coVRSlave::started " << co << "" << endl;
    }
}

void coVRTcpSlave::accept()
{
    if (socket->acceptOnly(120) < 0)
    {
        cerr << "Client " << myID << "did not connect within 2 minutes" << endl;
    }
#ifdef DEBUG_MESSAGES
    int i = sizeof(port2);
    send((const char *)&i, sizeof(i));
    send((char *)&debugMessageCounter, sizeof(debugMessageCounter));
    read((char *)&i, sizeof(i));
#endif
    send((const char *)&port2, sizeof(port2));
    if (socketDraw->acceptOnly(120) < 0)
    {
        cerr << "Client " << myID << "did not connect within 2 minutes" << endl;
    }
    cerr << "connected to slave " << myID << endl;
}

#ifdef HAS_MPI

coVRMpiSlave::coVRMpiSlave(int ID, MPI_Comm appComm, MPI_Comm drawComm)
    : coVRSlave(ID)
    , appComm(appComm)
    , drawComm(drawComm)
{
}

coVRMpiSlave::~coVRMpiSlave()
{
}

int coVRMpiSlave::read(void *c, int n)
{
    MPI_Status status;
    MPI_Recv(c, n, MPI_BYTE, myID, coVRMSController::AppTag, appComm, &status);
    int count;
    MPI_Get_count(&status, MPI_BYTE, &count);
    return count;
}

int coVRMpiSlave::send(const void *c, int n)
{
    CO_MPI_SEND(const_cast<void *>(c), n, MPI_BYTE, myID, coVRMSController::AppTag, appComm);
    return n;
}

int coVRMpiSlave::readDraw(void *c, int n)
{
    MPI_Status status;
    MPI_Recv(c, n, MPI_BYTE, myID, coVRMSController::DrawTag, drawComm, &status);
    int count;
    MPI_Get_count(&status, MPI_BYTE, &count);
    return count;
}

int coVRMpiSlave::sendDraw(const void *c, int n)
{
    CO_MPI_SEND(const_cast<void *>(c), n, MPI_BYTE, myID, coVRMSController::DrawTag, drawComm);
    return n;
}

void coVRMpiSlave::start()
{
}

void coVRMpiSlave::accept()
{
}

int coVRMpiSlave::readMessage(Message *msg)
{
    const int headerSize = 4 * sizeof(int);

    int buffer[headerSize];

    int received = read(buffer, headerSize);

    if (received < headerSize)
        return received;

    int *bufferInt = (int *)buffer;
    msg->sender = bufferInt[0];
    msg->send_type = bufferInt[1];
    msg->type = bufferInt[2];

    msg->data = DataHandle(bufferInt[3]);

    return received + read(msg->data.accessData(), msg->data.length());
}

int coVRMpiSlave::sendMessage(const Message *msg)
{

    int header[4];
    int sent = 0;

    header[0] = msg->sender;
    header[1] = msg->send_type;
    header[2] = msg->type;
    header[3] = msg->data.length();

    sent = send(reinterpret_cast<char *>(&header[0]), 4 * sizeof(int));

    return sent + send(msg->data.data(), msg->data.length());
}
#endif
