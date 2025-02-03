/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#ifdef HAS_MPI
#include <mpi.h>
#endif

namespace covise
{
class Socket;
class Message;
}
#include <util/coTypes.h>

namespace vive
{
class VVCORE_EXPORT vvSlave
{
public:
    vvSlave(int ID);
    virtual ~vvSlave();

    int getID()
    {
        return myID;
    }

    virtual void start() = 0;
    virtual void accept() = 0;
    virtual int read(void *c, int n) = 0;
    virtual int send(const void *c, int n) = 0;
    virtual int readDraw(void *c, int n) = 0;
    virtual int sendDraw(const void *c, int n) = 0;

    virtual int readMessage(covise::Message *msg);
    virtual int sendMessage(const covise::Message *msg);

	virtual void sendMessage(const covise::UdpMessage* msg);
	virtual int readMessage(covise::UdpMessage* msg);

protected:
    int myID;
};

class VVCORE_EXPORT vvTcpSlave : public vvSlave
{
public:
    vvTcpSlave(int ID);
    virtual ~vvTcpSlave();
    virtual void start();
    virtual void accept();
    virtual int read(void *c, int n);
    virtual int send(const void *c, int n);
    virtual int readDraw(void *c, int n);
    virtual int sendDraw(const void *c, int n);

private:
    int port;
    int port2;
    covise::Socket *socket;
    covise::Socket *socketDraw;
};

#ifdef HAS_MPI
class VVCORE_EXPORT vvMpiSlave : public vvSlave
{
public:
    vvMpiSlave(int ID, MPI_Comm appComm, int drawRank, MPI_Comm drawComm);
    virtual ~vvMpiSlave();
    virtual void start();
    virtual void accept();
    virtual int read(void *c, int n);
    virtual int send(const void *c, int n);
    virtual int readDraw(void *c, int n);
    virtual int sendDraw(const void *c, int n);

    virtual int readMessage(covise::Message *msg);
    virtual int sendMessage(const covise::Message *msg);

private:
    MPI_Comm appComm;
    MPI_Comm drawComm;
    int drawRank = -1;
};
#endif
}
