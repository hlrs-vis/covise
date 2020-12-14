/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_VRSlave_H
#define CO_VRSlave_H

/*! \file
 \brief  cluster slave

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C) 2003
         High Performance Computing Center Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

#ifdef HAS_MPI
#include <mpi.h>
#endif

namespace covise
{
class Socket;
class Message;
}
#include <util/coTypes.h>

namespace opencover
{
class COVEREXPORT coVRSlave
{
public:
    coVRSlave(int ID);
    virtual ~coVRSlave();

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

class COVEREXPORT coVRTcpSlave : public coVRSlave
{
public:
    coVRTcpSlave(int ID);
    virtual ~coVRTcpSlave();
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
class COVEREXPORT coVRMpiSlave : public coVRSlave
{
public:
    coVRMpiSlave(int ID, MPI_Comm appComm, MPI_Comm drawComm);
    virtual ~coVRMpiSlave();
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
};
#endif
}
#endif
