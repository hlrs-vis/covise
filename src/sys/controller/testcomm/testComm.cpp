/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include <sys/types.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#ifdef _WIN32
#include <process.h>
#endif
#ifndef _WIN32
#include <inttypes.h>
#endif
#include <comm/transport/coMkAddr.h>
#include <comm/transport/coMkLayer.h>
#include <comm/msg/coSendBuffer.h>
#include <comm/msg/coRecvBuffer.h>
#include <comm/msg/coMsg.h>
#include <comm/msg/coPrintMsg.h>
#include <comm/msg/coCommMsg.h>
#include <comm/logic/coCommunicator.h>
#include <comm/logic/coMkConn.h>
#include <sys/time.h>

#ifdef _STANDARD_C_PLUS_PLUS
#define istrstream std::istringstream
using std::flush;
#endif

char selectMenu(const char *const *menu, const char *allowed)
{
    char ch;
    do
    {
        const char *const *menuEntry = menu;
        cout << "=====================================" << endl;
        while (*menuEntry)
        {
            cout << *menuEntry++ << endl;
        }
        cout << "=====================================" << endl;
        cout << "Enter key: " << flush;
        cin >> ch;
    } while (!strchr(allowed, ch));
    return ch;
}

////////////////////////////////////////////////////////////////////////

void addConn(coCommunicator &comm)
{
    char addInfo[128];
    char name[128];
    cout << "connect host / 'x'=open server : " << flush;
    cin >> name;

    if (strcmp(name, "x") == 0)
        strcpy(name, "0.0.0.0");

    // fetch address + create coAddr
    coNetAddr *addr = coMkAddr::create(name);
    if (!addr)
    {
        cout << "Cannot convert '" << name << "to valid address"
             << endl;
        return;
    }

    // fetch addInfo + create Layer
    if (strcmp(name, "0.0.0.0") != 0)
    {
        cout << "additional info     : " << flush;
        cin >> addInfo;
    }
    else
        addInfo[0] = '\0';

    if (addInfo[0] == 'x')
        addInfo[0] = '\0';

    coLayer *layer = coMkLayer::create(addr, addInfo, 60.0);
    coNetAddr::destroy(addr);

    if (!layer)
    {
        cout << "cannot convert \"" << name
             << "\" to valid address" << endl;
        return;
    }

    if (layer->kind() == coLayer::SERVER)
    {
        cout << "Server, addInfo = \"" << layer->getAddInfo() << "\""
             << endl;
        layer->accept(60.0);
    }

    if (layer->isBad())
    {
        cout << layer->getError() << endl;
        coLayer::destroy(layer);
        return;
    }

    // open connection
    coConn *conn = coMkConn::create(layer, NULL, NULL);

    if ((!conn) || (conn->isBad()))
    {
        if (conn)
            cout << conn->getError() << endl;
        else
            cout << "could not create Connection" << endl;
        return;
    }

    // add to communicator
    comm.addConnection(conn);
}

////////////////////////////////////////////////////////////////////////

void startCovise(coCommunicator &comm)
{
    char addInfo[128];
    char name[128];

    // fetch address + create coAddr
    coNetAddr *addr = coMkAddr::create("0.0.0.0");
    if (!addr)
    {
        cout << "Cannot convert to valid address"
             << endl;
        return;
    }

    addInfo[0] = '\0';

    coLayer *layer = coMkLayer::create(addr, addInfo, 60.0);
    coNetAddr::destroy(addr);

    if (!layer)
    {
        cout << "cannot convert \"" << name
             << "\" to valid address" << endl;
        return;
    }

    if (layer->kind() == coLayer::SERVER)
    {
        cout << "Server, addInfo = \"" << layer->getAddInfo() << "\""
             << endl;
        const char *info = layer->getAddInfo();
        const char *t;
        t = strstr(info, ",");
        t = strstr(t + 1, ",");

        char start_cmd[128];
        sprintf(start_cmd, "covise -d %s ROOM_NAME&", t + 1);
        cerr << "please start in another shell: " << start_cmd << endl;
        //system( start_cmd);
        layer->accept(60.0);
    }

    if (layer->isBad())
    {
        cout << layer->getError() << endl;
        coLayer::destroy(layer);
        return;
    }

    // open connection
    coConn *conn = coMkConn::create(layer, NULL, NULL);

    if ((!conn) || (conn->isBad()))
    {
        if (conn)
            cout << conn->getError() << endl;
        else
            cout << "could not create Connection" << endl;
        return;
    }

    // add to communicator
    comm.addConnection(conn);
}

////////////////////////////////////////////////////////////////////////

void addHost(coCommunicator &comm)
{
    char hostname[128];
    cout << "connect host: " << flush;
    cin >> hostname;

    char partner_msg[128];
    sprintf(partner_msg, "ADD_HOST\n%s\npartner\nNONE\n%d\n20\n \n",
            hostname, 8);

    coMsg *msg = new coMsg(coMsg::UIF, -2, 0, 0, 1, strlen(partner_msg) + 1, partner_msg);
    comm.send(msg, coMsg::recvID(1, 0));

    comm.destroy(msg);
    msg = comm.recv(0.5, 1);

    cout << "answer from controller: " << endl;
    cout << msg->getBody() << endl;
}

/////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////

void addPartner(coCommunicator &comm)
{
    char hostname[128];
    cout << "connect host: " << flush;
    cin >> hostname;

    char partner_msg[128];
    sprintf(partner_msg, "ADD_PARTNER\n%s\npartner\nNONE\n%d\n20\n \n",
            hostname, 8);

    coMsg *msg = new coMsg(coMsg::UIF, -2, 0, 0, 1, strlen(partner_msg) + 1, partner_msg);
    comm.send(msg, coMsg::recvID(1, 0));

    comm.destroy(msg);
    msg = comm.recv(0.5, 1);
    cout << "answer from controller: " << endl;
    cout << msg->getBody() << endl;
}

/////////////////////////////////////////////////////////////////////////////////

void removeConn(coCommunicator &comm)
{
    cout << "Connection ID: " << flush;
    int idx;
    cin >> idx;
    comm.removeConnection(idx, true);
}

/////////////////////////////////////////////////////////////////////////////////

void addRecv(coCommunicator &comm)
{
    cout << "Receiver: " << flush;
    char recvName[128];
    cin >> recvName;
    comm.addRecv(recvName);
}

/////////////////////////////////////////////////////////////////////////////////

void removeRecv(coCommunicator &comm)
{
    cout << "Receiver: " << flush;
    int recvID;
    cin >> recvID;
    comm.removeRecvID(recvID);
}

/////////////////////////////////////////////////////////////////////////////////

void sendTest(coCommunicator &comm)
{
    cout << "send to: " << flush;

    char buffer[128];
    int recvID, node = 0;
    cin >> buffer;
    cout << "'" << buffer << "'" << endl;
    istrstream str(buffer);
    if (strchr(buffer, ':'))
    {
        char x;
        str >> recvID >> x >> node;
    }
    else
    {
        str >> recvID;
    }

    coMsg *msg = new coMsg(-1, -2, 0, -1, 0);
    cout << "Msg-ID = " << comm.send(msg, coMsg::recvID(recvID, node)) << endl;
    delete msg;
}

/////////////////////////////////////////////////////////////////////////////////

void send777(coCommunicator &comm)
{
    cout << "send to: " << flush;

    char buffer[128];
    int recvID, node = 0;
    cin >> buffer;
    cout << "'" << buffer << "'" << endl;
    istrstream str(buffer);
    if (strchr(buffer, ':'))
    {
        char x;
        str >> recvID >> x >> node;
    }
    else
    {
        str >> recvID;
    }

    coMsg *msg = new coMsg(777, 0, 0, -1, 0);
    cout << "Msg-ID = " << comm.send(msg, coMsg::recvID(recvID, node)) << endl;
    delete msg;
}

/////////////////////////////////////////////////////////////////////////////////

void wait777(coCommunicator &comm)
{
    cout << "wait for msg 777 from: " << flush;
    int from;
    cin >> from;
    coMsg *msg = comm.waitMsg(60.0, 10, from, 777, -1);
    cout << coPrintMsg(msg) << endl;
    coCommunicator::destroy(msg);
}

/////////////////////////////////////////////////////////////////////////////////

void longTest(coCommunicator &comm)
{
    int length;
    cout << "length [kB]: " << flush;
    cin >> length;
    length *= 1024;

    cout << "send to: " << flush;

    char buffer[128];
    int recvID, node = 0;
    cin >> buffer;
    cout << "'" << buffer << "'" << endl;
    istrstream str(buffer);
    if (strchr(buffer, ':'))
    {
        char x;
        str >> recvID >> x >> node;
    }
    else
    {
        str >> recvID;
    }

    char *dummy = new char[length];

    coMsg *msg = new coMsg(-1, -2, 0, 0, 1, length, dummy);
    cout << "Msg-ID = " << comm.send(msg, coMsg::recvID(recvID, node)) << endl;
    delete dummy;
    delete msg;
}

/////////////////////////////////////////////////////////////////////////////////

void timeTest(coCommunicator &comm)
{
    int length;
    cout << "length [kB]: " << flush;
    cin >> length;
    length *= 1024;

    cout << "send to: " << flush;
    char buffer[128];
    int recvID, node = 0;
    cin >> buffer;
    cout << "'" << buffer << "'" << endl;
    istrstream str(buffer);
    if (strchr(buffer, ':'))
    {
        char x;
        str >> recvID >> x >> node;
    }
    else
    {
        str >> recvID;
    }

    struct timeval start, end;
    double t = 0.0;
    int n = 0;

    while ((t < 20) || (n < 10))
    {
        n++;
        coMsg *msg = new coMsg(-1, -2, 0, 0, length, 0, NULL);
        gettimeofday(&start, NULL);
        comm.send(msg, coMsg::recvID(recvID, node));
        delete msg;
        msg = NULL;
        do
            msg = comm.recv(60, 60);
        while (!msg);
        gettimeofday(&end, NULL);
        comm.destroy(msg);

        t += (end.tv_sec + 1e-6 * end.tv_usec) - (start.tv_sec + 1e-6 * start.tv_usec);

        if (length > 2048 * 1024)
            cout << "." << flush;
    }
    cout << endl;

    t /= n;

    cout << "Average time:         " << t << " s" << endl;
    cout << "Transfer Rate (eff):  " << length * 8 / t / 1024 / 1024 << " MBit/s" << endl;
    cout << "       incl. Header:  " << (length + 32) * 8 / t / 1024 / 1024 << " MBit/s" << endl;
    cout << "Messages/s:           " << 1.0 / t << endl;
}

/////////////////////////////////////////////////////////////////////////////////

void checkConn(coCommunicator &comm)
{
    cout << "xx numConn()   = " << comm.numConn() << endl;
    cout << "xx maxConnID() = " << comm.maxConnID() << endl;
    if (comm.getConnID(0))
        cout << "xx getConnID(0)     = " << *comm.getConnID(0) << endl;
    if (comm.getConnID(1))
        cout << "xx getConnID(1)     = " << *comm.getConnID(1) << endl;
    if (comm.getConnRecv(0))
        cout << "xx getConnRecv(0)   = " << *comm.getConnRecv(0) << endl;
    if (comm.getConnRecv(1))
        cout << "xx getConnRecv(1)   = " << *comm.getConnRecv(1) << endl;
    cout << endl;
}

/////////////////////////////////////////////////////////////////////////////////

void route(coCommunicator &comm)
{
    int from, to;
    cout << "from   = " << flush;
    cin >> from;
    cout << "to     = " << flush;
    cin >> to;
    comm.route(from, to);
}

/////////////////////////////////////////////////////////////////////////////////

void unroute(coCommunicator &comm)
{
    int from, to;
    cout << "from   = " << flush;
    cin >> from;
    cout << "to     = " << flush;
    cin >> to;
    comm.unroute(from, to);
}

/////////////////////////////////////////////////////////////////////////////////
/*
void canContact(coCommunicator &comm)
{
   int to;
   cout << "which   = " << flush;
   cin >> to;
   if (comm.canContact(to))
      cout << "We can contact " << to << endl;
   else
      cout << "We can NOT contact " << to << endl;
}
*/
/////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////

void server(coCommunicator &comm)
{
    cout << comm.openServer("0.0.0.0") << endl;
    comm.acceptServer(30.0);
}

/////////////////////////////////////////////////////////////////////////////////

void client(coCommunicator &comm)
{
    char addInfo[128];
    char name[128];
    cout << "connect host:    " << flush;
    cin >> name;
    cout << "additional Info: " << flush;
    cin >> addInfo;
    comm.addClient(name, addInfo, 60.0);
}

/////////////////////////////////////////////////////////////////////////////////

void transpond(coCommunicator &comm) // transponder for timings
{
    cout << "Transponder started: send 777 message to quit transponder" << endl;
    int i;
    char dummy[16 * 1024 * 1024];
    for (i = 0; i < 16 * 1024 * 1024; i += 16)
        dummy[i] = 'a';

    coMsg *msg = NULL;
    do
    {
        coCommunicator::destroy(msg);
        msg = comm.recv(0, 10000); // now only short timeouts
        if (msg)
        {
            coMsg *reply;
            coInt32 length = msg->getUser();
            if (length == 0)
                reply = msg->reply(111, 111, 0, 0, NULL);
            else
                reply = msg->reply(111, 111, 1, length, dummy);
            comm.send(reply);
            coMsg::destroy(reply);
        }

    } while ((!msg) || (msg->getType() != 777));

    coCommunicator::destroy(msg);
}

/////////////////////////////////////////////////////////////////////////////////

int main()
{

    //   signal(SIGPIPE,SIG_IGN);  // ignore 'broken pipe'

    char command[128];
    coMsg *msg;

    // +++++++++++++++++++ create a communicator
    long lpid = (long)getpid();
    sprintf(command, "TestComm-%ld", lpid);
    coCommunicator comm(command);
    // +++++++++++++++++++++++++++++++++++++++++

    command[0] = '\0';
    while (strcmp(command, "quit"))
    {
        cout << comm << "\n" << endl;
        cout << "Commands: addConn removeConn addRecv removeRecv\n"
             << "          route unroute checkConn\n\n startCovise addHost addPartner\n\n"
             << "          stat recv send time 777 wait777 transpond long\n"
             << endl;

        cout << "command : " << flush;
        cin >> command;

        if (strcmp(command, "addConn") == 0)
            addConn(comm);
        else if (strcmp(command, "removeConn") == 0)
            removeConn(comm);
        else if (strcmp(command, "addRecv") == 0)
            addRecv(comm);
        else if (strcmp(command, "removeRecv") == 0)
            removeRecv(comm);
        else if (strcmp(command, "route") == 0)
            route(comm);
        else if (strcmp(command, "unroute") == 0)
            unroute(comm);
        else if (strcmp(command, "checkConn") == 0)
            checkConn(comm);
        else if (strcmp(command, "addHost") == 0)
            addHost(comm);
        else if (strcmp(command, "startCovise") == 0)
            startCovise(comm);
        else if (strcmp(command, "addPartner") == 0)
            addPartner(comm);
        else if (strcmp(command, "server") == 0)
            server(comm);
        else if (strcmp(command, "client") == 0)
            client(comm);
        else if (strcmp(command, "recv") == 0)
        {
            msg = comm.recv(30, 1);
            cout << coPrintMsg(msg) << endl;
            if (msg)
                comm.destroy(msg);
            else
                cout << "No user message received" << endl;
        }
        else if (strcmp(command, "time") == 0)
            timeTest(comm);
        else if (strcmp(command, "send") == 0)
            sendTest(comm);
        else if (strcmp(command, "777") == 0)
            send777(comm);
        else if (strcmp(command, "wait777") == 0)
            wait777(comm);
        else if (strcmp(command, "transpond") == 0)
            transpond(comm);
        else if (strcmp(command, "long") == 0)
            longTest(comm);

        // always check for messages after command
        msg = comm.recv(0.5, 5);
        if (msg)
        {
            cout << coPrintMsg(msg) << endl;
            cout << "body: " << endl;
            cout << msg->getBody() << endl;
            comm.destroy(msg);
        }
    }

    return 0;
}
