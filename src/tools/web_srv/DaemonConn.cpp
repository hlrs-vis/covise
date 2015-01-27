/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "DaemonConn.h"
#include "../daemon/coCovdMsg.h"
#include <sys/stat.h>
////////////////////////////////////////////////////////////////////////

void DaemonConn::addServerConn(coCommunicator &comm)
{

    is_bad_ = true;
    conn_ = NULL;

    const char *name = "0.0.0.0";

    // fetch address + create coAddr
    coNetAddr *addr = coMkAddr::create(name);
    if (!addr)
    {
        cout << "Cannot convert '" << name << "to valid address"
             << endl;
        return;
    }

    const char *addInfo = "TCP,SERVER, 43000";

    coLayer *layer = coMkLayer::create(addr, addInfo, 180.0);
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
    conn_ = coMkConn::create(layer, NULL, NULL);

    if ((!conn_) || (conn_->isBad()))
    {
        if (conn_)
            cout << conn_->getError() << endl;
        else
            cout << "could not create Connection" << endl;
        return;
    }

    // add to communicator
    comm.addConnection(conn_);

    is_bad_ = false;
}

bool
DaemonConn::isBad()
{
    if (!is_bad_ && conn_)
    {
        return (conn_->isBad());
        /*********************
       coMsg *msg = conn_->recv(0.3);
       if( msg )
       {
              coMsg::Header hdr = msg->getHeader();
        cout << "\n websrv.print \n";

        cout <<   "coMsg: msgSize   = " << hdr.msgSize;
        cout << "      msgID(" << hdr.msgID;
        cout << "), Usr(" << hdr.userData;
        cout << "), Reply(" << hdr.inReplyTo;
      cout << "), ID(" << hdr.msgID << ")";
      cout << "\n       msgType   = " << hdr.msgType << " (COVD)";
      cout << "\n       subType   = " << hdr.subType;

      cout << "\n" << msg->getBody();
      cout << endl;
      comm_.destroy(msg);
      return true;
      }
      ****************************/
    }

    return is_bad_;
}

void DaemonConn::sendUIFMsg(const char *text, coCommunicator &comm)
{
    sendMsg(coMsg::UIF, -2, text, comm);
}

void
DaemonConn::getPartnerHosts(char **partner_hosts, int *num_partners)
{
    char host_file[1024], *filebuf;
    char line[1024];
    int size;

    sprintf(host_file, "%s/.partners", getenv("COVISEDIR"));
    FILE *f = openFile(host_file, &filebuf, &size);

    *num_partners = 0;

    if (f)
    {
        while (fgets(line, size, f))
        {
            int len = strlen(line);
            partner_hosts[(*num_partners)] = new char[len + 1];
            strcpy(partner_hosts[(*num_partners)], line);
            partner_hosts[(*num_partners)++][len - 1] = '\0';
        }

        delete[] filebuf;
    }
}

FILE *
DaemonConn::openFile(const char *filename, char **file_buf, int *size)
{
    struct stat st;
    FILE *f = NULL;
    *size = 0;

    if (!stat(filename, &st) && (f = fopen(filename, "r")) != NULL)
    {
        *file_buf = new char[st.st_size + 1];
        memset(*file_buf, 0, st.st_size + 1);
        *size = st.st_size;
    }
    return f;
}

char *DaemonConn::getFile(const char *map_file)
{
    char *buf = NULL;
    int size;

    FILE *f = openFile(map_file, &buf, &size);

    if (f)
    {
        fread(buf, size, 1, f);
        fclose(f);
    }
    return buf;
}

void DaemonConn::sendLaunchMsg()
{
    sendMsg(coMsg::COVD, coCovdMsg::LAUNCH, " ", comm_);
    sleep(4);
}

bool DaemonConn::sendLoadMsg(const char *map_file)
{
    if (!map_file)
    {
        return true;
    }

    char *filebuf = getFile(map_file);
    if (!filebuf)
    {
        return false;
    }

    char *msg_body = new char[strlen(map_file) + strlen(filebuf) + 2];
    sprintf(msg_body, "%s\n%s", map_file, filebuf);
    sendMsg(coMsg::COVD, coCovdMsg::LOAD_MAP, msg_body, comm_);

    delete[] filebuf;
    delete[] msg_body;
    return true;
}

void DaemonConn::sendQuitMsg()
{
    sendMsg(coMsg::COVD, coCovdMsg::QUIT, " ", comm_);
}

void
DaemonConn::askState(char *answer)
{
    sendMsg(coMsg::COVD, coCovdMsg::INFO, " ", comm_);

    coMsg *msg = comm_.recv(5.);
    if (msg == NULL)
    {
        strcpy(answer, "WAITING");
    }
    else
    {
        strcpy(answer, msg->getBody());
        comm_.destroy(msg);
    }
    cout << "WebSrv:: askState answers " << answer << endl;
}

void DaemonConn::sendMsg(int32_t type, int32_t subType, const char *text, coCommunicator &comm)
{
    int len = strlen(text) + 1;
    int body_len;
    const char *body;
    char *tmp = NULL;

    if (len % 4 != 0)
    {
        body_len = ((len / 4) + 1) * 4;
        tmp = new char[body_len];
        memset(tmp, 0, body_len);
        strcpy(tmp, text);
        body = tmp;
    }
    else
    {
        body = text;
        body_len = len;
    }

    coMsg *msg = new coMsg(type, subType, 0, 0, 1, body_len, body);
    comm.send(msg, coMsg::recvID(1, 0));

    cout << "WebSrv: send " << body_len << " Bytes" << endl;

    comm.destroy(msg);
    if (tmp)
    {
        delete[] tmp;
    }
}

/////////////////////////////////////////////////////////////////////////////////

DaemonConn::~DaemonConn()
{
    comm_.removeConnection(comm_.getIDConn(comm_.getConnID(0)), true);
}

DaemonConn::DaemonConn()
    : comm_("WebDaemonComm")
{

    //   signal(SIGPIPE,SIG_IGN);  // ignore 'broken pipe'

    addServerConn(comm_);
}
