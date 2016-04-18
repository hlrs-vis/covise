/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Uwe Woessner (c) 2013
**   <woessner@hlrs.de.de>
**   03/2013
**
**************************************************************************/

#include "coverconnection.hpp"
#include "../gui/lodsettings.hpp"
#include "../../PluginUtil/PluginMessageTypes.h"

#include "../../plugins/hlrs/OddlotLink/oddlotMessageTypes.h"
#include <net/covise_host.h>
#include <net/message_types.h>
#include <net/message.h>

// Data //

COVERConnection *COVERConnection::inst = NULL;

//################//
// CONSTRUCTOR    //
//################//

COVERConnection::COVERConnection()
{
    inst = this;
    m_periodictimer = new QTimer;
    QObject::connect(m_periodictimer, SIGNAL(timeout()), this, SLOT(processMessages()));
    m_periodictimer->start(1000);
    toCOVERSN = NULL;
    toCOVER = NULL;
    msg = new covise::Message;
}

COVERConnection::~COVERConnection()
{
    inst = NULL;
    delete toCOVERSN;
    delete m_periodictimer;
}

void COVERConnection::closeConnection()
{
    delete toCOVER;
    toCOVER=NULL;
    LODSettings::instance()->setConnected(false);
}

void COVERConnection::send(covise::TokenBuffer &tb)
{
    if (toCOVER != NULL)
    {
        covise::Message m(tb);
        m.type = opencover::PluginMessageTypes::HLRS_Oddlot_Message;
        toCOVER->send_msg(&m);
    }
}

void COVERConnection::resizeMap(float x, float y, float width, float height)
{
    if(toCOVER!=NULL)
    {
        covise::TokenBuffer tb;
        tb << MSG_GetMap;
        tb << x;
        tb << y;
        tb << width;
        tb << height;
        send(tb);
    }
}
//------------------------------------------------------------------------
void COVERConnection::processMessages()
//------------------------------------------------------------------------
{
    if(toCOVER == NULL)
    {
        if(LODSettings::instance()->doConnect())
        {
            covise::Host *h = new covise::Host(LODSettings::instance()->hostname.toUtf8().constData());
            toCOVER = new covise::ClientConnection(h ,LODSettings::instance()->getPort(),0,0,0,0.0000000001);
            if(toCOVER->is_connected())
            {
                LODSettings::instance()->setConnected(true);
                struct linger linger;
                linger.l_onoff = 0;
                linger.l_linger = 0;
                setsockopt(toCOVER->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

                toCOVERSN = new QSocketNotifier(toCOVER->get_id(NULL), QSocketNotifier::Read);
                QObject::connect(toCOVERSN, SIGNAL(activated(int)),
                                 this, SLOT(processMessages()));
            }
            else
            {
                closeConnection();
            }
        }
    }
    
    while (toCOVER && toCOVER->check_for_input(0.0001f))
    {
            if (toCOVER->recv_msg(msg))
            {
                if (msg)
                {
                    if (handleClient(msg))
                    {
                        return; // we have been deleted, exit immediately
                    }
                }
            }
    }
}

bool COVERConnection::waitForMessage(covise::Message **m)
{
    while (toCOVER->recv_msg(msg))
    {
        if (msg)
        {
            if(msg->type == opencover::PluginMessageTypes::HLRS_Oddlot_Message)
            {
                *m = msg;
                return true;
            }
            if (handleClient(msg))
            {
                return false; // we have been deleted, exit immediately
            }
        }
    }
}
//------------------------------------------------------------------------
bool COVERConnection::handleClient(covise::Message *msg)
//------------------------------------------------------------------------
{
    if((msg->type == covise::COVISE_MESSAGE_SOCKET_CLOSED) || (msg->type == covise::COVISE_MESSAGE_CLOSE_SOCKET))
    {
        closeConnection();
        return true; // we have been deleted, exit immediately
    }
    covise::TokenBuffer tb(msg);
    switch (msg->type)
    {
    case opencover::PluginMessageTypes::HLRS_Oddlot_Message:
        {
            int type;
            tb >> type;
            switch (type)
            {

            case MSG_GetHeight:
                {
                    std::cerr << "this message should not arrive here, oddlot should wait for this reply " << msg->type  << std::endl;
                }
                break;
            }
            //}
        }
        break;
    default:
        {
            if (msg->type > 0)
                std::cerr << "CoverConnection::handleClient err: unknown COVISE message type " << msg->type  << std::endl;
        }
        break;
    }
    return false;
}