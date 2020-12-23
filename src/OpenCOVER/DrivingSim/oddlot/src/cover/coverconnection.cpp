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
#include "../gui/projectwidget.hpp"
#include "../graph/topviewgraph.hpp"
#include "../graph/graphview.hpp"
#include "../mainwindow.hpp"
#include "ui_coverconnection.h"

// Data //

COVERConnection *COVERConnection::inst = NULL;

void COVERConnection::okPressed()
{
    port = ui->portSpinBox->value();
    hostname = ui->hostnameEdit->text();
}
//################//
// CONSTRUCTOR    //
//################//

COVERConnection::COVERConnection()
    : ui(new Ui::COVERConnection)
{
    inst = this;
    ui->setupUi(this);
    //connect(this, SIGNAL(accepted()), this, SLOT(okPressed()));
#ifdef WIN32
    char *pValue;
    size_t len;
    errno_t err = _dupenv_s(&pValue, &len, "ODDLOTDIR");
    if (err || pValue == NULL || strlen(pValue) == 0)
        err = _dupenv_s(&pValue, &len, "COVISEDIR");
    if (err)
        pValue="";
    QString covisedir = pValue;
#else
    QString covisedir = getenv("ODDLOTDIR");
    if (covisedir == "")
        covisedir = getenv("COVISEDIR");
#endif
    QString dir = covisedir + "/share/covise/icons/";
    coverConnected = new QIcon(dir + "cover_connected.png");
    coverDisconnected = new QIcon(dir + "cover_disconnected.png");
    //Timer init for 1000 ms interrupt => call processMessages()
    m_periodictimer = new QTimer;
    QObject::connect(m_periodictimer, SIGNAL(timeout()), this, SLOT(processMessages()));
    m_periodictimer->start(1000);
    //monitoring activity to file descriptor
    toCOVERSN = NULL;
    //connect to covise
    toCOVER = NULL;
    msg = new covise::Message;
    mainWindow = NULL;
    connected = false;
    inst = this;
}

void COVERConnection::setMainWindow(MainWindow *mw)
{
    mainWindow = mw;
}

COVERConnection::~COVERConnection()
{
    inst = NULL;
    delete toCOVERSN;
    delete m_periodictimer;
    delete ui;
    delete coverConnected;
    delete coverDisconnected;
}

bool COVERConnection::isConnected()
{
    //return (ui->connectedState->isChecked());
    return connected;
}

void COVERConnection::setConnected(bool c)
{
    //ui->connectedState->setChecked(c);
    connected = c;
    if(connected)
    {
        mainWindow->updateCOVERConnectionIcon(*coverConnected);
        ui->Instruction->setText("");
    }
    else {
        mainWindow->updateCOVERConnectionIcon(*coverDisconnected);
        closeConnection();
    }
}

int COVERConnection::getPort()
{
    return (ui->portSpinBox->value());
}

void COVERConnection::closeConnection()
{
    delete toCOVER;
    toCOVER=NULL;
    //LODSettings::instance()->setConnected(false);
    //setConnected(false);
}

void COVERConnection::send(covise::TokenBuffer &tb)
{
    if (toCOVER != NULL)
    {
        covise::Message m(tb);
        m.type = opencover::PluginMessageTypes::HLRS_Oddlot_Message;
        toCOVER->sendMessage(&m);
    }
}

void COVERConnection::resizeMap(float x, float y, float width, float height)
{
    if(toCOVER!=NULL)
    {
        int xRes = 1024;
        int yRes = 768;
        covise::TokenBuffer tb;
        tb << MSG_GetMap;
        tb << x;
        tb << y+height;
        tb << width;
        tb << -height;
        tb << xRes;
        tb << yRes;
        send(tb);
    }
}
//------------------------------------------------------------------------
void COVERConnection::processMessages()
//------------------------------------------------------------------------
{
    if(toCOVER == NULL)
    {
        //UI
        if(/*LODSettings::instance()->*/isConnected())
        {
            std::string hostname = /*LODSettings::instance()->*/(this->hostname).toStdString();
            if (hostname.empty())
                hostname = "localhost";
            covise::Host *h = new covise::Host(hostname.c_str());
            toCOVER = new covise::ClientConnection(h ,/*LODSettings::instance()->*/getPort(),0,0,0,0.0000000001);
            if(toCOVER->is_connected())
            {
                /*LODSettings::instance()->*/setConnected(true);
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
                //closeConnection();
                setConnected(false);
                mainWindow->getFileSettings()->show();
                mainWindow->getFileSettings()->getTabWidget()->setCurrentWidget(this);
                ui->Instruction->setText("\n\nConnection not possible. Please check the connection details."
                                         "\n\nInstructions:"
                                         "\n- open tabletUI and openCOVER in seperate Terminals"
                                         "\n- load plugin \"OddlotLink\" in tabletUI"
                                         "\n- open the project in opencover"
                                         "\n- start oddlot"
                                         "\n- create a new project in oddlot and click the COVERConnection Button in the right corner");
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
    return true;
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
            case MSG_GetMap:
                {
                    float x,y,width,height;
                    int xRes,yRes;
                    tb >> x;
                    tb >> y;
                    tb >> width;
                    tb >> height;
                    tb >> xRes;
                    tb >> yRes;
                    const char *buf = tb.getBinary(xRes*yRes);
                    if(mainWindow->getActiveProject())
                    {
                    mainWindow->getActiveProject()->getTopviewGraph()->getView()->setMap(x,y,width,height,xRes,yRes,buf);
                    }
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
