/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Uwe Woessner (c) 2013
**   <woessner@hlrs.de.de>
**   04/2016
**
**************************************************************************/
#ifndef COVERConnection_HPP
#define COVERConnection_HPP
#include <net/covise_socket.h>
#include <net/covise_connect.h>
#include <qsocketnotifier.h>
#include <qtimer.h>
#include "../../plugins/hlrs/OddlotLink/oddlotMessageTypes.h"
#include <net/tokenbuffer.h>
#include "../mainwindow.hpp"

class COVERConnection: QObject
{
    
    Q_OBJECT
    //################//
    // FUNCTIONS      //
    //################//
    
private slots:
    void processMessages();
public:
    explicit COVERConnection();
    virtual ~COVERConnection();

    
    void resizeMap(float x, float y, float width, float height);
    
    void send(covise::TokenBuffer &tb);
    void setMainWindow(MainWindow *mw);

    static COVERConnection *instance()
    {
        if(inst==NULL)
            inst = new COVERConnection();
        return inst;
    };
    bool isConnected()
    {
        return(toCOVER!=NULL);
    }
    
    bool waitForMessage(covise::Message **m);

private:
    static COVERConnection *inst;
    QTimer *m_periodictimer;
    covise::ClientConnection *toCOVER;
    QSocketNotifier *toCOVERSN;
    
    void closeConnection();
    
    bool handleClient(covise::Message *msg);
    covise::Message *msg;
    MainWindow *mainWindow;


};

#endif // COVERConnection_HPP
