/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_PORT_H
#define _INV_PORT_H

//
// defines
//
#define MAXDATALEN 500
#define MAXTOKENS 25
#define MAXHOSTLEN 20
#define MAX_REPLACE 32768

#include <qwidget.h>

class InvMsgManager : public QObject
{
    Q_OBJECT

public:
    InvMsgManager();
    ~InvMsgManager();

public slots:
    void dataReceived(int);
};

#endif
