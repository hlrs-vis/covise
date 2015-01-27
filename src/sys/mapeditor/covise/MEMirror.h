/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_MIRROR_H
#define CO_MIRROR_H

#include <QWidget>

class cyHost;

//================================================
class myButtonGroup : public QWidget
//================================================
{
    Q_OBJECT

public:
    myButtonGroup(cyHost *forHost, QWidget *parent = 0);

    cyHost *getMainHost()
    {
        return host;
    }

private:
    cyHost *host;
};

#endif
