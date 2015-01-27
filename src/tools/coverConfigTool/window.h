/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** window.h
 ** 2004-01-20, Matthias Feurer
 ****************************************************************************/

#ifndef WINDOW_H
#define WINDOW_H

#include <qstring.h>
#include <qmap.h>
#include <qptrlist.h>
#include "channel.h"

typedef QMap<QString, Channel> ChannelMap;

class Window
{

public:
    Window();
    void setIndex(int i);
    void setName(QString s);
    void setSoftPipeNo(int i);
    void setOriginX(int i);
    void setOriginY(int i);
    void setWidth(int i);
    void setHeight(int i);
    void setChannelMap(ChannelMap cm);
    void addChannel(QString s, Channel c);

    int getIndex();
    QString getName();
    int getSoftPipeNo();
    int getOriginX();
    int getOriginY();
    int getWidth();
    int getHeight();
    ChannelMap *getChannelMap();
    int getNumChannels();

private:
    int index;
    QString name;
    int softPipeNo;
    int origin[2];
    int width;
    int height;
    ChannelMap channelMap;
};
#endif // WINDOW_H
