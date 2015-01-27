/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************
 ** window.cpp
 ** 2004-01-29, Matthias Feurer
 ****************************************************************************/

#include <qstring.h>
#include "window.h"
#include "channel.h"

Window::Window()
{
    index = 0;
    name = "";
    softPipeNo = 0;
    origin[0] = 0;
    origin[1] = 0;
    width = 0;
    height = 0;
    channelMap = ChannelMap();
}

void Window::setIndex(int i)
{
    index = i;
}

void Window::setName(QString s)
{
    name = s;
}

void Window::setSoftPipeNo(int i)
{
    softPipeNo = i;
}

void Window::setOriginX(int i)
{
    origin[0] = i;
}

void Window::setOriginY(int i)
{
    origin[1] = i;
}

void Window::setWidth(int i)
{
    width = i;
}

void Window::setHeight(int i)
{
    height = i;
}

void Window::setChannelMap(ChannelMap cm)
{
    channelMap = cm;
}

void Window::addChannel(QString s, Channel c)
{
    channelMap[s] = c;
}

int Window::getIndex()
{
    return index;
}

QString Window::getName()
{
    return name;
}

int Window::getSoftPipeNo()
{
    return softPipeNo;
}

int Window::getOriginX()
{
    return origin[0];
}

int Window::getOriginY()
{
    return origin[1];
}

int Window::getWidth()
{
    return width;
}

int Window::getHeight()
{
    return height;
}

ChannelMap *Window::getChannelMap()
{
    return &channelMap;
}

int Window::getNumChannels()
{
    return channelMap.count();
}
