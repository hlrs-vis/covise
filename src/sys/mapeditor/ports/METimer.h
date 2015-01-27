/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_TIMER_H
#define ME_TIMER_H

class MEParameterPort;
class MENode;

//================================================
class METimer
//================================================
{

public:
    METimer(MEParameterPort *);
    ~METimer();

    enum playMode
    {
        REVERSE = -2,
        BACKWARD,
        STOP,
        FORWARD,
        PLAY
    };

    bool isActive()
    {
        return active;
    };
    int getAction()
    {
        return playMode;
    };
    void setActive(bool mode)
    {
        active = mode;
    };
    void setAction(int type)
    {
        playMode = type;
    };
    MEParameterPort *getPort()
    {
        return port;
    };

private:
    int playMode;
    bool active;
    MEParameterPort *port;
};
#endif
