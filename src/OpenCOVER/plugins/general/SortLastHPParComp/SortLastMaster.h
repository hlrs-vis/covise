/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SORTLASTMASTER_H
#define SORTLASTMASTER_H

#include "SortLastImplementation.h"

#include <pcapi.h>
#include <GL/gl.h>

#include <cassert>

class SortLastMaster : public SortLastImplementation
{
public:
    SortLastMaster(const std::string &nodename, int session);
    virtual ~SortLastMaster();

    virtual bool init();
    virtual void preSwapBuffers(int windowNumber);

    virtual bool initialiseAsMaster();
    virtual bool initialiseAsSlave()
    {
        assert(0);
    }
    virtual bool createContext(const std::list<std::string> &hostlist, int groupIdentifier);

private:
    void initTextures(PCchannel *frameBuffer, PCchannel *depthBuffer);
    GLuint makeShader(GLuint program, GLuint type, const char *source);

    void compositeSimpleReadback();
    void compositeSimpleShader();

    PCcontext context;
    PCchannel frameBuffer;
    PCchannel depthBuffer;

    int frameLeft, frameBottom, frameWidth, frameHeight;
    int channelBottom, channelLeft, channelWidth, channelHeight;

    GLuint textures[2];

    std::list<std::string> hostlist;
    std::string hostname;

    int frameCtr;
    int session;

    inline void callPcFunc(PCerr error, const char *location, int line);

    bool initPending;
};

#endif // SORTLASTMASTER_H
