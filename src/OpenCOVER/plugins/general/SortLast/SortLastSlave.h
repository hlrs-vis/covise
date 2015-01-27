/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SORTLASTSLAVE_H
#define SORTLASTSLAVE_H

#include "SortLastImplementation.h"

#include <osgText/Text>
#include <osg/MatrixTransform>

#include <cassert>
#include <list>

class SortLastSlave : public SortLastImplementation
{

public:
    SortLastSlave(const std::string &nodename, int session);
    virtual ~SortLastSlave();

    virtual bool init();
    virtual void preSwapBuffers(int windowNumber);

    virtual bool initialiseAsMaster()
    {
        assert(0);
    }
    virtual bool initialiseAsSlave();
    virtual bool createContext(const std::list<std::string> &hostlist, int groupIdentifier);

private:
    int index;

    int width, height;

    std::vector<int> hostlist;

    GLubyte *pixels;
    GLfloat *depth;

    //int hostid;

    bool inFrame;

    osg::ref_ptr<osgText::Text> text;
    osg::ref_ptr<osg::MatrixTransform> group;
};

#endif // SORTLASTSLAVE_H
