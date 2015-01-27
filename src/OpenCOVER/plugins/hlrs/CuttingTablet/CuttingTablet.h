/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CUTTINGTABLET_H
#define CUTTINGTABLET_H
#include <sys/time.h>

#include <osg/Geode>
#include <cover/coVRPlugin.h>

class Slicer;
class Server;

class CuttingTablet : public opencover::coVRPlugin
{

public:
    CuttingTablet();
    virtual ~CuttingTablet();

    virtual bool init();
    virtual void preFrame();
    void key(int type, int keySym, int mod);

private:
    Server *server;
    Slicer *slicer;

    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::Group> root;

    osg::Vec3 oldPosition;
    osg::Vec3 oldNormal;

    struct timeval *t0, t1;

    int count;
    bool renderState;
};

#endif
