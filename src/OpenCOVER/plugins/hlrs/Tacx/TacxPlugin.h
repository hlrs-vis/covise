/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#ifdef WIN32
#include <windows.h>
//#include "lusb0_usb.h"
#include <conio.h>
#else
//#include <usb.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cover/coVRNavigationManager.h>
#include <cover/coVRPlugin.h>
#include <Tacx.h>



#define MAX_CALLSIGN_LEN        8



#ifndef TacxPlugin_H
#define TacxPlugin_H



class PLUGINEXPORT TacxPlugin : public opencover::coVRPlugin, public opencover::coVRNavigationProvider
{
public:
    TacxPlugin();
    ~TacxPlugin();
    bool update();
    osg::Vec3d getNormal();
    void Initialize();
private:
    float stepSizeUp;
    float stepSizeDown;
    bool init();
    float getYAccelaration();
    osg::Matrix getMatrix();
    virtual void setEnabled(bool);
    Tacx *tacx=nullptr;
    float speed=0.0;
    float wheelBase = 0.97;
    osg::Matrix TransformMat;
    osg::Matrix TacxPos;
};

#endif /* TacxPlugin_H */
