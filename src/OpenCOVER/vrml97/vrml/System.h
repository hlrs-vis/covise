/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef SYSTEM_H
#define SYSTEM_H
//
//  System dependent utilities class
//

#include "vrmlexport.h"
#include "Viewer.h"
#include <sys/types.h>
#include <stdlib.h>

namespace vrml
{

class Player;
class VRMLEXPORT VrmlScene;
class VRMLEXPORT VrmlNodeViewpoint;

class VRMLEXPORT VrmlMessage
{
public:
    VrmlMessage(size_t len);
    virtual ~VrmlMessage();
    virtual void append(int &);
    virtual void append(float &);
    virtual void append(double &);
    virtual void append(const char *buf, size_t len);

    char *buf;
    size_t size;
    size_t pos;
};

class VRMLEXPORT System
{
private:
    bool correctBackFaceCulling;
    bool correctSpatializedAudio;
    int frameCounter;

protected:
    System();
    virtual ~System();

public:
    static System *the;

    virtual void update();

    virtual double time() = 0;
    double realTime();

    virtual int frame();

    virtual void error(const char *, ...);

    virtual void warn(const char *, ...);

    virtual void inform(const char *, ...);

    virtual void debug(const char *, ...);

    virtual bool loadUrl(const char *url, int np, char **parameters);

    virtual int connectSocket(const char *host, int port);

    virtual const char *httpHost(const char *url, int *port);
    virtual const char *httpFetch(const char *url);
    virtual const char *remoteFetch(const char *filename) = 0;

    virtual void removeFile(const char *fn);

    virtual void setBuiltInFunctionState(const char *fname, int val) = 0;
    virtual void setBuiltInFunctionValue(const char *fname, float val) = 0;
    virtual void callBuiltInFunctionCallback(const char *fname) = 0;

    virtual void setSyncMode(const char *mode) = 0;

    virtual bool isMaster() = 0;
    virtual void becomeMaster() = 0;

    virtual void setTimeStep(int ts) = 0; // set the timestep number for COVISE Animations
    virtual void setActivePerson(int p) = 0; // set the active Person

    virtual Player *getPlayer() = 0;

    virtual VrmlMessage *newMessage(size_t size) = 0;
    virtual void sendAndDeleteMessage(VrmlMessage *msg) = 0;
    virtual bool hasRemoteConnection() = 0;

    virtual float getSyncInterval()
    {
        return 0.2f;
    };
    virtual bool getHeadlight()
    {
        return true;
    };
    virtual void setHeadlight(bool enable) = 0;

    virtual float getLODScale()
    {
        return 1.f;
    }
    virtual float defaultCreaseAngle()
    {
        return 0.f;
    }

    virtual long getMaxHeapBytes()
    {
        return 8L * 1024 * 1024;
    };
    virtual bool getPreloadSwitch()
    {
        return true;
    };

    virtual void addViewpoint(VrmlScene *scene, VrmlNodeViewpoint *viewpoint) = 0;
    virtual bool removeViewpoint(VrmlScene *scene, const VrmlNodeViewpoint *viewpoint) = 0;
    virtual bool setViewpoint(VrmlScene *scene, const VrmlNodeViewpoint *viewpoint) = 0;

    virtual void setCurrentFile(const char *filename) = 0;

    virtual void setMenuVisibility(bool visible) = 0;
    virtual void createMenu() = 0;
    virtual void destroyMenu() = 0;

    enum NavigationType
    {
        NAV_NONE,
        NAV_WALK,
        NAV_EXAMINE,
        NAV_FLY,
        NAV_SCALE,
        NAV_DRIVE
    };

    virtual void setNavigationType(NavigationType nav) = 0;
    virtual void setNavigationStepSize(double stepsize) = 0;
    virtual void setNavigationDriveSpeed(double drivespeed) = 0;
    virtual void setNearFar(float near, float far) = 0;

    virtual double getAvatarHeight()
    {
        return 1.6;
    }
    virtual int getNumAvatars()
    {
        return 0;
    }
    virtual bool getAvatarPositionAndOrientation(int num, float pos[3], float ori[4])
    {
        (void)num;
        (void)pos;
        (void)ori;
        return false;
    }

    virtual bool getViewerPositionAndOrientation(float pos[3], float ori[4]) = 0;
    virtual bool getLocalViewerPositionAndOrientation(float pos[3], float ori[4]) = 0;
    virtual bool getViewerFeetPositionAndOrientation(float pos[3], float ori[4]) = 0;
    virtual bool getPositionAndOrientationFromMatrix(const double *M, float pos[3], float ori[4]) = 0;
    virtual void transformByMatrix(const double *M, float pos[3], float ori[4]) = 0;
    virtual void getInvBaseMat(double *M) = 0;
    virtual void getPositionAndOrientationOfOrigin(const double *M, float pos[3], float ori[4]) = 0;

    virtual bool loadPlugin(const char *name)
    {
        (void)name;
        return false;
    }

    virtual std::string getConfigEntry(const char *key)
    {
        (void)key;
        return std::string();
    }
    virtual bool getConfigState(const char *key, bool defaultVal)
    {
        (void)key;
        (void)defaultVal;
        return false;
    }

    virtual void enableCorrectBackFaceCulling(bool value)
    {
        correctBackFaceCulling = value;
    }
    virtual void enableCorrectSpatializedAudio(bool value)
    {
        correctSpatializedAudio = value;
    }
    virtual bool isCorrectBackFaceCulling()
    {
        return correctBackFaceCulling;
    }
    virtual bool isCorrectSpatializedAudio()
    {
        return correctSpatializedAudio;
    }

    virtual void storeInline(const char *nameBase, const Viewer::Object d_viewerObject)
    {
        (void)nameBase;
        (void)d_viewerObject;
    }
    virtual Viewer::Object getInline(const char *name)
    {
        (void)name;
        return 0L;
    }
    virtual void insertObject(Viewer::Object d_viewerObject, Viewer::Object sgObject)
    {
        (void)d_viewerObject;
        (void)sgObject;
    }

    virtual void saveTimestamp(const char *name);
};
}
#endif // SYSTEM_H
