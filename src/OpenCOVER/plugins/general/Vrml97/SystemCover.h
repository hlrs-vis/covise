/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
#ifndef SYSTEM_COVER_H
#define SYSTEM_COVER_H
//
//  System dependent utilities class, COVER version
//

#include <vrml97/vrml/System.h>
#include <cover/coVRFileManager.h>
#include <OpenVRUI/coMenu.h>
#include <util/coTypes.h>
#include <osg/Matrix>
#include <map>
#include <cover/coTabletUI.h>

namespace vrui
{
class coCheckboxMenuItem;
class coSubMenuItem;
class coRowMenu;
class coCheckboxGroup;
class coButtonMenuItem;
}

using namespace vrui;
using namespace vrml;
using namespace opencover;

class VRML97PLUGINEXPORT ViewpointEntry : public coMenuListener
{
public:
    ViewpointEntry(VrmlNodeViewpoint *, VrmlScene *);
    virtual ~ViewpointEntry();
    virtual void menuEvent(coMenuItem *button);
    void setMenuItem(coCheckboxMenuItem *aMenuItem);
    VrmlNodeViewpoint *getViewpoint()
    {
        return viewPoint;
    };
    coCheckboxMenuItem *getMenuItem()
    {
        return menuItem;
    };
    coTUIToggleButton *getTUIItem()
    {
        return tuiItem;
    };
    void activate();
    int entryNumber;

private:
    VrmlScene *scene;
    VrmlNodeViewpoint *viewPoint;
    coCheckboxMenuItem *menuItem;
    coTUIToggleButton *tuiItem;
};

class VRML97PLUGINEXPORT SystemCover : public System, public coMenuListener, public coTUIListener
{

public:
    SystemCover();
    virtual ~SystemCover()
    {
    }
    virtual double time();

    virtual const char *remoteFetch(const char *filename);

    bool loadUrl(const char *url, int np, char **parameters);

    virtual void setBuiltInFunctionState(const char *fname, int val);
    virtual void setBuiltInFunctionValue(const char *fname, float val);
    virtual void callBuiltInFunctionCallback(const char *fname);

    virtual void setSyncMode(const char *mode);

    virtual void setTimeStep(int ts); // set the timestep number for COVISE Animations
    virtual void setActivePerson(int p); // set the active Person

    virtual bool isMaster();
    virtual void becomeMaster();

    virtual VrmlMessage *newMessage(size_t size);
    virtual void sendAndDeleteMessage(VrmlMessage *msg);
    virtual bool hasRemoteConnection();

    virtual Player *getPlayer();

    virtual long getMaxHeapBytes();
    virtual bool getHeadlight();
    virtual void setHeadlight(bool enable);
    virtual bool getPreloadSwitch();
    virtual float getSyncInterval();

    virtual void addViewpoint(VrmlScene *scene, VrmlNodeViewpoint *viewpoint);
    virtual bool removeViewpoint(VrmlScene *scene, const VrmlNodeViewpoint *viewpoint);
    virtual bool setViewpoint(VrmlScene *scene, const VrmlNodeViewpoint *viewpoint);

    virtual void setCurrentFile(const char *filename);

    virtual void setMenuVisibility(bool visible);
    virtual void createMenu();
    virtual void destroyMenu();

    virtual void setNavigationType(NavigationType nav);
    virtual void setNavigationStepSize(double size);
    virtual void setNavigationDriveSpeed(double speed);
    virtual void setNearFar(float near, float far);

    virtual double getAvatarHeight();
    virtual int getNumAvatars();
    virtual bool getAvatarPositionAndOrientation(int num, float pos[3], float ori[4]);

    virtual bool getViewerPositionAndOrientation(float pos[3], float ori[4]);
    virtual bool getLocalViewerPositionAndOrientation(float pos[3], float ori[4]);

    virtual bool getViewerFeetPositionAndOrientation(float pos[3], float ori[4]);
    virtual bool getPositionAndOrientationFromMatrix(const double *M, float pos[3], float ori[4]);
    bool getPositionAndOrientationFromMatrix(const osg::Matrix &mat, float pos[3], float ori[4]);
    virtual void transformByMatrix(const double *M, float pos[3], float ori[4]);
    virtual void getInvBaseMat(double *M);
    virtual void getPositionAndOrientationOfOrigin(const double *M, float pos[3], float ori[4]);

    virtual std::string getConfigEntry(const char *key);
    virtual bool getConfigState(const char *key, bool defaultVal);

    virtual void storeInline(const char *name, const Viewer::Object d_viewerObject);
    virtual Viewer::Object getInline(const char *name);
    virtual void insertObject(Viewer::Object d_viewerObject, Viewer::Object sgObject);

    bool loadPlugin(const char *name);

    virtual float getLODScale();
    virtual float defaultCreaseAngle();
    virtual void menuEvent(coMenuItem *aButton);
    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);

    void startCapture();
    void stopCapture();
    void printViewpoint();
    virtual void update();
    int maxEntryNumber;
    coTUITab *vrmlTab;

    list<ViewpointEntry *> getViewpointEntries()
    {
        return viewpointEntries;
    }

protected:
    coSubMenuItem *VRMLButton;
    coRowMenu *viewpointMenu;
    coCheckboxGroup *cbg;
    list<ViewpointEntry *> viewpointEntries;
    coButtonMenuItem *addVPButton;
    coVRFileManager *mFileManager;
    coButtonMenuItem *reloadButton;
    coTUIButton *saveViewpoint;
    coTUIToggleButton *saveAnimation;
    coTUIButton *reload;
    FILE *fp;
    int fileNumber;
    int frameNumber;
    float *positions;
    float *orientations;
    bool record;
    bool doRemoteFetch;
};
#endif // SYSTEM_COVER_H
