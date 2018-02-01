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
#include <util/coTypes.h>
#include <osg/Matrix>
#include <map>
#include <list>
#include <cover/coTabletUI.h>
#include <cover/ui/Owner.h>

namespace opencover {
namespace ui {
class Action;
class Button;
class ButtonGroup;
class Menu;
class Group;
}
}

using namespace vrml;
using namespace opencover;

class VRML97PLUGINEXPORT ViewpointEntry
{
public:
    ViewpointEntry(VrmlNodeViewpoint *, VrmlScene *);
    virtual ~ViewpointEntry();
    void setMenuItem(ui::Button *aMenuItem);
    VrmlNodeViewpoint *getViewpoint()
    {
        return viewPoint;
    };
    ui::Button *getMenuItem()
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
    VrmlScene *scene = nullptr;
    VrmlNodeViewpoint *viewPoint = nullptr;
    ui::Button *menuItem = nullptr;
    coTUIToggleButton *tuiItem = nullptr;
};

class VRML97PLUGINEXPORT SystemCover : public System, public ui::Owner, public coTUIListener
{

public:
    SystemCover();
    virtual ~SystemCover()
    {
    }
    virtual double time();

    virtual const char *remoteFetch(const char *filename);

    bool loadUrl(const char *url, int np, char **parameters);

#if 0
    virtual void setBuiltInFunctionState(const char *fname, int val);
    virtual void setBuiltInFunctionValue(const char *fname, float val);
    virtual void callBuiltInFunctionCallback(const char *fname);
#endif

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
    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);

    void startCapture();
    void stopCapture();
    void printViewpoint();
    virtual void update();
    int maxEntryNumber;
    coTUITab *vrmlTab;

    std::list<ViewpointEntry *> getViewpointEntries()
    {
        return viewpointEntries;
    }

protected:
    ui::Menu *vrmlMenu = nullptr;
    ui::Group *viewpointGroup = nullptr;
    ui::ButtonGroup *cbg = nullptr;
    std::list<ViewpointEntry *> viewpointEntries;
    ui::Action *addVPButton;
    coVRFileManager *mFileManager;
    ui::Action *reloadButton;
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
    int viewPointCount = 0;
};
#endif // SYSTEM_COVER_H
