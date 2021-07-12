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
    double time() override;

    std::string remoteFetch(const std::string &filename, bool isTmp = false) override;
    int getFileId(const char* url) override;
    bool loadUrl(const char *url, int np, char **parameters) override;

#if 0
    virtual void setBuiltInFunctionState(const char *fname, int val);
    virtual void setBuiltInFunctionValue(const char *fname, float val);
    virtual void callBuiltInFunctionCallback(const char *fname);
#endif

    void setSyncMode(const char *mode) override;

    void setTimeStep(int ts) override; // set the timestep number for COVISE Animations
    void setActivePerson(int p) override; // set the active Person

    bool isMaster() override;
    void becomeMaster() override;

    VrmlMessage *newMessage(size_t size) override;
    void sendAndDeleteMessage(VrmlMessage *msg) override;
    bool hasRemoteConnection() override;

    Player *getPlayer() override;

    long getMaxHeapBytes() override;
    bool getHeadlight() override;
    void setHeadlight(bool enable) override;
    bool getPreloadSwitch() override;
    float getSyncInterval() override;

    void addViewpoint(VrmlScene *scene, VrmlNodeViewpoint *viewpoint) override;
    bool removeViewpoint(VrmlScene *scene, const VrmlNodeViewpoint *viewpoint) override;
    bool setViewpoint(VrmlScene *scene, const VrmlNodeViewpoint *viewpoint) override;

    void setCurrentFile(const char *filename) override;

    void setMenuVisibility(bool visible) override;
    void createMenu() override;
    void destroyMenu() override;

    void setNavigationType(std::string) override;
    void setNavigationStepSize(double size) override;
    void setNavigationDriveSpeed(double speed) override;
    void setNearFar(float near, float far) override;

    double getAvatarHeight() override;
    int getNumAvatars() override;
    bool getAvatarPositionAndOrientation(int num, float pos[3], float ori[4]) override;

    bool getViewerPositionAndOrientation(float pos[3], float ori[4]) override;
    bool getLocalViewerPositionAndOrientation(float pos[3], float ori[4]) override;

    bool getViewerFeetPositionAndOrientation(float pos[3], float ori[4]) override;
    bool getPositionAndOrientationFromMatrix(const double *M, float pos[3], float ori[4]) override;
    bool getPositionAndOrientationFromMatrix(const osg::Matrix &mat, float pos[3], float ori[4]);
    void transformByMatrix(const double *M, float pos[3], float ori[4]) override;
    void getInvBaseMat(double *M) override;
    void getPositionAndOrientationOfOrigin(const double *M, float pos[3], float ori[4]) override;

    std::string getConfigEntry(const char *key) override;
    bool getConfigState(const char *key, bool defaultVal) override;

    CacheMode getCacheMode() const override;
    std::string getCacheName(const char *url, const char *pathname) const override;
    void storeInline(const char *name, const Viewer::Object d_viewerObject) override;
    Viewer::Object getInline(const char *name) override;
    void insertObject(Viewer::Object d_viewerObject, Viewer::Object sgObject) override;

    bool loadPlugin(const char *name) override;

    float getLODScale() override;
    float defaultCreaseAngle() override;
    void tabletEvent(coTUIElement *tUIItem) override;
    void tabletPressEvent(coTUIElement *tUIItem) override;

    void startCapture();
    void stopCapture();
    void printViewpoint();
    void update() override;
    int maxEntryNumber;
    coTUITab *vrmlTab;

    std::list<ViewpointEntry *> getViewpointEntries()
    {
        return viewpointEntries;
    }
	bool doOptimize() override;
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
    CacheMode cacheMode = CACHE_CREATE;

	bool m_optimize;
};
#endif // SYSTEM_COVER_H
