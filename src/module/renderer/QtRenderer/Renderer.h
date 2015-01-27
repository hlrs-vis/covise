/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__INVENTOR_RENDERER_MAIN_H)
#define __INVENTOR_RENDERER_MAIN_H

#include <list>
#include <qobject.h>
#include <api/coAPI.h>
#include <comm/event/coFdListener.h>
#include <renderer/coRenderer.h>

class InventorRenderer;
class InvObjectManager;
class InvRenderManager;
class QSocketNotifier;
class InvMain;

using namespace covise;

extern InventorRenderer *g_inv_renderer;

class InventorRenderer : public QObject, public covise::coRenderer, public covise::coFdListener, public covise::coRegEntryObserver
{
    Q_OBJECT
public:
    // constructor/destructor
    InventorRenderer(int argc, char *argv[], InvMain *main, const char *desc);
    virtual ~InventorRenderer()
    {
        quit();
    };

    virtual int quit();

    // deleteObj is allways the same --- don't change anything
    void deleteObj(covise::coDistributedObject *obj)
    {
        delete obj;
    };

    // remove object
    void removeObject(void *obj, void *root);
    void removeColors(void *obj, void *root);

    virtual int initialize();

    // build a new group
    void *getNewGroup(const char *desc, void *grp = NULL, bool is_timestep = false, int max = -1);

    // add points
    void *addPoints(covise::coDoPoints *points, covise::coDoRGBA *colors, void *grp, int grpOffset);

    // add lines
    void *addLines(covise::coDoLines *obj, covise::coDoRGBA *colors, void *grp, int grpOffset);

    // add trianglestrips
    void *addTriangleStrips(covise::coDoTriangleStrips *obj, covise::coDoRGBA *colors, covise::coDoVec3 *normals, void *grp, int grpOffset);

    // add polygons
    void *addPolygons(covise::coDoPolygons *obj, covise::coDoRGBA *colors, covise::coDoVec3 *normals, void *grp, int grpOffset);
    virtual void addFd(int fd);
    virtual void removeFd(int fd);

    virtual void setMaxTimesteps(int);

    virtual void update(covise::coRegEntry *);

private:
    void addColorMap(const char *objName, covise::coDoRGBA *colors);
    void removeColorMap(const char *objName);

    int fds[30];
    QSocketNotifier *notifiers[30];
    QSocketNotifier *eNotifiers[30];
    InvObjectManager *om;
    InvRenderManager *rm;
    InvMain *mainWindow;

private slots:
    void dataReceived(int);
    void processMPI_DATA();
};

#endif // __INVENTOR_RENDERER_MAIN_H
