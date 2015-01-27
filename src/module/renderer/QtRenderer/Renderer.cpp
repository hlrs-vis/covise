/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//**************************************************************************
//
// * Description    : This is the main module for the renderer
//
//
// * Class(es)      : none
//
//
// * inherited from : none
//
//
// * Author  : Dirk Rantzau
//
//
// * History : 29.03.94 V 1.0
//             08.01.99(!) V 2.0 (YAC)
//
//
//**************************************************************************

//
// renderer stuff
//
#include "InvDefs.h"
#include "Renderer.h"

//
// environment stuff
//
#include <iostream>
#include <sstream>
#include <stdlib.h>

#ifndef _WIN32
#include <unistd.h>
#endif
#include <string.h>
#include <qapplication.h>
#include <qsocketnotifier.h>
#include <Inventor/Qt/SoQt.h>
#ifndef YAC
#include "InvMain.h"
#else
#include "InvMain_yac.h"
#ifdef HAS_MPI
#include <mpi.h>
#endif
#endif
#include "SoVolume.h"
#include "qtimer.h"
#include "SoVolumeDetail.h"
#include "InvComposePlane.h"
#include "InvObjectManager_yac.h"
#include "SoBillboard.h"

#include "do/coDoGeom.h"
#include "do/coDoData.h"
#include "util/coObjID.h"
#include "util/coErr.h"
#include "comm/transport/coFdTracer.h"

#include <Inventor/Qt/SoQt.h>
#include <Inventor/Qt/viewers/SoQtExaminerViewer.h>
#include <Inventor/nodes/SoGroup.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoNode.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoTransform.h>
#include <Inventor/nodes/SoSelection.h>
#include <Inventor/SoInput.h>
using namespace covise;

InventorRenderer *g_inv_renderer = NULL;
//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
int main(int argc, char *argv[])
{
#if defined(YAC) && defined(HAS_MPI)
    MPI_Init(&argc, &argv);
#endif
    new QApplication(argc, argv);

    covise::coDispatcher *dispatcher = covise::coDispatcher::Instance();
    // You should not create a QApplication instance if you want
    // to receive spaceball events.

    // Initialize Qt and SoQt.
    SoQt::init(argc, argv, argv[0]);

    // Initialized Inventor extensions
    SoVolume::initClass();
    SoVolumeDetail::initClass();
    SoBillboard::initClass();
    InvComposePlane::initClass();

    LOGINFO("QtRenderer started with %d arguments", argc);
    for (int i = 0; i < argc; i++)
    {
        LOGINFO("argv[%d] --> %s", i, argv[i]);
    }
    // Set up a new main window.
    InvMain *invmain = new InvMain(argc, argv);
    //

    InventorRenderer *yacModule = new InventorRenderer(argc, argv, invmain, "Inventor Renderer module");
    g_inv_renderer = yacModule;

    // register the module in the dispatcher
    dispatcher->add(yacModule);

    // Start event loop.
    SoQt::mainLoop();

    // clean up
    coDispatcher::deleteDispatcher();

    // done
    return (0);
}

void InventorRenderer::addFd(int fd)
{
    int i = 0;
    while (fds[i])
    {
        i++;
    }
    fds[i] = fd;
    fds[i + 1] = 0;
    notifiers[i + 1] = 0;

    notifiers[i] = new QSocketNotifier(fds[i], QSocketNotifier::Read, 0, 0);
    notifiers[i]->setEnabled(true);

    QObject::connect(notifiers[i], SIGNAL(activated(int)),
                     this, SLOT(dataReceived(int)));
    notifiers[i + 1] = 0;
    eNotifiers[i] = new QSocketNotifier(fds[i], QSocketNotifier::Exception);
    eNotifiers[i]->setEnabled(true);
    QObject::connect(eNotifiers[i], SIGNAL(activated(int)),
                     this, SLOT(dataReceived(int)));
    eNotifiers[i + 1] = 0;
    dataReceived(fds[i]); // handle all pending messages
}

void InventorRenderer::processMPI_DATA()
{
    coDispatcher::Instance()->dispatch(0.0001);
}
void InventorRenderer::dataReceived(int /*socket*/)
{
    coDispatcher::Instance()->dispatch(0);
}

void InventorRenderer::removeFd(int fd)
{
    int i = 0;
    while (fds[i] && fds[i] != fd)
    {
        i++;
    }
    delete notifiers[i];
    delete eNotifiers[i];
    while (fds[i])
    {
        fds[i] = fds[i + 1];
        notifiers[i] = notifiers[i + 1];
        eNotifiers[i] = eNotifiers[i + 1];
        i++;
    }
}

InventorRenderer::InventorRenderer(int argc, char *argv[], InvMain *main, const char *desc)
    : coRenderer(argc, argv, desc)
{
    notifiers[0] = NULL;
    eNotifiers[0] = NULL;
    fds[0] = 0;
    om = renderer->om;
    coFdTracer::addFdListener(this);
    mainWindow = main;

#if defined(HAS_MPI)
    QTimer *periodictimer = new QTimer;
    QObject::connect(periodictimer, SIGNAL(timeout()), this, SLOT(processMPI_DATA()));
    periodictimer->start(100);
#endif
}

int InventorRenderer::initialize()
{
    // get available colormaps from registry
    coRegistryAccess *regAccess = new coRegistryAccess(g_inv_renderer);

    // we also want to keep track of changes
    regAccess->subscribeClass("CMap", 0, this);

    return coRenderer::initialize();
}

void *InventorRenderer::addPoints(coDoPoints *points, coDoRGBA *colors, void *grp, int /* grpOffset */)
{
    float *xc = NULL, *yc = NULL, *zc = NULL;
    int *pc = NULL;
    points->getAddresses(&xc, &yc, &zc);
    int nc = 0, cb = INV_NONE;
    if (colors)
    {
        colors->getAddress(&pc);
        nc = colors->getNumPoints();
        if (getBinding(points, colors) == coRendererBinding::CO_OVERALL)
            cb = INV_OVERALL;
        else if (getBinding(points, colors) == coRendererBinding::CO_PER_VERTEX)
            cb = INV_PER_VERTEX;
    }
    const char *ptSizeStr = points->getAttribute("POINTSIZE");
    float ptSize = ptSizeStr ? atof(ptSizeStr) : 0.0;

    addColorMap(points->getHdr()->getID().getString(), colors);

    const char *parent_name = NULL;
    if (grp)
        parent_name = ((SoNode *)grp)->getName().getString();

    //char myname[80];
    //if (sprintf(myname,"POINTS_%d_%d", points->getHdr()->getModID(), points->getHdr()->getPortID()) == -1)
    //{
    //  LOGERROR("Could not create object-name!");
    //}

    return (om->addPoint((char *)(points->getHdr()->getID().getString()), parent_name, points->getNumPoints(), xc, yc, zc, cb, INV_RGBA, NULL, NULL, NULL, pc, ptSize));
}

void *InventorRenderer::addLines(coDoLines *obj, coDoRGBA *colors, void *grp, int /* grpOffset */)
{
    float *xc = NULL, *yc = NULL, *zc = NULL;
    int *vl = NULL, *ll = NULL, *pc = NULL;
    obj->getAddresses(&xc, &yc, &zc, &vl, &ll);
    int nc = 0, cb = INV_NONE;
    if (colors)
    {
        colors->getAddress(&pc);
        nc = colors->getNumPoints();
        if (getBinding(obj, colors) == coRendererBinding::CO_OVERALL)
            cb = INV_OVERALL;
        else if (getBinding(obj, colors) == coRendererBinding::CO_PER_VERTEX)
            cb = INV_PER_VERTEX;
        else if (getBinding(obj, colors) == coRendererBinding::CO_PER_ELEMENT)
            cb = INV_PER_FACE;
        else if (getBinding(obj, colors) == coRendererBinding::CO_PER_PRIMITIVE)
            cb = INV_PER_FACE;
    }
    const char *rName = NULL;
    rName = obj->getHdr()->getID().getString();

    char *feedbackStr = NULL;
    const char *tmpStr = obj->getAttribute("FEEDBACK");
    if (tmpStr)
    {
        feedbackStr = new char[1 + strlen(tmpStr)];
        strcpy(feedbackStr, tmpStr);
        // lets see if we have simultaniously an IGNORE attribute (for CuttingSurfaces)
        tmpStr = obj->getAttribute("IGNORE");
        if (tmpStr)
        {
            char *tmp = new char[18 + strlen(feedbackStr) + strlen(tmpStr)];
            strcpy(tmp, feedbackStr);
            strcat(tmp, "<IGNORE>");
            strcat(tmp, tmpStr);
            strcat(tmp, "<IGNORE>");
            delete[] feedbackStr;
            feedbackStr = tmp;
        }
    }

    addColorMap(rName, colors);

    const char *parent_name = NULL;
    if (grp)
        parent_name = ((SoNode *)grp)->getName().getString();

    //   char myname[80];
    //   if (sprintf(myname,"LINE_%d_%d", obj->getHdr()->getModID(), obj->getHdr()->getPortID()) == -1)
    //   {
    //     LOGERROR("Could not create object-name!");
    //   }

    return (om->addLine((char *)(obj->getHdr()->getID().getString()), parent_name,
                        obj->getNumLines(), obj->getNumVertices(), obj->getNumPoints(),
                        xc, yc, zc, vl, ll, nc, cb, INV_RGBA, NULL, NULL, NULL, pc,
                        0, INV_NONE, NULL, NULL, NULL, /* material */ NULL, rName, feedbackStr));
}

void *InventorRenderer::addPolygons(coDoPolygons *obj, coDoRGBA *colors, coDoVec3 *normals, void *grp, int /* grpOffset */)
{
    float *xc = NULL, *yc = NULL, *zc = NULL;
    int *vl = NULL, *ll = NULL, *pc = NULL;
    obj->getAddresses(&xc, &yc, &zc, &vl, &ll);
    float *xn = NULL, *yn = NULL, *zn = NULL;
    int nn = 0, nc = 0, nb = INV_NONE, cb = INV_NONE;
    if (normals)
    {
        normals->getAddresses(&xn, &yn, &zn);
        nn = normals->getNumPoints();
        if (getBinding(obj, normals) == coRendererBinding::CO_OVERALL)
            nb = INV_OVERALL;
        else if (getBinding(obj, normals) == coRendererBinding::CO_PER_VERTEX)
            nb = INV_PER_VERTEX;
        else if (getBinding(obj, normals) == coRendererBinding::CO_PER_ELEMENT)
            nb = INV_PER_FACE;
        else if (getBinding(obj, normals) == coRendererBinding::CO_PER_PRIMITIVE)
            nb = INV_PER_FACE;
    }
    if (colors)
    {
        colors->getAddress(&pc);
        nc = colors->getNumPoints();
        if (getBinding(obj, colors) == coRendererBinding::CO_OVERALL)
            cb = INV_OVERALL;
        else if (getBinding(obj, colors) == coRendererBinding::CO_PER_VERTEX)
            cb = INV_PER_VERTEX;
        else if (getBinding(obj, colors) == coRendererBinding::CO_PER_ELEMENT)
            cb = INV_PER_FACE;
        else if (getBinding(obj, colors) == coRendererBinding::CO_PER_PRIMITIVE)
            cb = INV_PER_FACE;
    }

    const char *rName = NULL;
    rName = obj->getHdr()->getID().getString();

    char *feedbackStr = NULL;
    const char *tmpStr = obj->getAttribute("FEEDBACK");
    if (tmpStr)
    {
        feedbackStr = new char[1 + strlen(tmpStr)];
        strcpy(feedbackStr, tmpStr);
        // lets see if we have simultaniously an IGNORE attribute (for CuttingSurfaces)
        tmpStr = obj->getAttribute("IGNORE");
        if (tmpStr)
        {
            char *tmp = new char[18 + strlen(feedbackStr) + strlen(tmpStr)];
            strcpy(tmp, feedbackStr);
            strcat(tmp, "<IGNORE>");
            strcat(tmp, tmpStr);
            strcat(tmp, "<IGNORE>");
            delete[] feedbackStr;
            feedbackStr = tmp;
        }
    }

    const char *parent_name = NULL;
    if (grp)
        parent_name = ((SoNode *)grp)->getName().getString();

    return (om->addPolygon((char *)(obj->getHdr()->getID().getString()), parent_name,
                           obj->getNumPolygons(), obj->getNumVertices(), obj->getNumPoints(),
                           xc, yc, zc, vl, ll, nc, cb, INV_RGBA, NULL, NULL, NULL, pc,
                           nn, nb, xn, yn, zn, 0.0, 2, 0, 0, 0, NULL, 0, NULL, NULL, NULL, rName, feedbackStr));

    addColorMap(rName, colors);
    /*
    float *xc,*yc,*zc;
	points->getAddresses(&xc,&yc,&zc);
	return (om->addPoint((char *)(points->getHdr()->getID().getString()),NULL,points->getNumPoints(),xc,yc,zc,INV_NONE,INV_NONE,NULL,NULL,NULL,NULL));
*/
}

// add trianglestrips
void *InventorRenderer::addTriangleStrips(coDoTriangleStrips *obj, coDoRGBA *colors, coDoVec3 *normals, void *grp, int /* grpOffset */)
{
    float *xc = NULL, *yc = NULL, *zc = NULL;
    int *vl = NULL, *ll = NULL, *pc = NULL;
    obj->getAddresses(&xc, &yc, &zc, &vl, &ll);
    float *xn = NULL, *yn = NULL, *zn = NULL;
    int nn = 0, nc = 0, nb = INV_NONE, cb = INV_NONE;

    LOGINFO("No.of strips: %d   No.of vertices: %d   No.of points: %d", obj->getNumStrips(), obj->getNumVertices(), obj->getNumPoints());

    if (normals)
    {
        normals->getAddresses(&xn, &yn, &zn);
        nn = normals->getNumPoints();
        if (getBinding(obj, normals) == coRendererBinding::CO_OVERALL)
            nb = INV_OVERALL;
        else if (getBinding(obj, normals) == coRendererBinding::CO_PER_VERTEX)
            nb = INV_PER_VERTEX;
        else if (getBinding(obj, normals) == coRendererBinding::CO_PER_ELEMENT)
            nb = INV_PER_FACE;
        else if (getBinding(obj, normals) == coRendererBinding::CO_PER_PRIMITIVE)
            nb = INV_PER_FACE;
    }
    if (colors)
    {
        colors->getAddress(&pc);
        nc = colors->getNumPoints();
        if (getBinding(obj, colors) == coRendererBinding::CO_OVERALL)
            cb = INV_OVERALL;
        else if (getBinding(obj, colors) == coRendererBinding::CO_PER_VERTEX)
            cb = INV_PER_VERTEX;
        else if (getBinding(obj, colors) == coRendererBinding::CO_PER_ELEMENT)
            cb = INV_PER_FACE;
        else if (getBinding(obj, colors) == coRendererBinding::CO_PER_PRIMITIVE)
            cb = INV_PER_FACE;
    }

    const char *rName = NULL;
    rName = obj->getHdr()->getID().getString();

    char *feedbackStr = NULL;
    const char *tmpStr = obj->getAttribute("FEEDBACK");
    if (tmpStr)
    {
        feedbackStr = new char[1 + strlen(tmpStr)];
        strcpy(feedbackStr, tmpStr);
        // lets see if we have simultaniously an IGNORE attribute (for CuttingSurfaces)
        tmpStr = obj->getAttribute("IGNORE");
        if (tmpStr)
        {
            char *tmp = new char[18 + strlen(feedbackStr) + strlen(tmpStr)];
            strcpy(tmp, feedbackStr);
            strcat(tmp, "<IGNORE>");
            strcat(tmp, tmpStr);
            strcat(tmp, "<IGNORE>");
            delete[] feedbackStr;
            feedbackStr = tmp;
        }
    }

    addColorMap(rName, colors);

    const char *parent_name = NULL;
    if (grp)
        parent_name = ((SoNode *)grp)->getName().getString();

    return (om->addTriangleStrip((char *)(obj->getHdr()->getID().getString()), parent_name,
                                 obj->getNumStrips(), obj->getNumVertices(), obj->getNumPoints(),
                                 xc, yc, zc, vl, ll, nc, cb, INV_RGBA, NULL, NULL, NULL, pc,
                                 nn, nb, xn, yn, zn, 0.0, 2, 0, 0, 0, NULL, 0, NULL, NULL, NULL, rName, feedbackStr));
}

void InventorRenderer::addColorMap(const char *objName, coDoRGBA *colors)
{
    if (!colors || !objName)
        return;

    const char *colorMap = colors->getAttribute("COLORMAP");
    if (colorMap)
    {
        char mapName[256];
        float min, max;
        stringstream stream;
        stream << colorMap;
        stream >> mapName;
        stream >> min >> max;
        mainWindow->insertColorListItem(objName, mapName, min, max, 256);
    }
}

void InventorRenderer::removeColorMap(const char *objName)
{
    mainWindow->removeColorListItem(objName);
}

void InventorRenderer::removeObject(void *obj, void *root)
{
    (void)root;

    SoNode *node = (SoNode *)obj;
    SoLabel *objName = NULL;
    if (node)
    {
        SoNode *tmpNode = ((SoGroup *)node)->getChild(0);
        if (tmpNode && tmpNode->isOfType(SoLabel::getClassTypeId()))
        {
            objName = (SoLabel *)tmpNode;
            LOGINFO("removeObject() calling deleteObject() for %s", (char *)(objName->label.getValue().getString()));
            removeColorMap(objName->label.getValue().getString());
            om->deleteObject((char *)(objName->label.getValue().getString()));
        }
        else
        {
            SoGroup *sep = (SoGroup *)tmpNode; // the separator
            int numnodes = sep->getNumChildren();
            LOGINFO("InventorRenderer::removeObject() called - numnodes=%d", numnodes);
            for (int i = 0; i < numnodes; i++)
            {
                node = sep->getChild(i);
                if (node && node->isOfType(SoLabel::getClassTypeId()))
                {
                    objName = (SoLabel *)node;
                    LOGINFO("removeObject() calling deleteObject() for %s", (char *)(objName->label.getValue().getString()));
                    removeColorMap(objName->label.getValue().getString());
                    om->deleteObject((char *)(objName->label.getValue().getString()));
                    break;
                }
            }
        }
    }
    /*
  if (!obj) return;
  if (renderer->sequencer) renderer->sequencer->stopPlayback();

  SoNode *node = (SoNode *)obj;
  SoGroup *parent = (SoGroup *)root;
  const SbName objName = node->getName();
  QString obj_name_str(objName.getString());
  LOGINFO("removeObject() calling deleteObject() for %s", obj_name_str.latin1());
  if (parent)
  {
    // strip S_ or G_ from the name
    obj_name_str.remove(0,2);
    // remove from InvObjectManager and the ListBox
    om->deleteObject(obj_name_str.latin1());
    // remove from inventor
    parent->removeChild(node);
    LOGINFO("Child removed ! Parent has now %d children !", parent->getNumChildren());
    // if there are no more children left, remove the parent too
    if (parent->getNumChildren() == 0)
    {
      const SbName parentName = parent->getName();
      obj_name_str = parentName.getString();
      LOGINFO("**** parent has no more children so removing it (%s) !!!!!!!!!", obj_name_str.latin1());
      // remove from QtRenderer's internal list
      om->deleteObject(obj_name_str.latin1());
      if (!isLockedGroup(parent)) renderer->viewer->removeFromSceneGraph((SoGroup *)parent, NULL);
//      // and also remove it from the list-box
//      QListViewItem *item = renderer->objListBox->findItem(obj_name_str.latin1(), 0);
//      if(item != NULL)
//      {
//        delete item;
//      } else {
//        LOGWARNING("Could not find >%s< in ListBox to delete!");
//      }
//
    }
  } else {
    renderer->viewer->removeFromSceneGraph((SoGroup *)node, NULL);
  }
  // hide sequencer
  if (renderer->viewer->selection->getNumChildren() == 1)
  {
    if (renderer->sequencer) renderer->sequencer->hide();
  }
*/
    LOGINFO("InventorRenderer::removeObject() done");
}

void InventorRenderer::removeColors(void *obj, void *root)
{
    /*  int i;
  SoNode *node=(SoNode *)obj;
  SoLabel *objName = NULL;
  if(node)
  {
    SoGroup *sep  = (SoGroup *)((SoGroup *)node)->getChild(0); // the separator
    int numnodes=sep->getNumChildren();
    LOGINFO("InventorRenderer::removeColors() called - numnodes=%d", numnodes);
    for(i=0;i<numnodes;i++)
    {
      node = sep->getChild(i);
      if (node && node->isOfType(SoLabel::getClassTypeId()))
      {
        objName=(SoLabel *)node;
        LOGINFO("removeColors() Node-Label is %s", (char *)(objName->label.getValue().getString()));
        //om->deleteObject((char *)(objName->label.getValue().getString()));
        break;
      }
    }
  }
*/
    LOGINFO("InventorRenderer::removeColors(0x%x, 0x%x) called (this is disabled for the moment)", obj, root);
}

int InventorRenderer::quit()
{
    //exit(1); // very nasty
    LOGINFO("InventorRenderer::quit() - called");
    SoQt::exitMainLoop();
    return true;
}

void *InventorRenderer::getNewGroup(const char *desc, void *grp, bool is_timestep, int max)
{

    const char *parent_name = NULL;
    if (grp)
        parent_name = ((SoNode *)grp)->getName().getString();

    return (om->addSeparator((char *)desc, (char *)parent_name, is_timestep, 0, max));
    /*

  SoGroup *group = (SoGroup *)grp;
  SoGroup *retval = NULL;
  SoNode  *tmp = NULL;
  SoSeparator *sep = NULL;

  if (!group) // new group requested
  {
    if (is_timestep) // create a switch (for TS) or just a normal group
    {
      // check whether TS-switch with given name already exists
      tmp = SoNode::getByName(desc);
      
      if (tmp && tmp->isOfType(SoSwitch::getClassTypeId()))
      {
        // oops we already have a switch with the given name, so remove it
        //renderer->viewer->sceneGraph->removeChild(tmp);
        //renderer->viewer->removeFromSceneGraph((SoGroup *)tmp, desc);
        return tmp;
      }
      // create new TS-group
      tmp = new SoSwitch;
      assert(tmp != NULL);
      // give it a name
      tmp->setName(desc);
      //renderer->viewer->sceneGraph->addChild(tmp);
      renderer->viewer->addToSceneGraph((SoGroup *)tmp, desc, NULL);
      retval = (SoSwitch *)tmp;
    } else {
      // create just a "normal" group (SoSeparator)
      sep = new SoSeparator;
      assert(sep != NULL);
      sep->setName(desc);
      //renderer->viewer->sceneGraph->addChild(sep);
      renderer->viewer->addToSceneGraph(sep, desc, NULL);
      retval = sep;
    }
  } else {
    // a group is given. it does not really matter whether the given group
    // is a time-switch or not
    sep = new SoSeparator;
    assert(sep != NULL);
    sep->setName(desc);
    //group->addChild(sep);
    renderer->viewer->addToSceneGraph(sep, desc, group);
    retval = sep;
  }

  om->addGroup(retval->getName().getString(), retval, group);
  return retval;*/
}

void InventorRenderer::setMaxTimesteps(int max)
{

    renderer->sequencer->setSliderBounds(1, max, 1);
}

void InventorRenderer::update(coRegEntry *changedEntry)
{
    const char *na = changedEntry->getVar();

    if (changedEntry->isDeleted())
        mainWindow->removeColorMapEntry(na);
    else
        mainWindow->addColorMapEntry(na, changedEntry->getValue());
}
