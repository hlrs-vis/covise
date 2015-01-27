/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_RENDERER_H
#define _INV_RENDERER_H

/* $Id: InvRenderer.h,v 1.1 1994/04/12 13:39:31 zrfu0125 Exp zrfu0125 $ */

/* $Log: InvRenderer.h,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

//**************************************************************************
//
// * Description    : Inventor renderer abstract base class
//
// * Class(es)      : InvRenderer
//
//
// * inherited from : none
//
//
// * Author  : Dirk Rantzau
//
//
// * History : 29.03.94 V 1.0
//
//
//
//**************************************************************************

//
// classes
//
class InvRenderer;

//
// ec stuff
//
#include <covise/covise_process.h>

//
// X11 stuff
//
#include <X11/Intrinsic.h>
#include <Xm/Xm.h>

//
// Inventor stuff
//
#include <Inventor/SoDB.h>
#include <Inventor/SoNodeKitPath.h>
#include <Inventor/SoPickedPoint.h>
#include <Inventor/Xt/SoXt.h>
#include <Inventor/Xt/SoXtComponent.h>
#include <Inventor/Xt/SoXtClipboard.h>
#include <Inventor/Xt/SoXtDirectionalLightEditor.h>
#include <Inventor/Xt/SoXtMaterialEditor.h>
#include <Inventor/Xt/SoXtPrintDialog.h>
#include <Inventor/Xt/SoXtResource.h>
#include <Inventor/Xt/SoXtTransformSliderSet.h>
#include <Inventor/Xt/viewers/SoXtExaminerViewer.h>
#include <Inventor/Xt/viewers/SoXtFlyViewer.h>
#include <Inventor/Xt/viewers/SoXtPlaneViewer.h>
#include <Inventor/Xt/viewers/SoXtWalkViewer.h>
#include <Inventor/actions/SoBoxHighlightRenderAction.h>
#include <Inventor/actions/SoGetBoundingBoxAction.h>
#include <Inventor/actions/SoGetMatrixAction.h>
#include <Inventor/actions/SoSearchAction.h>
#include <Inventor/actions/SoWriteAction.h>
#include <Inventor/details/SoNodeKitDetail.h>
#include <Inventor/draggers/SoDirectionalLightDragger.h>
#include <Inventor/draggers/SoTabBoxDragger.h>
#include <Inventor/manips/SoCenterballManip.h>
#include <Inventor/manips/SoDirectionalLightManip.h>
#include <Inventor/manips/SoHandleBoxManip.h>
#include <Inventor/manips/SoJackManip.h>
#include <Inventor/manips/SoPointLightManip.h>
#include <Inventor/manips/SoSpotLightManip.h>
#include <Inventor/manips/SoTabBoxManip.h>
#include <Inventor/manips/SoTrackballManip.h>
#include <Inventor/manips/SoTransformBoxManip.h>
#include <Inventor/nodekits/SoBaseKit.h>
#include <Inventor/nodes/SoCallback.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/nodes/SoDirectionalLight.h>
#include <Inventor/nodes/SoEnvironment.h>
#include <Inventor/nodes/SoLabel.h>
#include <Inventor/nodes/SoLight.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoPointLight.h>
#include <Inventor/nodes/SoSelection.h>
#include <Inventor/nodes/SoShape.h>
#include <Inventor/nodes/SoSpotLight.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoTexture2.h>
//
// Renderer stuff
//
#include "InvDefs.h"
#include "InvError.h"

//=========================================================================
// InvRenderer
//=========================================================================

class InvRenderer
{

public:
    InvRenderer(){};

    virtual void show() = 0;
    virtual void hide() = 0;
    virtual Widget buildWidget(Widget parent, const char *name) = 0;
    virtual void setSceneGraph(SoNode *root) = 0;
    virtual void addToSceneGraph(SoGroup *child, const char *name, SoGroup *root) = 0;
    virtual void removeFromSceneGraph(SoGroup *root, const char *name) = 0;
    virtual void replaceSceneGraph(SoNode *root) = 0;
    virtual void addToTextureList(SoTexture2 *tex) = 0;
    virtual void removeFromTextureList(SoTexture2 *tex) = 0;
    virtual void addColormap(const char *name, const char *colormap) = 0;
    virtual void deleteColormap(const char *name) = 0;
    virtual void replaceColormap(const char *name, const char *colormap) = 0;
    virtual void addPart(const char *name, int partId, SoSwitch *s) = 0;
    virtual void replacePart(const char *name, int partId, SoSwitch *s) = 0;
    virtual void deletePart(const char *name) = 0;
    virtual void addTimePart(const char *name, int timeStep, int partId, SoSwitch *s) = 0;
    virtual void replaceTimePart(const char *name, int timeStep, int partId, SoSwitch *s) = 0;
    virtual void deleteTimePart(const char *name) = 0;
    virtual void setMaster() = 0;
    virtual void setSlave() = 0;
    virtual void setMasterSlave() = 0;
    virtual int isMaster() = 0;
    virtual int isSynced() = 0;
    virtual void setTransformation(float pos[3], float ori[4], int view,
                                   float aspect, float near, float far,
                                   float focal, float angle) = 0;
    virtual void setRenderTime(float time) = 0;
    virtual void receiveTransformation(char *message) = 0;
    virtual void receiveTelePointer(char *message) = 0;
    virtual void receiveDrawstyle(char *message) = 0;
    virtual void receiveLightMode(char *message) = 0;
    virtual void receiveTransparency(char *message) = 0;
    virtual void receiveSyncMode(char *message) = 0;
    virtual void receiveSelection(char *message) = 0;
    virtual void receiveDeselection(char *message) = 0;
    virtual void receivePart(char *message) = 0;
    virtual void receiveReferencePart(char *message) = 0;
    virtual void receiveResetScene() = 0;
    virtual void receiveFog(char *message) = 0;
    virtual void receiveAntialiasing(char *message) = 0;
    virtual void receiveBackcolor(char *message) = 0;
    virtual void receiveAxis(char *message) = 0;
    virtual void receiveClippingPlane(char *message) = 0;
    virtual void receiveViewing(char *message) = 0;
    virtual void receiveProjection(char *message) = 0;
    virtual void receiveDecoration(char *message) = 0;
    virtual void receiveHeadlight(char *message) = 0;
    virtual void receiveColormap(char *message) = 0;

    virtual void unmanageObjs(){};
    virtual void manageObjs(){};

    virtual ~InvRenderer(){};
};
#endif
