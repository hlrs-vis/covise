/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Log:  $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

// **************************************************************************
//
// * Description    : Inventor telepointer stuff
//
// * Class(es)      : InvTelePointer
//
// * inherited from :
//
// * Author  : Dirk Rantzau
//
// * History : 07.04.94 V 1.0
//
// **************************************************************************

#include "InvTelePointer.h"
#include "InvError.h"
#include <util/coLog.h>
#include <covise/covise_process.h>
#include <Inventor/nodes/SoAnnotation.h>
#include <Inventor/elements/SoCacheElement.h>

extern int port;
extern char *host;

//======================================================================
//
// Description:
//	create telepointer geometry
//
//
// Use: private
//======================================================================
void TelePointer::makeTelePointer()
{

    tp_shape = new SoSeparator;
    tp_color = new SoMaterial;
    tp_font = new SoFont;
    tp_icon = new SoText2;
    tp_translation = new SoTranslation;
    tp_transIcon = new SoTranslation;

#ifdef __linux__
    drawCBPre_ = new SoCallback;
    drawCBPre_->setCallback(drawManipPre, this);

    drawCBPost_ = new SoCallback;
    drawCBPost_->setCallback(drawManipPost, this);
#endif

    tp_font->name.setValue("Times-Roman");
    tp_font->size.setValue(24.0);
    tp_transIcon->translation.setValue(0.0, -0.025, 0.0);
    tp_icon->string.setValue(name);

    tp_shape->addChild(tp_color);
    tp_shape->addChild(tp_translation);
    tp_shape->addChild(tp_font);
    tp_shape->addChild(tp_transIcon);
#ifdef __linux__
    tp_shape->addChild(drawCBPre_);
#endif

    tp_shape->addChild(tp_icon);
#ifdef __linux__
    tp_shape->addChild(drawCBPost_);
#endif
}

void
TelePointer::drawManipPre(void *d, SoAction *action)
{
    if (action->isOfType(SoGLRenderAction::getClassTypeId()))
    {
        // Make my custom GL calls
        ((TelePointer *)d)->doGLpre();

        // Invalidate the state so that a cache is not made
        SoCacheElement::invalidate(action->getState());
    }
}

void
TelePointer::drawManipPost(void *d, SoAction *action)
{
    if (action->isOfType(SoGLRenderAction::getClassTypeId()))
    {
        // Make my custom GL calls
        ((TelePointer *)d)->doGLpost();

        // Invalidate the state so that a cache is not made
        SoCacheElement::invalidate(action->getState());
    }
}

void
TelePointer::doGLpre()
{
    //    cerr << "TelePointer::doGLpre() called" << endl;
    glDisable(GL_DEPTH_TEST);
}

void
TelePointer::doGLpost()
{
    //    cerr << "TelePointer::doGLpost() called" << endl;
    glEnable(GL_DEPTH_TEST);
}

TPHandler::TPHandler()
{

    tpList = new SbPList();

    // make a scengraph root to put all the telepointers underneath
    // tp_sceneGraph = new SoSeparator;
    tp_sceneGraph = new SoAnnotation;

    tp_camera = new SoOrthographicCamera;

    //  tp_camera->position.setValue(0,0,-10000);
    tp_camera->nearDistance.setValue(-10000);
    tp_camera->farDistance.setValue(10000);

    tp_sceneGraph->addChild(tp_camera);
}

//======================================================================
//
// Description:
//	constructor
//
//
// Use: public
//======================================================================
TPHandler::TPHandler(InvExaminerViewer *viewer)
{
    SbColor color(0.5, 0.5, 0.5);
    tpList = new SbPList();

    // make a scengraph root to put all the telepointers underneath
    tp_sceneGraph = new SoSeparator;
    tp_camera = new SoOrthographicCamera;

    tp_sceneGraph->addChild(tp_camera);

    viewer->setOverlaySceneGraph(tp_sceneGraph);
    viewer->setOverlayColorMap(1, 1, &color);
}

SoSeparator *
TPHandler::getRoot()
{
    return tp_sceneGraph;
}

//======================================================================
//
// Description:
//	handle routine for telepointers
//
//
// Use: public
//======================================================================
void
TPHandler::handle(char *TPmessage, InvExaminerViewer *currentViewer)
{
    short found;
    TelePointer *ptr, *tp = NULL;
    char tpname[32];
    char tpLabel[34];
    int state;
    float px, py, pz;
    float aspectRatio;

    //   SbVec3f intersection(0,0,-10001);
    SbVec3f intersection;
    SbVec3f vec;

    // scan message
    int ret = sscanf(TPmessage, "%s %d %f %f %f %f", &tpname[0], &state, &px, &py, &pz, &aspectRatio);
    if (ret != 6)
    {
        fprintf(stderr, "TPHandler::handle: sscanf failed\n");
    }
    strcpy(tpLabel, "< ");
    strcat(tpLabel, tpname);

    // look if TP with this name already exists
    found = FALSE;
    int i = 0;
    int nb = tpList->getLength();
    while ((!found) && (i < nb))
    {
        ptr = (TelePointer *)(*tpList)[i];
        if (strcmp(ptr->getName(), tpLabel) == 0)
        {
            // found !
            found = TRUE;
            tp = ptr;
            //cerr << "Found existing TelePointer with name :" << tp->name << endl;
        }
        else
        {
            i++;
        }
    }

    if ((!found) && (state != CO_RMV))
    {
        // add a new TP to list
        TelePointer *tp_new = new TelePointer(tpname);
        tpList->append((void *)tp_new);
        covise::print_comment(__LINE__, __FILE__, "Creating a new TelePointer");
        // append to scenegraph
        tp_sceneGraph->addChild(tp_new->tp_shape);
        tp = tp_new;
    }

    switch (state)
    {
    case CO_OFF:
        // hide it
        if (tp)
            clearTelePointer(tp);
        break;
    case CO_ON:
        // set new translation
        // show it
        vec.setValue(px, py, pz);
        if (tp)
        {
            projectTelePointer(tp, vec, aspectRatio, intersection, currentViewer);
            setTelePointer(tp, intersection);
        }
        break;
    case CO_RMV:
        if (found)
        {
            tpList->remove(i);
            if (tp)
                tp_sceneGraph->removeChild(tp->tp_shape);
        }
        break;
    } // end switch
}

//======================================================================
//
// Description:
//	projection of telepointer
//
//
// Use: private
//======================================================================
void TPHandler::projectTelePointer(TelePointer *, const SbVec3f pos, float aspectRatio,
                                   SbVec3f &intersection, InvExaminerViewer *viewer)
{
    const int xOffset = 61;
    const int yOffset = 33;

    float xs, ys, zs; // normalized screen coordinates
    SbVec3f screen;

    SbVec2s size = viewer->getSize();
    if (viewer->isDecoration())
    {
        // !! Attention: SoXtRenderArea with offset
        size[0] = size[0] - xOffset;
        size[1] = size[1] - yOffset;
    }
    float aspRat = size[0] / (float)size[1];

    // set aspect ratio explicitely
    tp_camera->aspectRatio.setValue(aspRat);
    // default setting
    tp_camera->height.setValue(2.0);
    // scale height
    tp_camera->scaleHeight(1 / aspRat);

    /*
   float h = tp_camera->height.getValue();
   fprintf(stderr, "Height Camera: %.6f\n", h);
   */

    // get the view volume -> rectangular box
    SbViewVolume viewVolume = tp_camera->getViewVolume();
    // determine mouse position
    viewVolume.projectToScreen(pos, screen);
    screen.getValue(xs, ys, zs);
    /*
   cerr << "xs: " << xs << endl;
   cerr << "ys: " << ys << endl;
   */

    // project the mouse point to a line
    SbVec3f p0, p1;
    viewVolume.projectPointToLine(SbVec2f(xs, ys), p0, p1);

    // take the midpoint of the line as telepointer position
    intersection = (p0 + p1) / 2.0f;

    // adapt telepointer position to the aspect ratio
    if (aspectRatio > 1.0)
    {
        intersection[0] = intersection[0] * aspectRatio / aspRat;
        intersection[1] = intersection[1] * aspectRatio / aspRat;
    }

    if (aspectRatio < 1.0)
    {
        // in this case the aspect ratio submitted to the function
        // and belonging to the delivered position leads to wrong projection point
        // => local correction for x-, y-value
        intersection[0] = intersection[0] / aspRat;
        intersection[1] = intersection[1] / aspRat;
    }

    /*
   float fx,fy,fz;
   intersection.getValue(fx,fy,fz);
   cerr << "TP: (" << fx << "," << fy << "," << fz << ")" << endl;
   */
}

//======================================================================
//
// Description:
//	make telepointer visible
//
//
// Use: private
//======================================================================
void TPHandler::setTelePointer(TelePointer *tp, const SbVec3f point)
{
    tp->tp_translation->translation.setValue(point);
    tp->tp_icon->string.setValue(tp->getName());
}

//======================================================================
//
// Description:
//	remove telepointer from the screen
//
//
// Use: private
//======================================================================
void TPHandler::clearTelePointer(TelePointer *tp)
{
    tp->tp_icon->string.setValue("");
}
