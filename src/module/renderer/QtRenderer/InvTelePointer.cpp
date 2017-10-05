/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


// **************************************************************************
//
// * Description    : Inventor telepointer stuff
//
// * Class(es)      : InvTelePointer
//
// **************************************************************************

#ifndef YAC
#include <covise/covise_process.h>
#endif

#ifdef WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif
#include <iostream>

#include <iostream>

#include <util/coTypes.h>
#include <qstring.h>

#include "InvTelePointer.h"
#include "InvError.h"

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include <Inventor/nodes/SoAnnotation.h>
#include <Inventor/elements/SoCacheElement.h>

extern int port;
extern QString host;

//======================================================================
//
// Description:
//	create telepointer geometry
//
//======================================================================
void TelePointer::makeTelePointer()
{

    tp_shape = new SoSeparator;
    tp_color = new SoMaterial;
    tp_font = new SoFont;
    tp_icon = new SoText2;
    tp_translation = new SoTranslation;
    tp_transIcon = new SoTranslation;

#ifndef __sgi
    drawCBPre_ = new SoCallback;
    drawCBPre_->setCallback(drawManipPre, this);

    drawCBPost_ = new SoCallback;
    drawCBPost_->setCallback(drawManipPost, this);
#endif

    tp_font->name.setValue("Times-Roman");
    tp_font->size.setValue(24.0);
    tp_transIcon->translation.setValue(0.0f, -0.025f, 0.0f);
    tp_icon->string.setValue(name.toLatin1());

    tp_shape->addChild(tp_color);
    tp_shape->addChild(tp_translation);
    tp_shape->addChild(tp_font);
    tp_shape->addChild(tp_transIcon);

#ifndef __sgi
    tp_shape->addChild(drawCBPre_);
#endif

    tp_shape->addChild(tp_icon);

#ifndef __sgi
    tp_shape->addChild(drawCBPost_);
#endif
}

void TelePointer::drawManipPre(void *d, SoAction *action)
{
    if (action->isOfType(SoGLRenderAction::getClassTypeId()))
    {
        // Make my custom GL calls
        ((TelePointer *)d)->doGLpre();

        // Invalidate the state so that a cache is not made
        SoCacheElement::invalidate(action->getState());
    }
}

void TelePointer::drawManipPost(void *d, SoAction *action)
{
    if (action->isOfType(SoGLRenderAction::getClassTypeId()))
    {
        // Make my custom GL calls
        ((TelePointer *)d)->doGLpost();

        // Invalidate the state so that a cache is not made
        SoCacheElement::invalidate(action->getState());
    }
}

void TelePointer::doGLpre()
{
    //    g << "TelePointer::doGLpre() called" << endl;
    glDisable(GL_DEPTH_TEST);
}

void TelePointer::doGLpost()
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

SoSeparator *TPHandler::getRoot()
{
    return tp_sceneGraph;
}

//======================================================================
//
// Description:
//	handle routine for telepointers
//
//======================================================================
void TPHandler::handle(const char *TPmessage)
{
    TelePointer *ptr, *tp = NULL;
    short found;
    char tpname[100];
    char tpLabel[34];
    int state;
    float px, py, pz;
    float aspectRatio;
    SbVec3f intersection;
    SbVec3f vec;

    // scan message
    int retval;
    retval = sscanf(TPmessage, "%s %d %f %f %f %f", &tpname[0], &state, &px, &py, &pz, &aspectRatio);
    if (retval != 6)
    {
        std::cerr << "TPHandler::handle: sscanf failed" << std::endl;
        return;
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
        if (ptr->getName() == tpLabel)
        {
            // found !
            found = TRUE;
            tp = ptr;
        }

        else
        {
            i++;
        }
    }

    //if( (!found) && (state != CO_RMV))
    if ((!found) && (state == CO_ON))
    {
        // add a new TP to list
        TelePointer *tp_new = new TelePointer(tpname);
        tpList->append((void *)tp_new);

        // append to scenegraph
        tp_sceneGraph->addChild(tp_new->tp_shape);
        tp = tp_new;
    }

    switch (state)
    {
    /*case CO_OFF:
         // hide it
         if(tp)
            clearTelePointer(tp);
      break;*/

    case CO_ON:
        // set new translation
        // show it
        vec.setValue(px, py, pz);
        if (tp)
        {
            projectTelePointer(tp, vec, aspectRatio, intersection);
            setTelePointer(tp, intersection);
        }
        break;

    case CO_RMV:
        if (found)
        {
            tpList->remove(i);
            if (tp)
            {
                tp_sceneGraph->removeChild(tp->tp_shape);
            }
        }
        break;

    } // end switch
}

//======================================================================
//
// Description:
//	make telepointer visible
//
//======================================================================
void TPHandler::setTelePointer(TelePointer *tp, const SbVec3f point)
{
    tp->tp_translation->translation.setValue(point);
    tp->tp_icon->string.setValue(tp->getName().toLatin1());
}

//======================================================================
//
// Description:
//	remove telepointer from the screen
//
//======================================================================
void TPHandler::clearTelePointer(TelePointer *tp)
{
    tp->tp_icon->string.setValue("");
}

//======================================================================
//
// Description:
//	projection of telepointer
//
//======================================================================
void TPHandler::projectTelePointer(TelePointer *, const SbVec3f pos, float aspectRatio,
                                   SbVec3f &intersection)
{
    const int xOffset = 61;
    const int yOffset = 33;

    float xs, ys, zs; // normalized screen coordinates
    SbVec3f screen;

    SbVec2s size = renderer->viewer->getSize();
    if (renderer->viewer->isDecoration())
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

    // get the view volume -> rectangular box
    SbViewVolume viewVolume = tp_camera->getViewVolume();

    // determine mouse position
    viewVolume.projectToScreen(pos, screen);
    screen.getValue(xs, ys, zs);

    // project the mouse point to a line
    SbVec3f p0, p1;
    viewVolume.projectPointToLine(SbVec2f(xs, ys), p0, p1);

    // take the midpoint of the line as telepointer position
    intersection = (p0 + p1) / (float)2.0;

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
}
