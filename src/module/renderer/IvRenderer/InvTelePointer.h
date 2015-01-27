/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_TELE_POINTER_H
#define _INV_TELE_POINTER_H

/* $Id: InvTelePointer.h,v 1.1 1994/04/12 13:39:31 zrfu0125 Exp zrfu0125 $ */

/* $Log: InvTelePointer.h,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

//x**************************************************************************
//
//x * Description    : Inventor telepointer stuff
//
//x * Class(es)      : InvTelePointer
//
//x * inherited from :
//x
//x * Author  : Dirk Rantzau
//
//x * History : 07.04.94 V 1.0
//
//x**************************************************************************
#include <Inventor/SoDB.h>
#include <Inventor/SoInput.h>
#include <Inventor/nodes/SoNode.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoCamera.h>
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoFont.h>
#include <Inventor/nodes/SoText2.h>

#include <covise/covise.h>

class InvTelePointer;
class InvExaminerViewer;
class TPHandler;

#include "InvDefs.h"
#include "InvExaminerViewer.h"

//==========================================================================
// TelePointer class
//==========================================================================

class TelePointer
{

private:
    SoSeparator *tp_shape;
    SoText2 *tp_icon;
    SoMaterial *tp_color;
    SoFont *tp_font;
    SoTranslation *tp_translation;
    SoTranslation *tp_transIcon;
    SoCallback *drawCBPre_;
    SoCallback *drawCBPost_;

    char name[64];
    void makeTelePointer();
    void doGLpre();
    void doGLpost();

    static void drawManipPre(void *d, SoAction *action);
    static void drawManipPost(void *d, SoAction *action);

public:
    friend class TPHandler;

    TelePointer(char *Name)
    {
        strcpy(name, "< ");
        strcat(name, Name);
        makeTelePointer();
    };
    char *getName()
    {
        return &name[0];
    };
    ~TelePointer(){};
};

//==========================================================================
// TelePointer handler
//==========================================================================
class TPHandler
{

private:
    SbPList *tpList;
    SoSeparator *tp_sceneGraph;
    SoOrthographicCamera *tp_camera;

    void projectTelePointer(TelePointer *tp, const SbVec3f pos, float aspectRatio,
                            SbVec3f &intersection, InvExaminerViewer *viewer);
    void setTelePointer(TelePointer *tp, const SbVec3f point);
    void clearTelePointer(TelePointer *tp);

public:
    friend class TelePointer;

    TPHandler(InvExaminerViewer *viewer);
    TPHandler();

    SoSeparator *getRoot();

    void handle(char *TPmessage, InvExaminerViewer *currentViewer);
    SoCamera *getCamera()
    {
        return (SoCamera *)tp_camera;
    };

    ~TPHandler()
    {
        delete tpList;
    };
};
#endif /*  _INV_TELE_POINTER_H */
