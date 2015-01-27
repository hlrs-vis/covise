/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_TELE_POINTER_H
#define _INV_TELE_POINTER_H

//x**************************************************************************
//
//x * Description    : Inventor telepointer stuff
//
//x * Class(es)      : InvTelePointer
//
//x**************************************************************************

#include <util/coTypes.h>

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

class QString;

class InvTelePointer;
class TPHandler;

#include "InvDefs.h"
#include "InvViewer.h"
#ifndef YAC
#include "InvMain.h"
#else
#include "InvMain_yac.h"
#endif

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

    QString name;
    void makeTelePointer();
    void doGLpre();
    void doGLpost();

    static void drawManipPre(void *d, SoAction *action);
    static void drawManipPost(void *d, SoAction *action);

public:
    friend class TPHandler;

    TelePointer(QString Name)
    {
        name = "< ";
        name.append(Name);
        makeTelePointer();
    };

    QString getName()
    {
        return name;
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

    void setTelePointer(TelePointer *tp, const SbVec3f point);
    void clearTelePointer(TelePointer *tp);
    void projectTelePointer(TelePointer *tp, const SbVec3f pos, float aspectRatio,
                            SbVec3f &intersection);

public:
    friend class TelePointer;

    TPHandler();

    SoSeparator *getRoot();

    SoCamera *getCamera()
    {
        return tp_camera;
    };

    void handle(const char *TPmessage);

    ~TPHandler()
    {
        delete tpList;
    };
};
#endif /*  _INV_TELE_POINTER_H */
