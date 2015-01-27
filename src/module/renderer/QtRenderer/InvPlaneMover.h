/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:  InvPlaneMover.h                                       ++
// ++               Class which supplies a handle to move plane like      ++
// ++               objects                                               ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 03.08.2001                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ifndef INVPLANEMOVER_H
#define INVPLANEMOVER_H

#include <util/coTypes.h>
#include <iostream>
#include <stdlib.h>
using namespace std;

// include Inventor NODES
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoSelection.h>
#include <Inventor/nodes/SoDrawStyle.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <Inventor/nodes/SoRotation.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoScale.h>

#include <Inventor/draggers/SoJackDragger.h>

#include <Inventor/SoPickedPoint.h>

#include <Inventor/SbBox.h>

#include "InvViewer.h"

class InvPlaneMover
{
public:
    static enum
    {
        SNAP,
        FREE
    } mode;

    //InvPlaneMover( InvViewer *);
    //InvViewer   *v;

    InvPlaneMover();

    // convienience method it will set all nessesarry callbacks;
    void bind(SoSelection *sel)
    {
        (void)sel;
    };

    SoSeparator *getSeparator() const
    {
        return handle_;
    };

    // show and activate the handle
    void show();

    // hide and deactivate the handle
    void hide();

    // set size of the handle parts using the bounding box of the plane to be moved
    void setSize(const SbBox3f &bb);

    // call this callback either from application level selection CB
    // or use it as selection CB
    static void selectionCB(void *me, SoPath *sp);

    // call this callback either from application level deselection CB
    // or use it as deselection CB
    static void deSelectionCB(void *me, SoPath *);

    // call this callback either from application level pickFilter CB
    // or use it as pickFilter CB
    static SoPath *pickFilterCB(void *me, const SoPickedPoint *pick);

    // set the picked point position and normal
    void setPosition(SbVec3f &point);

    // set snap to axis mode
    void setSnapToAxis();

    // set free motion (default)
    void setFreeMotion();

    ~InvPlaneMover();

private:
    /*InvPlaneMover() {};
      // we don't want to be copied or assigned -> live clean and easy <-
      InvPlaneMover(const InvPlaneMover &p) { cerr << "ERROR: InvPlaneMover copy-constructor called" << endl;};
      const InvPlaneMover& operator= (const InvPlaneMover& p) {return p;}*/

    int show_;
    SbVec3f distOffset_;
    char *feedbackInfo_;
    SbVec3f nnn_;
    SbVec3f planeNormal_;

    SoSeparator *handle_;
    SoSwitch *handleSwitch_;
    SoDrawStyle *handleDrawStyle_;
    SoJackDragger *jDrag_;

    SbVec3f iNorm_;
    SbVec3f tStart_;
    SoTranslation *transl_;
    SoRotation *fullRot_;
    SoScale *scale_;

    SoGroup *makeArrow();
    SoGroup *makePlane();

    int motionMode_;

    // internal callback
    static void dragFinishCB(void *me, SoDragger *drag);

    void dragStartCB(void *me, SoDragger *drag);

    // send feedback msg. tzo the contoller (old style; only works for CuttingSurface)
    void sendFeedback(float *data);
};
#endif
