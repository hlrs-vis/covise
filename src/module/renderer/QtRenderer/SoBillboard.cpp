/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 *
 *  Copyright (C) 2000 Silicon Graphics, Inc.  All Rights Reserved.
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  Further, this software is distributed without any warranty that it is
 *  free of the rightful claim of any third person regarding infringement
 *  or the like.  Any license provided herein, whether implied or
 *  otherwise, applies only to this software file.  Patent licenses, if
 *  any, provided herein do not apply to combinations of this program with
 *  other software, or any other product whatsoever.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *  Contact information: Silicon Graphics, Inc., 1600 Amphitheatre Pkwy,
 *  Mountain View, CA  94043, or:
 *
 *  http://www.sgi.com
 *
 *  For further information regarding this notice, see:
 *
 *  http://oss.sgi.com/projects/GenInfo/NoticeExplan/
 *
 */

/*
 * Copyright (C) 1990,91   Silicon Graphics, Inc.
 *
 _______________________________________________________________________
 ______________  S I L I C O N   G R A P H I C S   I N C .  ____________
 |
 |   $Revision: 1.1.1.1 $
 |
 |   Classes:
 |      SoBillboard
 |
|   Author(s)          : Paul S. Strauss
|
______________  S I L I C O N   G R A P H I C S   I N C .  ____________
_______________________________________________________________________
*/

#include <Inventor/actions/SoCallbackAction.h>
#include <Inventor/actions/SoGLRenderAction.h>
#include <Inventor/actions/SoGetBoundingBoxAction.h>
#include <Inventor/actions/SoGetMatrixAction.h>
#include <Inventor/actions/SoPickAction.h>
#include <Inventor/elements/SoModelMatrixElement.h>
#include <Inventor/elements/SoViewingMatrixElement.h>
#include <Inventor/elements/SoViewVolumeElement.h>
#include "SoBillboard.h"

SO_NODE_SOURCE(SoBillboard);

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Constructor
//
// Use: public

SoBillboard::SoBillboard()
//
////////////////////////////////////////////////////////////////////////
{
    SO_NODE_CONSTRUCTOR(SoBillboard);
    SO_NODE_ADD_FIELD(axis, (0.0, 0.0, 0.0));
    //isBuiltIn = TRUE;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Destructor
//
// Use: private

SoBillboard::~SoBillboard()
//
////////////////////////////////////////////////////////////////////////
{
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Implements most actions.
//
// Use: extender

void
SoBillboard::doAction(SoAction *action)
//
////////////////////////////////////////////////////////////////////////
{
    SbRotation rot = calculateRotation(action->getState());
    //if (! rotation.isIgnored() && ! rotation.isDefault())
    SoModelMatrixElement::rotateBy(action->getState(), this, rot);
    //rotation.getValue());
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Handles callback action
//
// Use: extender

void
SoBillboard::callback(SoCallbackAction *action)
//
////////////////////////////////////////////////////////////////////////
{
    SoBillboard::doAction(action);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Handles GL render action
//
// Use: extender

void
SoBillboard::GLRender(SoGLRenderAction *action)
//
////////////////////////////////////////////////////////////////////////
{
    SoBillboard::doAction(action);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Handles get bounding box action
//
// Use: extender

void
SoBillboard::getBoundingBox(SoGetBoundingBoxAction *action)
//
////////////////////////////////////////////////////////////////////////
{
    SoBillboard::doAction(action);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Returns transformation matrix.
//
// Use: extender

void
SoBillboard::getMatrix(SoGetMatrixAction *action)
//
////////////////////////////////////////////////////////////////////////
{
    //if (! rotation.isIgnored() && ! rotation.isDefault()) {
    SbRotation rot = calculateRotation(action->getState());
    SbMatrix &ctm = action->getMatrix();
    SbMatrix &inv = action->getInverse();
    SbMatrix m;

    rot.getValue(m);
    ctm.multLeft(m);
    rot.invert();
    rot.getValue(m);
    inv.multRight(m);
    //}
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Handles pick action.
//
// Use: extender

void
SoBillboard::pick(SoPickAction *action)
//
////////////////////////////////////////////////////////////////////////
{
    SoBillboard::doAction(action);
}

void
SoBillboard::initClass()
{
#ifdef __COIN__
    SO_NODE_INIT_CLASS(SoBillboard, SoTransformation, "Transformation");
#endif
}

SbRotation
SoBillboard::calculateRotation(SoState *state)
{
    SbRotation rot;
#ifdef INVENTORRENDERER
    const SbViewVolume &viewVolume = SoViewVolumeElement::get(state);

    if (SbVec3f(0.0f, 0.0f, 0.0f) == axis.getValue())
    {
        rot = viewVolume.getAlignRotation();
    }
#else
    const SbMatrix &mm = SoModelMatrixElement::get(state);
    SbMatrix imm = mm.inverse();

    SbVec3f toviewer;
    SbVec3f cameray(0.0f, 1.0f, 0.0f);
    const SbViewVolume &vv = SoViewVolumeElement::get(state);

    toviewer = -vv.getProjectionDirection();
    imm.multDirMatrix(toviewer, toviewer);

    (void)toviewer.normalize();

    SbVec3f rotaxis = this->axis.getValue();

    if (rotaxis == SbVec3f(0.0f, 0.0f, 0.0f))
    {
        // 1. Compute the billboard-to-viewer vector.
        // 2. Rotate the Z-axis of the billboard to be collinear with the
        //    billboard-to-viewer vector and pointing towards the viewer's position.
        // 3. Rotate the Y-axis of the billboard to be parallel and oriented in the
        //    same direction as the Y-axis of the viewer.
        rot.setValue(SbVec3f(0.f, 0.0f, 1.0f), toviewer);
        SbVec3f viewup = vv.getViewUp();
        imm.multDirMatrix(viewup, viewup);

        SbVec3f yaxis(0.0f, 1.0f, 0.0f);
        rot.multVec(yaxis, yaxis);
        SbRotation rot2(yaxis, viewup);

        SbVec3f axis;
        float angle;
        rot.getValue(axis, angle);
        rot2.getValue(axis, angle);
        rot = rot * rot2;
        //SoModelMatrixElement::rotateBy(state, (SoNode*) this, rot);
    }
#endif
    else
    {
        fprintf(stderr, "SoBillboard: axis != (0.0, 0.0, 0.0) not implemented\n");
    }

    return rot;
}
