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
 * ThumbWheelP.h - private header file for empty widget
 */

#ifndef _SG_ThumbWheelP_h
#define _SG_ThumbWheelP_h

#include <Xm/XmP.h>
#include <Xm/PrimitiveP.h>
#include <Xm/DrawP.h>
#include "ThumbWheel.h"

typedef struct
{
    int make_compiler_happy;
#ifdef __sgi
    caddr_t _SG_vendorExtension;
#endif /* __sgi */
} SgThumbWheelClassPart;

typedef struct
{
    CoreClassPart core_class;
    XmPrimitiveClassPart primitive_class;
    SgThumbWheelClassPart thumbWheel_class;
} SgThumbWheelClassRec;

extern SgThumbWheelClassRec sgThumbWheelClassRec;

typedef struct
{
    /* resources */
    int lower_bound;
    int upper_bound;
    int home_position;
    int angle_range;
    int angle_factor;
    int value;
    unsigned char orientation;
    Boolean animate;
    Boolean show_home_button;

    XtCallbackList value_changed_callback;
    XtCallbackList drag_callback;

    /* private state */
    Boolean infinite;
    Boolean dragging;
    int drag_begin_value;
    int last_mouse_position;
    Boolean pegged;
    int pegged_mouse_position;
    Dimension viewable_pixels;
    int user_pixels;
    Pixmap pix1;
    Pixmap pix2;
    Pixmap pix3;
    Pixmap pix4;
    Pixmap pix1_hilite;
    Pixmap pix2_hilite;
    Pixmap pix3_hilite;
    Pixmap pix4_hilite;
    Pixmap current_quiet_pixmap; /* this will be equal to one of the others */
    Pixmap current_hilite_pixmap; /* this will be equal to one of the others */
    Boolean wheel_hilite;

    Pixmap button_quiet_pixmap;
    Pixmap button_hilite_pixmap;
    Boolean button_hilite;

    GC foreground_GC;
#ifdef __sgi
    shaderptr shader;
#endif /* __sgi */

    int wheel_x;
    int wheel_y;
    int button_x;
    int button_y;

    Boolean home_button_armed;

#ifdef __sgi
    caddr_t _SG_vendorExtension;
#endif /* __sgi */
} SgThumbWheelPart;

typedef struct _SgThumbWheelRec
{
    CorePart core;
    XmPrimitivePart primitive;
    SgThumbWheelPart thumbWheel;
} SgThumbWheelRec;

/* DON'T ADD ANYTHING AFTER THIS #ENDIF */
#endif /* _SG_ThumbWheelP_h */
