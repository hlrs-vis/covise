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
 * ThumbWheel.h - public header file for thumbwheel widget
 */

#ifndef _SG_ThumbWheel_h
#define _SG_ThumbWheel_h

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SgNlowerBound
#define SgNlowerBound (char *) "minimum"
#endif
/* SgNlowerBound is OBSOLETE!  Please use XmNminimum! */

#ifndef SgNupperBound
#define SgNupperBound (char *) "maximum"
#endif
/* SgNupperBound is OBSOLETE!  Please use XmNminimum! */

#ifndef SgNhomePosition
#define SgNhomePosition (char *) "homePosition"
#endif
#ifndef SgNangleRange
#define SgNangleRange (char *) "angleRange"
#endif
#ifndef SgNunitsPerRotation
#define SgNunitsPerRotation (char *) "unitsPerRotation"
#endif
#ifndef SgNanimate
#define SgNanimate (char *) "animate"
#endif
#ifndef SgNshowHomeButton
#define SgNshowHomeButton (char *) "showHomeButton"
#endif

#ifndef SgCLowerBound
#define SgCLowerBound (char *) "Minimum"
#endif
/* SgCLowerBound is OBSOLETE!  Please use XmCMinimum! */

#ifndef SgCUpperBound
#define SgCUpperBound (char *) "Maximum"
#endif
/* SgCUpperBound is OBSOLETE!  Please use XmCMaximum! */

#ifndef SgCHomePosition
#define SgCHomePosition (char *) "HomePosition"
#endif
#ifndef SgCAngleRange
#define SgCAngleRange (char *) "AngleRange"
#endif
#ifndef SgCUnitsPerRotation
#define SgCUnitsPerRotation (char *) "UnitsPerRotation"
#endif
#ifndef SgCAnimate
#define SgCAnimate (char *) "Animate"
#endif
#ifndef SgCShowHomeButton
#define SgCShowHomeButton (char *) "ShowHomeButton"
#endif

extern WidgetClass sgThumbWheelWidgetClass;

typedef struct _SgThumbWheelClassRec *SgThumbWheelWidgetClass;
typedef struct _SgThumbWheelRec *SgThumbWheelWidget;

/********    Public Function Declarations    ********/
#ifdef _NO_PROTO

extern Widget SgCreateThumbWheel();

#else

extern Widget SgCreateThumbWheel(Widget parent,
                                 char *name,
                                 ArgList arglist,
                                 Cardinal argcount);
#endif /* _NO_PROTO */
/********    End Public Function Declarations    ********/

/*
    * Structure for both callbacks (Drag and Value Changed).
    */
typedef struct
{
    int reason;
    XEvent *event;
    int value;
} SgThumbWheelCallbackStruct;

#ifdef __cplusplus
} /* Close scope of 'extern "C"' declaration which encloses file. */
#endif

/* DON'T ADD ANYTHING AFTER THIS #ENDIF */
#endif /* _SG_ThumbWheel_h */
