/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_UI_REGION_
#define _INV_UI_REGION_

/* $Id: InvUIRegion.h,v 1.1 1994/04/12 13:39:31 zrfu0125 Exp zrfu0125 $ */

/* $Log: InvUIRegion.h,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

#include <Inventor/SbBasic.h>

/*
 * Defines
 */

//
// list of grey colors used when drawing regions
//
#define WHITE_UI_COLOR glColor3ub(255, 255, 255)
#define BLACK_UI_COLOR glColor3ub(0, 0, 0)
#define MAIN_UI_COLOR glColor3ub(170, 170, 170)
#define DARK1_UI_COLOR glColor3ub(128, 128, 128)
#define DARK2_UI_COLOR glColor3ub(85, 85, 85)
#define DARK3_UI_COLOR glColor3ub(50, 50, 50)
#define LIGHT1_UI_COLOR glColor3ub(215, 215, 215)

#define UI_THICK 3

#define SO_UI_REGION_GREY1 glColor3ub(240, 240, 240)
#define SO_UI_REGION_GREY2 glColor3ub(190, 190, 190)
#define SO_UI_REGION_GREY3 glColor3ub(150, 150, 150)
#define SO_UI_REGION_GREY4 glColor3ub(130, 130, 130)
#define SO_UI_REGION_GREY5 glColor3ub(110, 110, 110)
#define SO_UI_REGION_GREY6 glColor3ub(70, 70, 70)
#define SO_UI_REGION_GREY7 glColor3ub(30, 30, 30)

/*
 * Function prototypes
 */

extern void
drawDownUIRegion(short x1, short y1, short x2, short y2);

extern void
drawDownUIBorders(short x1, short y1, short x2, short y2, SbBool blackLast = FALSE);

extern void
drawThumbUIRegion(short x1, short y1, short x2, short y2);
#endif // _INV_UI_REGION_
