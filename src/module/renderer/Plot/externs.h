/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Id: externs.h,v 1.1 1994/05/13 01:29:47 pturner Exp $
 *
 *      declarations for draw.c
 *
 */

extern int devorient; /* device has 0,0 at upper left if true */
extern int devwidthmm; /* device width in mm */
extern int devwidth; /* device number of points in width */
extern int devheightmm; /* device height in mm */
extern int devheight; /* device number of points in height */
extern int devoffsx; /* device offset in x (if not 0) */
extern int devoffsy; /* device offset in y (if not 0) */
extern int devxticl, devyticl; /* common length for device */
extern int devarrowlength; /* length for arrow device */
extern int devsymsize; /* default symbol size */
extern int devcharh, devcharw; /* typical character height and width */
extern int (*devsetcolor)(int); /* routine to set colors */
extern int (*devconvx)(double); /* map world x to device */
extern int (*devconvy)(double); /* map world y to device y */
extern void (*devvector)(int, int, int); /* device line routine */
/* device text drawing */
extern void (*devwritestr)(int, int, int, char *, int, int);
extern void (*devdrawtic)(int, int, int, int); /* draw ticks using device draw */
extern void (*devleavegraphics)(); /* device exit */
extern int (*devsetline)(int); /* device set line style */
extern int (*devsetlinew)(int); /* device set line width */
extern void (*devsetfont)(int); /* set device font */
extern int (*devsetpat)(int); /* device set fill pattern */
extern void (*devdrawarc)(int, int, int); /* device arc routine */
extern void (*devfillarc)(int, int, int); /* device fill arc routine */
/* device ellipse routine */
extern void (*devdrawellipse)(int, int, int, int);
/* device ellipse arc routine */
extern void (*devfillellipse)(int, int, int, int);
extern void (*devfill)(int, int *, int *); /* device fill routine */
extern void (*devfillcolor)(int, int *, int *); /* device fill color routine */

#define MAXLINELEN 800
/* defined in xvlib.c */
extern int xpoints[MAXLINELEN], ypoints[MAXLINELEN];
extern int pathlength;
