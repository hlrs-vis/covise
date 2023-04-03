/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* ********************************************************************** */

/* xvertext 5.0, Copyright (c) 1993 Alan Richardson (mppa3@uk.ac.sussex.syma)
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose and without fee is hereby granted, provided
 * that the above copyright notice appear in all copies and that both the
 * copyright notice and this permission notice appear in supporting
 * documentation.  All work developed as a consequence of the use of
 * this program should duly acknowledge such use. No representations are
 * made about the suitability of this software for any purpose.  It is
 * provided "as is" without express or implied warranty.
 */

/* ********************************************************************** */

/* BETTER: xvertext now does rotation at any angle!!
 *
 * BEWARE: function arguments have CHANGED since version 2.0!!
 */

/* ********************************************************************** */

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <stdio.h>
#include <math.h>
#include "extern.h"
#include "rotated.h"
#include "globals.h"
#include <Xm/Xm.h>
// #include "extern2.h"

/* ---------------------------------------------------------------------- */

/* Make sure cache size is set */

#ifndef CACHE_SIZE_LIMIT
#define CACHE_SIZE_LIMIT 0
#endif /*CACHE_SIZE_LIMIT */

/* Make sure a cache method is specified */

#ifndef CACHE_XIMAGES
#ifndef CACHE_BITMAPS
#define CACHE_BITMAPS
#endif /*CACHE_BITMAPS*/
#endif /*CACHE_XIMAGES*/

/* ---------------------------------------------------------------------- */

/* Debugging macros */

#ifdef DEBUG
static int debug = 1;
#else
static int debug = 0;
#endif /*DEBUG*/

#define DEBUG_PRINT1(a) \
    if (debug)          \
    printf(a)
#define DEBUG_PRINT2(a, b) \
    if (debug)             \
    printf(a, b)
#define DEBUG_PRINT3(a, b, c) \
    if (debug)                \
    printf(a, b, c)
#define DEBUG_PRINT4(a, b, c, d) \
    if (debug)                   \
    printf(a, b, c, d)
#define DEBUG_PRINT5(a, b, c, d, e) \
    if (debug)                      \
    printf(a, b, c, d, e)

/* ---------------------------------------------------------------------- */
#define M_PIF 3.141592653589793238462643383279502884e+00F
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---------------------------------------------------------------------- */

/* A structure holding everything needed for a rotated string */

typedef struct rotated_text_item_template
{
    Pixmap bitmap;
    XImage *ximage;

    char *text;
    char *font_name;
    Font fid;
    float angle;
    int align;
    float magnify;

    int cols_in;
    int rows_in;
    int cols_out;
    int rows_out;

    int nl;
    int max_width;
    float *corners_x;
    float *corners_y;

    long int size;
    int cached;

    struct rotated_text_item_template *next;
} RotatedTextItem;

RotatedTextItem *first_text_item = NULL;

/* ---------------------------------------------------------------------- */

/* A structure holding current magnification and bounding box padding */

static struct style_template
{
    float magnify;
    int bbx_pad;
}

style = {
    1.,
    0
};

/* ---------------------------------------------------------------------- */

// static char            *my_strdup();
// static char            *my_strtok();

float XRotVersion();
void XRotSetMagnification();
void XRotSetBoundingBoxPad();
int XRotDrawString();
int XRotDrawImageString();
int XRotDrawAlignedString();
int XRotDrawAlignedImageString();

// static XImage          *MakeXImage();
// static RotatedTextItem *XRotCreateTextItem();
static void XRotAddToLinkedList(Display *dpy, RotatedTextItem *item);

XPoint *XRotTextExtents(Display *dpy, XFontStruct *font, float angle, int x, int y, char *text, int align);
static int XRotPaintAlignedString(Display *dpy, XFontStruct *font, float angle, Drawable drawable, GC gc, int x, int y, char *text, int align, int bg);
static XImage *XRotMagnifyImage(Display *dpy, XImage *ximage);
static RotatedTextItem *XRotCreateTextItem(Display *dpy, XFontStruct *font, float angle, char *text, int align);
static RotatedTextItem *XRotRetrieveFromCache(Display *dpy, XFontStruct *font, float angle, char *text, int align);
static int XRotDrawHorizontalString(Display *dpy, XFontStruct *font, Drawable drawable, GC gc, int x, int y, char *text, int align, int bg);
static void XRotFreeTextItem(Display *dpy, RotatedTextItem *item);

/* ---------------------------------------------------------------------- */

/**************************************************************************/
/* Routine to mimic `strdup()' (some machines don't have it)              */
/**************************************************************************/

static char *my_strdup(char *str)
{
    char *s;

    if (str == NULL)
        return NULL;

    s = (char *)malloc((unsigned)(strlen(str) + 1));
    if (s != NULL)
        strcpy(s, str);

    return s;
}

/* ---------------------------------------------------------------------- */

/**************************************************************************/
/* Routine to replace `strtok' : this one returns a zero length string if */
/* it encounters two consecutive delimiters                               */
/**************************************************************************/

static char *my_strtok(char *str1, char *str2)
{
    char *ret;
    int i, j, stop;
    static int start, len;
    static char *stext;

    if (str2 == NULL)
        return NULL;

    /* initialise if str1 not NULL */
    if (str1 != NULL)
    {
        start = 0;
        stext = str1;
        len = strlen(str1);
    }

    /* run out of tokens ? */
    if (start >= len)
        return NULL;

    /* loop through characters */
    for (i = start; i < len; i++)
    {
        /* loop through delimiters */
        stop = 0;
        for (j = 0; j < strlen(str2); j++)
            if (stext[i] == str2[j])
                stop = 1;

        if (stop)
            break;
    }

    stext[i] = '\0';

    ret = stext + start;

    start = i + 1;

    return ret;
}

/* ---------------------------------------------------------------------- */

/**************************************************************************/
/* Return version/copyright information                                   */
/**************************************************************************/

float XRotVersion(char *str, int n)
{
    if (str != NULL)
        strncpy(str, XV_COPYRIGHT, n);
    return XV_VERSION;
}

/* ---------------------------------------------------------------------- */

/**************************************************************************/
/* Set the font magnification factor for all subsequent operations        */
/**************************************************************************/

void XRotSetMagnification(float m)
{
    if (m > 0.)
        style.magnify = m;
}

/* ---------------------------------------------------------------------- */

/**************************************************************************/
/* Set the padding used when calculating bounding boxes                   */
/**************************************************************************/

void XRotSetBoundingBoxPad(int p)
{
    if (p >= 0)
        style.bbx_pad = p;
}

/* ---------------------------------------------------------------------- */

/**************************************************************************/
/*  Create an XImage structure and allocate memory for it                 */
/**************************************************************************/

static XImage *MakeXImage(Display *dpy, int w, int h)
{
    XImage *I;
    char *data;

    /* reserve memory for image */
    data = (char *)calloc((unsigned)(((w - 1) / 8 + 1) * h), 1);
    if (data == NULL)
        return NULL;

    /* create the XImage */
    I = XCreateImage(dpy, DefaultVisual(dpy, DefaultScreen(dpy)), 1, XYBitmap,
                     0, data, w, h, 8, 0);
    if (I == NULL)
        return NULL;

    I->byte_order = I->bitmap_bit_order = MSBFirst;
    return I;
}

/* ---------------------------------------------------------------------- */

/**************************************************************************/
/*  A front end to XRotPaintAlignedString:                                */
/*      -no alignment, no background                                      */
/**************************************************************************/

int XRotDrawString(Display *dpy, XFontStruct *font, float angle, Drawable drawable, GC gc, int x, int y, char *str)
{
    return (XRotPaintAlignedString(dpy, font, angle, drawable, gc,
                                   x, y, str, ALIGNNONE, 0));
}

/* ---------------------------------------------------------------------- */

/**************************************************************************/
/*  A front end to XRotPaintAlignedString:                                */
/*      -no alignment, paints background                                  */
/**************************************************************************/

int XRotDrawImageString(Display *dpy, XFontStruct *font, float angle, Drawable drawable, GC gc, int x, int y, char *str)
{
    return (XRotPaintAlignedString(dpy, font, angle, drawable, gc,
                                   x, y, str, ALIGNNONE, 1));
}

/* ---------------------------------------------------------------------- */

/**************************************************************************/
/*  A front end to XRotPaintAlignedString:                                */
/*      -does alignment, no background                                    */
/**************************************************************************/

int XRotDrawAlignedString(Display *dpy, XFontStruct *font, float angle, Drawable drawable, GC gc, int x, int y, char *text, int align)
{
    return (XRotPaintAlignedString(dpy, font, angle, drawable, gc,
                                   x, y, text, align, 0));
}

/* ---------------------------------------------------------------------- */

/**************************************************************************/
/*  A front end to XRotPaintAlignedString:                                */
/*      -does alignment, paints background                                */
/**************************************************************************/

int XRotDrawAlignedImageString(Display *dpy, XFontStruct *font, float angle, Drawable drawable, GC gc, int x, int y, char *text, int align)

{
    return (XRotPaintAlignedString(dpy, font, angle, drawable, gc,
                                   x, y, text, align, 1));
}

/* ---------------------------------------------------------------------- */

/**************************************************************************/
/*  Aligns and paints a rotated string                                    */
/**************************************************************************/

static int XRotPaintAlignedString(Display *dpy, XFontStruct *font, float angle, Drawable drawable, GC gc, int x, int y, char *text, int align, int bg)
{
    int i;
    GC my_gc;
    int xp, yp;
    float hot_x, hot_y;
    float hot_xp, hot_yp;
    float sin_angle, cos_angle;
    RotatedTextItem *item;
    Pixmap bitmap_to_paint;

    /* return early for NULL/empty strings */
    if (text == NULL)
        return 0;

    if (strlen(text) == 0)
        return 0;

    /* manipulate angle to 0<=angle<360 degrees */
    while (angle < 0)
        angle += 360;

    while (angle >= 360)
        angle -= 360;

    angle *= M_PIF / 180;

    /* horizontal text made easy */
    if (angle == 0. && style.magnify == 1.)
        return (XRotDrawHorizontalString(dpy, font, drawable, gc, x, y,
                                         text, align, bg));

    /* get a rotated bitmap */
    item = XRotRetrieveFromCache(dpy, font, angle, text, align);
    if (item == NULL)
        return 0;

    /* this gc has similar properties to the user's gc */
    my_gc = XCreateGC(dpy, drawable, 0, 0);
    XCopyGC(dpy, gc, GCForeground | GCBackground | GCFunction | GCPlaneMask,
            my_gc);

    /* alignment : which point (hot_x, hot_y) relative to bitmap centre
      coincides with user's specified point? */

    /* y position */
    if (align == TLEFT || align == TCENTRE || align == TRIGHT)
        hot_y = (float)item->rows_in / 2 * style.magnify;
    else if (align == MLEFT || align == MCENTRE || align == MRIGHT)
        hot_y = 0;
    else if (align == BLEFT || align == BCENTRE || align == BRIGHT)
        hot_y = -(float)item->rows_in / 2 * style.magnify;
    else
        hot_y = -((float)item->rows_in / 2 - (float)font->descent) * style.magnify;

    /* x position */
    if (align == TLEFT || align == MLEFT || align == BLEFT || align == ALIGNNONE)
        hot_x = -(float)item->max_width / 2 * style.magnify;
    else if (align == TCENTRE || align == MCENTRE || align == BCENTRE)
        hot_x = 0;
    else
        hot_x = (float)item->max_width / 2 * style.magnify;

    /* pre-calculate sin and cos */
    sin_angle = sin(angle);
    cos_angle = cos(angle);

    /* rotate hot_x and hot_y around bitmap centre */
    hot_xp = hot_x * cos_angle - hot_y * sin_angle;
    hot_yp = hot_x * sin_angle + hot_y * cos_angle;

    /* text background will be drawn using XFillPolygon */
    if (bg)
    {
        GC depth_one_gc;
        XPoint *xpoints;
        Pixmap empty_stipple;

        /* reserve space for XPoints */
        xpoints = (XPoint *)malloc((unsigned)(4 * item->nl * sizeof(XPoint)));
        if (!xpoints)
            return 1;

        /* rotate corner positions */
        for (i = 0; i < 4 * item->nl; i++)
        {
            xpoints[i].x = (short)(x + ((item->corners_x[i] - hot_x) * cos_angle + (item->corners_y[i] + hot_y) * sin_angle));
            xpoints[i].y = (short)(y + (-(item->corners_x[i] - hot_x) * sin_angle + (item->corners_y[i] + hot_y) * cos_angle));
        }

        /* we want to swap foreground and background colors here;
         XGetGCValues() is only available in R4+ */

        empty_stipple = XCreatePixmap(dpy, drawable, 1, 1, 1);

        depth_one_gc = XCreateGC(dpy, empty_stipple, 0, 0);
        XSetForeground(dpy, depth_one_gc, 0);
        XFillRectangle(dpy, empty_stipple, depth_one_gc, 0, 0, 2, 2);

        XSetStipple(dpy, my_gc, empty_stipple);
        XSetFillStyle(dpy, my_gc, FillOpaqueStippled);

        XFillPolygon(dpy, drawable, my_gc, xpoints, 4 * item->nl, Nonconvex,
                     CoordModeOrigin);

        /* free our resources */
        free((char *)xpoints);
        XFreeGC(dpy, depth_one_gc);
        XFreePixmap(dpy, empty_stipple);
    }

    /* where should top left corner of bitmap go ? */
    xp = (int)(x - ((float)item->cols_out / 2 + hot_xp));
    yp = (int)(y - ((float)item->rows_out / 2 - hot_yp));

    /* by default we draw the rotated bitmap, solid */
    bitmap_to_paint = item->bitmap;

/* handle user stippling */
#ifndef X11R3
    {
        GC depth_one_gc;
        XGCValues values;
        Pixmap new_bitmap, inverse;

        /* try and get some GC properties */
        if (XGetGCValues(dpy, gc,
                         GCStipple | GCFillStyle | GCForeground | GCBackground | GCTileStipXOrigin | GCTileStipYOrigin,
                         &values))
        {

            /* only do this if stippling requested */
            if ((values.fill_style == FillStippled || values.fill_style == FillOpaqueStippled) && !bg)
            {

                /* opaque stipple: draw rotated text in background colour */
                if (values.fill_style == FillOpaqueStippled)
                {
                    XSetForeground(dpy, my_gc, values.background);
                    XSetFillStyle(dpy, my_gc, FillStippled);
                    XSetStipple(dpy, my_gc, item->bitmap);
                    XSetTSOrigin(dpy, my_gc, xp, yp);
                    XFillRectangle(dpy, drawable, my_gc, xp, yp,
                                   item->cols_out, item->rows_out);
                    XSetForeground(dpy, my_gc, values.foreground);
                }

                /* this will merge the rotated text and the user's stipple */
                new_bitmap = XCreatePixmap(dpy, drawable,
                                           item->cols_out, item->rows_out, 1);

                /* create a GC */
                depth_one_gc = XCreateGC(dpy, new_bitmap, 0, 0);
                XSetForeground(dpy, depth_one_gc, 1);
                XSetBackground(dpy, depth_one_gc, 0);

                /* set the relative stipple origin */
                XSetTSOrigin(dpy, depth_one_gc,
                             values.ts_x_origin - xp, values.ts_y_origin - yp);

                /* fill the whole bitmap with the user's stipple */
                XSetStipple(dpy, depth_one_gc, values.stipple);
                XSetFillStyle(dpy, depth_one_gc, FillOpaqueStippled);
                XFillRectangle(dpy, new_bitmap, depth_one_gc,
                               0, 0, item->cols_out, item->rows_out);

                /* set stipple origin back to normal */
                XSetTSOrigin(dpy, depth_one_gc, 0, 0);

                /* this will contain an inverse copy of the rotated text */
                inverse = XCreatePixmap(dpy, drawable,
                                        item->cols_out, item->rows_out, 1);

                /* invert text */
                XSetFillStyle(dpy, depth_one_gc, FillSolid);
                XSetFunction(dpy, depth_one_gc, GXcopyInverted);
                XCopyArea(dpy, item->bitmap, inverse, depth_one_gc,
                          0, 0, item->cols_out, item->rows_out, 0, 0);

                /* now delete user's stipple everywhere EXCEPT on text */
                XSetForeground(dpy, depth_one_gc, 0);
                XSetBackground(dpy, depth_one_gc, 1);
                XSetStipple(dpy, depth_one_gc, inverse);
                XSetFillStyle(dpy, depth_one_gc, FillStippled);
                XSetFunction(dpy, depth_one_gc, GXcopy);
                XFillRectangle(dpy, new_bitmap, depth_one_gc,
                               0, 0, item->cols_out, item->rows_out);

                /* free resources */
                XFreePixmap(dpy, inverse);
                XFreeGC(dpy, depth_one_gc);

                /* this is the new bitmap */
                bitmap_to_paint = new_bitmap;
            }
        }
    }
#endif /*X11R3*/

    /* paint text using stipple technique */
    XSetFillStyle(dpy, my_gc, FillStippled);
    XSetStipple(dpy, my_gc, bitmap_to_paint);
    XSetTSOrigin(dpy, my_gc, xp, yp);
    XFillRectangle(dpy, drawable, my_gc, xp, yp,
                   item->cols_out, item->rows_out);

    /* free our resources */
    XFreeGC(dpy, my_gc);

    /* stippled bitmap no longer needed */
    if (bitmap_to_paint != item->bitmap)
        XFreePixmap(dpy, bitmap_to_paint);

#ifdef CACHE_XIMAGES
    XFreePixmap(dpy, item->bitmap);
#endif /*CACHE_XIMAGES*/

    /* if item isn't cached, destroy it completely */
    if (!item->cached)
        XRotFreeTextItem(dpy, item);

    /* we got to the end OK! */
    return 0;
}

/* ---------------------------------------------------------------------- */

/**************************************************************************/
/*  Draw a horizontal string in a quick fashion                           */
/**************************************************************************/

static int XRotDrawHorizontalString(Display *dpy, XFontStruct *font, Drawable drawable, GC gc, int x, int y, char *text, int align, int bg)
{
    GC my_gc;
    int nl = 1, i;
    int height;
    int xp, yp;
    char *str1, *str2, *str3;
    char *str2_a = (char *)"\0", *str2_b = (char *)"\n\0";
    int dir, asc, desc;
    XCharStruct overall;

    DEBUG_PRINT1("**\nHorizontal text.\n");

    /* this gc has similar properties to the user's gc (including stipple) */
    my_gc = XCreateGC(dpy, drawable, 0, 0);
    XCopyGC(dpy, gc,
            GCForeground | GCBackground | GCFunction | GCStipple | GCFillStyle | GCTileStipXOrigin | GCTileStipYOrigin | GCPlaneMask, my_gc);
    XSetFont(dpy, my_gc, font->fid);

    /* count number of sections in string */
    if (align != ALIGNNONE)
        for (i = 0; i < strlen(text) - 1; i++)
            if (text[i] == '\n')
                nl++;

    /* ignore newline characters if not doing alignment */
    if (align == ALIGNNONE)
        str2 = str2_a;
    else
        str2 = str2_b;

    /* overall font height */
    height = font->ascent + font->descent;

    /* y position */
    if (align == TLEFT || align == TCENTRE || align == TRIGHT)
        yp = y + font->ascent;
    else if (align == MLEFT || align == MCENTRE || align == MRIGHT)
        yp = y - nl * height / 2 + font->ascent;
    else if (align == BLEFT || align == BCENTRE || align == BRIGHT)
        yp = y - nl * height + font->ascent;
    else
        yp = y;

    str1 = my_strdup(text);
    if (str1 == NULL)
        return 1;

    str3 = my_strtok(str1, str2);

    /* loop through each section in the string */
    do
    {
        XTextExtents(font, str3, strlen(str3), &dir, &asc, &desc,
                     &overall);

        /* where to draw section in x ? */
        if (align == TLEFT || align == MLEFT || align == BLEFT || align == ALIGNNONE)
            xp = x;
        else if (align == TCENTRE || align == MCENTRE || align == BCENTRE)
            xp = x - overall.rbearing / 2;
        else
            xp = x - overall.rbearing;

        /* draw string onto bitmap */
        if (!bg)
            XDrawString(dpy, drawable, my_gc, xp, yp, str3, strlen(str3));
        else
            XDrawImageString(dpy, drawable, my_gc, xp, yp, str3, strlen(str3));

        /* move to next line */
        yp += height;

        str3 = my_strtok((char *)NULL, str2);
    } while (str3 != NULL);

    free(str1);
    XFreeGC(dpy, my_gc);

    return 0;
}

/* ---------------------------------------------------------------------- */

/**************************************************************************/
/*   Query cache for a match with this font/text/angle/alignment          */
/*       request, otherwise arrange for its creation                      */
/**************************************************************************/

static RotatedTextItem *XRotRetrieveFromCache(Display *dpy, XFontStruct *font, float angle, char *text, int align)
{
    Font fid;
    char *font_name = NULL;
    unsigned long name_value;
    RotatedTextItem *item = NULL;
    RotatedTextItem *i1 = first_text_item;

    /* get font name, if it exists */
    if (XGetFontProperty(font, XA_FONT, &name_value))
    {
        DEBUG_PRINT1("got font name OK\n");
        font_name = XGetAtomName(dpy, name_value);
        fid = 0;
    }
#ifdef CACHE_FID
    /* otherwise rely (unreliably?) on font ID */
    else
    {
        DEBUG_PRINT1("can't get fontname, caching FID\n");
        font_name = NULL;
        fid = font->fid;
    }
#else
    /* not allowed to cache font ID's */
    else
    {
        DEBUG_PRINT1("can't get fontname, can't cache\n");
        font_name = NULL;
        fid = 0;
    }
#endif /*CACHE_FID*/

    /* look for a match in cache */

    /* matching formula:
      identical text;
      identical fontname (if defined, font ID's if not);
      angles close enough (<0.00001 here, could be smaller);
      HORIZONTAL alignment matches, OR it's a one line string;
      magnifications the same */

    while (i1 && !item)
    {
        /* match everything EXCEPT fontname/ID */
        if (strcmp(text, i1->text) == 0 && fabs(angle - i1->angle) < 0.00001 && style.magnify == i1->magnify && (i1->nl == 1 || ((align == 0) ? 9 : (align - 1)) % 3 == ((i1->align == 0) ? 9 : (i1->align - 1)) % 3))
        {

            /* now match fontname/ID */
            if (font_name != NULL && i1->font_name != NULL)
            {
                if (strcmp(font_name, i1->font_name) == 0)
                {
                    item = i1;
                    DEBUG_PRINT1("Matched against font names\n");
                }
                else
                    i1 = i1->next;
            }
#ifdef CACHE_FID
            else if (font_name == NULL && i1->font_name == NULL)
            {
                if (fid == i1->fid)
                {
                    item = i1;
                    DEBUG_PRINT1("Matched against FID's\n");
                }
                else
                    i1 = i1->next;
            }
#endif /*CACHE_FID*/
            else
                i1 = i1->next;
        }
        else
            i1 = i1->next;
    }

    if (item)
        DEBUG_PRINT1("**\nFound target in cache.\n");
    if (!item)
        DEBUG_PRINT1("**\nNo match in cache.\n");

    /* no match */
    if (!item)
    {
        /* create new item */
        item = XRotCreateTextItem(dpy, font, angle, text, align);
        if (!item)
            return NULL;

        /* record what it shows */
        item->text = my_strdup(text);

        /* fontname or ID */
        if (font_name != NULL)
        {
            item->font_name = my_strdup(font_name);
            item->fid = 0;
        }
        else
        {
            item->font_name = NULL;
            item->fid = fid;
        }

        item->angle = angle;
        item->align = align;
        item->magnify = style.magnify;

        /* cache it */
        XRotAddToLinkedList(dpy, item);
    }

    if (font_name)
        XFree(font_name);

/* if XImage is cached, need to recreate the bitmap */

#ifdef CACHE_XIMAGES
    {
        GC depth_one_gc;

        /* create bitmap to hold rotated text */
        item->bitmap = XCreatePixmap(dpy, DefaultRootWindow(dpy),
                                     item->cols_out, item->rows_out, 1);

        /* depth one gc */
        depth_one_gc = XCreateGC(dpy, item->bitmap, NULL, 0);
        XSetBackground(dpy, depth_one_gc, 0);
        XSetForeground(dpy, depth_one_gc, 1);

        /* make the text bitmap from XImage */
        XPutImage(dpy, item->bitmap, depth_one_gc, item->ximage, 0, 0, 0, 0,
                  item->cols_out, item->rows_out);

        XFreeGC(dpy, depth_one_gc);
    }
#endif /*CACHE_XIMAGES*/

    return item;
}

/* ---------------------------------------------------------------------- */

/**************************************************************************/
/*  Create a rotated text item                                            */
/**************************************************************************/

static RotatedTextItem *XRotCreateTextItem(Display *dpy, XFontStruct *font, float angle, char *text, int align)
{
    RotatedTextItem *item = NULL;
    Pixmap canvas;
    GC font_gc;
    XImage *I_in;
    int i, j;
    char *str1, *str2, *str3;
    char *str2_a = (char *)"\0", *str2_b = (char *)"\n\0";
    int height;
    int byte_w_in, byte_w_out;
    int xp, yp;
    float sin_angle, cos_angle;
    int it, jt;
    float di, dj;
    int ic = 0;
    float xl, xr, xinc;
    int byte_out;
    int dir, asc, desc;
    XCharStruct overall;
    int old_cols_in = 0, old_rows_in = 0;

    /* allocate memory */
    item = (RotatedTextItem *)malloc((unsigned)sizeof(RotatedTextItem));
    if (!item)
        return NULL;

    /* count number of sections in string */
    item->nl = 1;
    if (align != ALIGNNONE)
        for (i = 0; i < strlen(text) - 1; i++)
            if (text[i] == '\n')
                item->nl++;

    /* ignore newline characters if not doing alignment */
    if (align == ALIGNNONE)
        str2 = str2_a;
    else
        str2 = str2_b;

    /* find width of longest section */
    str1 = my_strdup(text);
    if (str1 == NULL)
        return NULL;

    str3 = my_strtok(str1, str2);

    XTextExtents(font, str3, strlen(str3), &dir, &asc, &desc,
                 &overall);

    item->max_width = overall.rbearing;

    /* loop through each section */
    do
    {
        str3 = my_strtok((char *)NULL, str2);

        if (str3 != NULL)
        {
            XTextExtents(font, str3, strlen(str3), &dir, &asc, &desc,
                         &overall);

            if (overall.rbearing > item->max_width)
                item->max_width = overall.rbearing;
        }
    } while (str3 != NULL);

    free(str1);

    /* overall font height */
    height = font->ascent + font->descent;

    /* dimensions horizontal text will have */
    item->cols_in = item->max_width;
    item->rows_in = item->nl * height;

    /* bitmap for drawing on */
    canvas = XCreatePixmap(dpy, DefaultRootWindow(dpy),
                           item->cols_in, item->rows_in, 1);

    /* create a GC for the bitmap */
    font_gc = XCreateGC(dpy, canvas, 0, 0);
    XSetBackground(dpy, font_gc, 0);
    XSetFont(dpy, font_gc, font->fid);

    /* make sure the bitmap is blank */
    XSetForeground(dpy, font_gc, 0);
    XFillRectangle(dpy, canvas, font_gc, 0, 0,
                   item->cols_in + 1, item->rows_in + 1);
    XSetForeground(dpy, font_gc, 1);

    /* pre-calculate sin and cos */
    sin_angle = sin(angle);
    cos_angle = cos(angle);

    /* text background will be drawn using XFillPolygon */
    item->corners_x = (float *)malloc((unsigned)(4 * item->nl * sizeof(float)));
    if (!item->corners_x)
        return NULL;

    item->corners_y = (float *)malloc((unsigned)(4 * item->nl * sizeof(float)));
    if (!item->corners_y)
        return NULL;

    /* draw text horizontally */

    /* start at top of bitmap */
    yp = font->ascent;

    str1 = my_strdup(text);
    if (str1 == NULL)
        return NULL;

    str3 = my_strtok(str1, str2);

    /* loop through each section in the string */
    do
    {
        XTextExtents(font, str3, strlen(str3), &dir, &asc, &desc,
                     &overall);

        /* where to draw section in x ? */
        if (align == TLEFT || align == MLEFT || align == BLEFT || align == ALIGNNONE)
            xp = 0;
        else if (align == TCENTRE || align == MCENTRE || align == BCENTRE)
            xp = (item->max_width - overall.rbearing) / 2;
        else
            xp = item->max_width - overall.rbearing;

        /* draw string onto bitmap */
        XDrawString(dpy, canvas, font_gc, xp, yp, str3, strlen(str3));

        /* keep a note of corner positions of this string */
        item->corners_x[ic] = ((float)xp - (float)item->cols_in / 2) * style.magnify;
        item->corners_y[ic] = ((float)(yp - font->ascent) - (float)item->rows_in / 2)
                              * style.magnify;
        item->corners_x[ic + 1] = item->corners_x[ic];
        item->corners_y[ic + 1] = item->corners_y[ic] + (float)height * style.magnify;
        item->corners_x[item->nl * 4 - 1 - ic] = item->corners_x[ic] + (float)overall.rbearing * style.magnify;
        item->corners_y[item->nl * 4 - 1 - ic] = item->corners_y[ic];
        item->corners_x[item->nl * 4 - 2 - ic] = item->corners_x[item->nl * 4 - 1 - ic];
        item->corners_y[item->nl * 4 - 2 - ic] = item->corners_y[ic + 1];

        ic += 2;

        /* move to next line */
        yp += height;

        str3 = my_strtok((char *)NULL, str2);
    } while (str3 != NULL);

    free(str1);

    /* create image to hold horizontal text */
    I_in = MakeXImage(dpy, item->cols_in, item->rows_in);
    if (I_in == NULL)
        return NULL;

    /* extract horizontal text */
    XGetSubImage(dpy, canvas, 0, 0, item->cols_in, item->rows_in,
                 1, XYPixmap, I_in, 0, 0);
    I_in->format = XYBitmap;

    /* magnify horizontal text */
    if (style.magnify != 1.)
    {
        I_in = XRotMagnifyImage(dpy, I_in);

        old_cols_in = item->cols_in;
        old_rows_in = item->rows_in;
        item->cols_in = (int)(item->cols_in * style.magnify);
        item->rows_in = (int)(item->rows_in * style.magnify);
    }

    /* how big will rotated text be ? */
    item->cols_out = (int)(fabs((float)item->rows_in * sin_angle) + fabs((float)item->cols_in * cos_angle) + 0.99999 + 2);

    item->rows_out = (int)(fabs((float)item->rows_in * cos_angle) + fabs((float)item->cols_in * sin_angle) + 0.99999 + 2);

    if (item->cols_out % 2 == 0)
        item->cols_out++;

    if (item->rows_out % 2 == 0)
        item->rows_out++;

    /* create image to hold rotated text */
    item->ximage = MakeXImage(dpy, item->cols_out, item->rows_out);
    if (item->ximage == NULL)
        return NULL;

    byte_w_in = (item->cols_in - 1) / 8 + 1;
    byte_w_out = (item->cols_out - 1) / 8 + 1;

    /* we try to make this bit as fast as possible - which is why it looks
      a bit over-the-top */

    /* vertical distance from centre */
    dj = 0.5 - (float)item->rows_out / 2;
    /* where abouts does text actually lie in rotated image? */
    if (angle == 0 || angle == M_PIF / 2 || angle == M_PIF || angle == 3 * M_PIF / 2)
    {
        xl = 0;
        xr = (float)item->cols_out;
        xinc = 0;
    }
    else if (angle < M_PIF)
    {
        xl = (float)item->cols_out / 2 + (dj - (float)item->rows_in / (2 * cos_angle)) / tan(angle) - 2;
        xr = (float)item->cols_out / 2 + (dj + (float)item->rows_in / (2 * cos_angle)) / tan(angle) + 2;
        xinc = 1. / tan(angle);
    }
    else
    {
        xl = (float)item->cols_out / 2 + (dj + (float)item->rows_in / (2 * cos_angle)) / tan(angle) - 2;
        xr = (float)item->cols_out / 2 + (dj - (float)item->rows_in / (2 * cos_angle)) / tan(angle) + 2;

        xinc = 1. / tan(angle);
    }

    /* loop through all relevent bits in rotated image */
    for (j = 0; j < item->rows_out; j++)
    {

        /* no point re-calculating these every pass */
        di = (float)((xl < 0) ? 0 : (int)xl) + 0.5 - (float)item->cols_out / 2;
        byte_out = (item->rows_out - j - 1) * byte_w_out;

        /* loop through meaningful columns */
        for (i = ((xl < 0) ? 0 : (int)xl);
             i < ((xr >= item->cols_out) ? item->cols_out : (int)xr); i++)
        {

            /* rotate coordinates */
            it = (int)((float)item->cols_in / 2 + (di * cos_angle + dj * sin_angle));
            jt = (int)((float)item->rows_in / 2 - (-di * sin_angle + dj * cos_angle));

            /* set pixel if required */
            if (it >= 0 && it < item->cols_in && jt >= 0 && jt < item->rows_in)
                if ((I_in->data[jt * byte_w_in + it / 8] & 128 >> (it % 8)) > 0)
                    item->ximage->data[byte_out + i / 8] |= 128 >> i % 8;

            di += 1;
        }
        dj += 1;
        xl += xinc;
        xr += xinc;
    }
    XDestroyImage(I_in);

    if (style.magnify != 1.)
    {
        item->cols_in = old_cols_in;
        item->rows_in = old_rows_in;
    }

#ifdef CACHE_BITMAPS

    /* create a bitmap to hold rotated text */
    item->bitmap = XCreatePixmap(dpy, DefaultRootWindow(dpy),
                                 item->cols_out, item->rows_out, 1);

    /* make the text bitmap from XImage */
    XPutImage(dpy, item->bitmap, font_gc, item->ximage, 0, 0, 0, 0,
              item->cols_out, item->rows_out);

    XDestroyImage(item->ximage);
#endif /*CACHE_BITMAPS*/

    XFreeGC(dpy, font_gc);
    XFreePixmap(dpy, canvas);

    return item;
}

/* ---------------------------------------------------------------------- */

/**************************************************************************/
/*  Adds a text item to the end of the cache, removing as many items      */
/*      from the front as required to keep cache size below limit         */
/**************************************************************************/

static void XRotAddToLinkedList(Display *dpy, RotatedTextItem *item)
{

    static long int current_size = 0;
    static RotatedTextItem *last = NULL;
    RotatedTextItem *i1 = first_text_item, *i2 = NULL;

#ifdef CACHE_BITMAPS

    /* I don't know how much memory a pixmap takes in the server -
          probably this + a bit more we can't account for */

    item->size = ((item->cols_out - 1) / 8 + 1) * item->rows_out;

#else

    /* this is pretty much the size of a RotatedTextItem */

    item->size = ((item->cols_out - 1) / 8 + 1) * item->rows_out + sizeof(XImage) + strlen(item->text) + item->nl * 8 * sizeof(float) + sizeof(RotatedTextItem);

    if (item->font_name != NULL)
        item->size += strlen(item->font_name);
    else
        item->size += sizeof(Font);
#endif /*CACHE_BITMAPS */

#ifdef DEBUG
    /* count number of items in cache, for debugging */
    {
        int i = 0;

        while (i1)
        {
            i++;
            i1 = i1->next;
        }
        DEBUG_PRINT2("Cache has %d items.\n", i);
        i1 = first_text_item;
    }
#endif

    DEBUG_PRINT4("current cache size=%ld, new item=%ld, limit=%ld\n",
                 current_size, item->size, CACHE_SIZE_LIMIT * 1024L);

    /* if this item is bigger than whole cache, forget it */
    if (item->size > CACHE_SIZE_LIMIT * 1024)
    {
        DEBUG_PRINT1("Too big to cache\n\n");
        item->cached = 0;
        return;
    }

    /* remove elements from cache as needed */
    while (i1 && current_size + item->size > CACHE_SIZE_LIMIT * 1024)
    {

        DEBUG_PRINT2("Removed %d bytes\n", (int)i1->size);

        if (i1->font_name != NULL)
            DEBUG_PRINT5("  (`%s'\n   %s\n   angle=%f align=%d)\n",
                         i1->text, i1->font_name, i1->angle, i1->align);

#ifdef CACHE_FID
        if (i1->font_name == NULL)
            DEBUG_PRINT5("  (`%s'\n  FID=%ld\n   angle=%f align=%d)\n",
                         i1->text, i1->fid, i1->angle, i1->align);
#endif /*CACHE_FID*/

        current_size -= i1->size;

        i2 = i1->next;

        /* free resources used by the unlucky item */
        XRotFreeTextItem(dpy, i1);

        /* remove it from linked list */
        first_text_item = i2;
        i1 = i2;
    }

    /* add new item to end of linked list */
    if (first_text_item == NULL)
    {
        item->next = NULL;
        first_text_item = item;
        last = item;
    }
    else
    {
        item->next = NULL;
        last->next = item;
        last = item;
    }

    /* new cache size */
    current_size += item->size;

    item->cached = 1;

    DEBUG_PRINT1("Added item to cache.\n");
}

/* ---------------------------------------------------------------------- */

/**************************************************************************/
/*  Free the resources used by a text item                                */
/**************************************************************************/

static void XRotFreeTextItem(Display *dpy, RotatedTextItem *item)
{
    free(item->text);

    if (item->font_name != NULL)
        free(item->font_name);

    free((char *)item->corners_x);
    free((char *)item->corners_y);

#ifdef CACHE_BITMAPS
    XFreePixmap(dpy, item->bitmap);
#else
    XDestroyImage(item->ximage);
#endif /* CACHE_BITMAPS */

    free((char *)item);
}

/* ---------------------------------------------------------------------- */

/**************************************************************************/
/* Magnify an XImage using bilinear interpolation                         */
/**************************************************************************/

static XImage *XRotMagnifyImage(Display *dpy, XImage *ximage)
{
    int i, j;
    float x, y;
    float u, t;
    XImage *I_out;
    int cols_in, rows_in;
    int cols_out, rows_out;
    int i2, j2;
    float z1, z2, z3, z4;
    int byte_width_in, byte_width_out;
    float mag_inv;

    /* size of input image */
    cols_in = ximage->width;
    rows_in = ximage->height;

    /* size of final image */
    cols_out = (int)(cols_in * style.magnify);
    rows_out = (int)(rows_in * style.magnify);

    /* this will hold final image */
    I_out = MakeXImage(dpy, cols_out, rows_out);
    if (I_out == NULL)
        return NULL;

    /* width in bytes of input, output images */
    byte_width_in = (cols_in - 1) / 8 + 1;
    byte_width_out = (cols_out - 1) / 8 + 1;

    /* for speed */
    mag_inv = 1. / style.magnify;

    y = 0.;

    /* loop over magnified image */
    for (j2 = 0; j2 < rows_out; j2++)
    {
        x = 0;
        j = (int)y;

        for (i2 = 0; i2 < cols_out; i2++)
        {
            i = (int)x;

            /* bilinear interpolation - where are we on bitmap ? */
            /* right edge */
            if (i == cols_in - 1 && j != rows_in - 1)
            {
                t = 0;
                u = y - (float)j;

                z1 = (ximage->data[j * byte_width_in + i / 8] & 128 >> (i % 8)) > 0;
                z2 = z1;
                z3 = (ximage->data[(j + 1) * byte_width_in + i / 8] & 128 >> (i % 8)) > 0;
                z4 = z3;
            }
            /* top edge */
            else if (i != cols_in - 1 && j == rows_in - 1)
            {
                t = x - (float)i;
                u = 0;

                z1 = (ximage->data[j * byte_width_in + i / 8] & 128 >> (i % 8)) > 0;
                z2 = (ximage->data[j * byte_width_in + (i + 1) / 8] & 128 >> ((i + 1) % 8)) > 0;
                z3 = z2;
                z4 = z1;
            }
            /* top right corner */
            else if (i == cols_in - 1 && j == rows_in - 1)
            {
                u = 0;
                t = 0;

                z1 = (ximage->data[j * byte_width_in + i / 8] & 128 >> (i % 8)) > 0;
                z2 = z1;
                z3 = z1;
                z4 = z1;
            }
            /* somewhere `safe' */
            else
            {
                t = x - (float)i;
                u = y - (float)j;

                z1 = (ximage->data[j * byte_width_in + i / 8] & 128 >> (i % 8)) > 0;
                z2 = (ximage->data[j * byte_width_in + (i + 1) / 8] & 128 >> ((i + 1) % 8)) > 0;
                z3 = (ximage->data[(j + 1) * byte_width_in + (i + 1) / 8] & 128 >> ((i + 1) % 8)) > 0;
                z4 = (ximage->data[(j + 1) * byte_width_in + i / 8] & 128 >> (i % 8)) > 0;
            }

            /* if interpolated value is greater than 0.5, set bit */
            if (((1 - t) * (1 - u) * z1 + t * (1 - u) * z2 + t * u * z3 + (1 - t) * u * z4) > 0.5)
                I_out->data[j2 * byte_width_out + i2 / 8] |= 128 >> i2 % 8;

            x += mag_inv;
        }
        y += mag_inv;
    }

    /* destroy original */
    XDestroyImage(ximage);

    /* return big image */
    return I_out;
}

/* ---------------------------------------------------------------------- */

/**************************************************************************/
/* Calculate the bounding box some text will have when painted            */
/**************************************************************************/

XPoint *XRotTextExtents(Display *, XFontStruct *font, float angle, int x, int y, char *text, int align)
{
    int i;
    char *str1, *str2, *str3;
    char *str2_a = (char *)"\0", *str2_b = (char *)"\n\0";
    int height;
    float sin_angle, cos_angle;
    int nl, max_width;
    int cols_in, rows_in;
    float hot_x, hot_y;
    XPoint *xp_in, *xp_out;
    int dir, asc, desc;
    XCharStruct overall;

    /* manipulate angle to 0<=angle<360 degrees */
    while (angle < 0)
        angle += 360;

    while (angle > 360)
        angle -= 360;

    angle *= M_PIF / 180;

    /* count number of sections in string */
    nl = 1;
    if (align != ALIGNNONE)
        for (i = 0; i < strlen(text) - 1; i++)
            if (text[i] == '\n')
                nl++;

    /* ignore newline characters if not doing alignment */
    if (align == ALIGNNONE)
        str2 = str2_a;
    else
        str2 = str2_b;

    /* find width of longest section */
    str1 = my_strdup(text);
    if (str1 == NULL)
        return NULL;

    str3 = my_strtok(str1, str2);

    XTextExtents(font, str3, strlen(str3), &dir, &asc, &desc,
                 &overall);

    max_width = overall.rbearing;

    /* loop through each section */
    do
    {
        str3 = my_strtok((char *)NULL, str2);

        if (str3 != NULL)
        {
            XTextExtents(font, str3, strlen(str3), &dir, &asc, &desc,
                         &overall);

            if (overall.rbearing > max_width)
                max_width = overall.rbearing;
        }
    } while (str3 != NULL);

    free(str1);

    /* overall font height */
    height = font->ascent + font->descent;

    /* dimensions horizontal text will have */
    cols_in = max_width;
    rows_in = nl * height;

    /* pre-calculate sin and cos */
    sin_angle = sin(angle);
    cos_angle = cos(angle);

    /* y position */
    if (align == TLEFT || align == TCENTRE || align == TRIGHT)
        hot_y = (float)rows_in / 2 * style.magnify;
    else if (align == MLEFT || align == MCENTRE || align == MRIGHT)
        hot_y = 0;
    else if (align == BLEFT || align == BCENTRE || align == BRIGHT)
        hot_y = -(float)rows_in / 2 * style.magnify;
    else
        hot_y = -((float)rows_in / 2 - (float)font->descent) * style.magnify;

    /* x position */
    if (align == TLEFT || align == MLEFT || align == BLEFT || align == ALIGNNONE)
        hot_x = -(float)max_width / 2 * style.magnify;
    else if (align == TCENTRE || align == MCENTRE || align == BCENTRE)
        hot_x = 0;
    else
        hot_x = (float)max_width / 2 * style.magnify;

    /* reserve space for XPoints */
    xp_in = (XPoint *)malloc((unsigned)(5 * sizeof(XPoint)));
    if (!xp_in)
        return NULL;

    xp_out = (XPoint *)malloc((unsigned)(5 * sizeof(XPoint)));
    if (!xp_out)
        return NULL;

    /* bounding box when horizontal, relative to bitmap centre */
    xp_in[0].x = (short)(-(float)cols_in * style.magnify / 2 - style.bbx_pad);
    xp_in[0].y = (short)((float)rows_in * style.magnify / 2 + style.bbx_pad);
    xp_in[1].x = (short)((float)cols_in * style.magnify / 2 + style.bbx_pad);
    xp_in[1].y = (short)((float)rows_in * style.magnify / 2 + style.bbx_pad);
    xp_in[2].x = (short)((float)cols_in * style.magnify / 2 + style.bbx_pad);
    xp_in[2].y = (short)(-(float)rows_in * style.magnify / 2 - style.bbx_pad);
    xp_in[3].x = (short)(-(float)cols_in * style.magnify / 2 - style.bbx_pad);
    xp_in[3].y = (short)(-(float)rows_in * style.magnify / 2 - style.bbx_pad);
    xp_in[4].x = xp_in[0].x;
    xp_in[4].y = xp_in[0].y;

    /* rotate and translate bounding box */
    for (i = 0; i < 5; i++)
    {
        xp_out[i].x = (short)((float)x + (((float)xp_in[i].x - hot_x) * cos_angle + ((float)xp_in[i].y + hot_y) * sin_angle));
        xp_out[i].y = (short)((float)y + (-((float)xp_in[i].x - hot_x) * sin_angle + ((float)xp_in[i].y + hot_y) * cos_angle));
    }

    free((char *)xp_in);

    return xp_out;
}
