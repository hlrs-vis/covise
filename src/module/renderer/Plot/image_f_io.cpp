/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * This file: image_f_io.c (part of the WSCRAWL program)
 *
 * This file contains the "Image File I/O" package for wscrawl (or anything
 * for that matter.)  The format used is the standard X-Window Dump form
 * that the MIT client "xwd" uses.  This File I/O was made possible by the
 * help and extensive source code of Mark Cook of Hewlett-Packard, who I
 * bothered endlessly to get this working.
 *
 * I tried to make this file of routines as self-contained and portable as
 * possible.  Please feel free to use this file as is, or with modifications,
 * for any and all applications.  -- Brian Wilson
 *
 * This file was last modified: 10/7/91 (initial port to wscrawl 2.0)
 */

#define BAD_CHOICE -1
#define NO_DUPLICATE -1
#define UNALLOCATED -1

#include <X11/Xos.h>
#include <X11/XWDFile.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" {
extern int _XInitImageFuncPtrs(XImage *);
}

static int Image_Size(XImage *image, int format);
static int Get_Colors(XColor **colors, Display *disp, Colormap the_colormap);
static void _swapshort(char *bp, unsigned n);
static void _swaplong(char *bp, unsigned n);
static int allocate_colors_and_assign_em(Display *disp, Window win_id, XImage **image_ptr, XColor *cells);

/*
 * save_image_on_disk - this routine saves the region specified by the various
 *               parameters out to disk in the "defacto standard" (xwd) format.
 *               This has now been modified to take two window id's.  The
 *               first is to gather "window" information such as visual
 *               type, etc, and the second is where the image comes from.
 *               This is because in wscrawl 2.0, the image comes from a
 *               Pixmap, so pixmaps are VALID for the second win_id, but
 *               not for the first.  In the simple case of getting an
 *               image from a window, just pass the window id in twice.
 *
 *               "the_colormap" can be NULL if you wish to use the default
 *               colormap.  Most times this will be the case.
 */
int save_image_on_disk(Display *disp,
                       Window info_win_id,
                       Pixmap win_id,
                       int x,
                       int y,
                       int width,
                       int height,
                       char *name_of_file,
                       Colormap the_colormap)
{
    unsigned long swaptest = true;
    XRectangle boxx2;
    XRectangle *boxx;
    FILE *out_file_ptr;
    XColor *colors;
    unsigned buffer_size;
    int win_name_size;
    int header_size;
    int format = ZPixmap;
    int ncolors, i;
    XWindowAttributes win_info;
    XImage *ImagePix;
    XWDFileHeader header;

    /*
    * convert to form this code understands (I got code from Mark Cook)
    */
    boxx = &boxx2;
    boxx->x = x;
    boxx->y = y;
    boxx->width = width;
    boxx->height = height;

    /*
    * Open the file in which the image is to be stored
    */
    if ((out_file_ptr = fopen(name_of_file, "w")) == NULL)
    {
        printf("ERROR: Could not open file %s.\n", name_of_file);
        return (0);
    } /* Dump the image to the specified file */
    else
    {
        if (the_colormap == (Colormap)NULL)
        {
            the_colormap = XDefaultColormapOfScreen(XDefaultScreenOfDisplay(disp));
        }
        if (!XGetWindowAttributes(disp, info_win_id, &win_info))
        {
            printf("Can't get window attributes.\n");
            return (0);
        }
        /*
       * sizeof(char) is included for the null string terminator.
       */
        win_name_size = strlen(name_of_file) + sizeof(char);

        ImagePix = XGetImage(disp, win_id, boxx->x, boxx->y, boxx->width,
                             boxx->height, AllPlanes, format);
        XFlush(disp);

        if (ImagePix == NULL)
        {
            printf("GetImage failed.\n");
            return (0);
        }
        buffer_size = Image_Size(ImagePix, format);
        /* determines size of
       * pixmap */

        /*
       * Get the RGB values for the current color cells
       */
        if ((ncolors = Get_Colors(&colors, disp, the_colormap)) == 0)
        {
            printf("Cannot alloc memory for color structs.\n");
            return (0);
        }
        XFlush(disp);

        header_size = sizeof(header) + win_name_size;
        /* Calculates header
       * size */

        /*
       * Assemble the file header information
       */
        header.header_size = (xwdval)header_size;
        header.file_version = (xwdval)XWD_FILE_VERSION;
        header.pixmap_format = (xwdval)format;
        header.pixmap_depth = (xwdval)ImagePix->depth;

        header.pixmap_width = (xwdval)ImagePix->width;
        header.pixmap_height = (xwdval)ImagePix->height;
        header.xoffset = (xwdval)ImagePix->xoffset;
        header.byte_order = (xwdval)ImagePix->byte_order;
        header.bitmap_unit = (xwdval)ImagePix->bitmap_unit;
        header.bitmap_bit_order = (xwdval)ImagePix->bitmap_bit_order;
        header.bitmap_pad = (xwdval)ImagePix->bitmap_pad;
        header.bits_per_pixel = (xwdval)ImagePix->bits_per_pixel;
        header.bytes_per_line = (xwdval)ImagePix->bytes_per_line;
        header.visual_class = (xwdval)win_info.visual->c_class;
        header.red_mask = (xwdval)win_info.visual->red_mask;
        header.green_mask = (xwdval)win_info.visual->green_mask;
        header.blue_mask = (xwdval)win_info.visual->blue_mask;
        header.bits_per_rgb = (xwdval)win_info.visual->bits_per_rgb;
        header.colormap_entries = (xwdval)win_info.visual->map_entries;
        header.ncolors = ncolors;
        header.window_width = (xwdval)ImagePix->width;
        header.window_height = (xwdval)ImagePix->height;
        header.window_x = (xwdval)0;
        header.window_y = (xwdval)0;
        header.window_bdrwidth = (xwdval)0;

        if (*(char *)&swaptest)
        {
            _swaplong((char *)&header, sizeof(header));
            for (i = 0; i < ncolors; i++)
            {
                _swaplong((char *)&colors[i].pixel, sizeof(long));
                _swapshort((char *)&colors[i].red, 3 * sizeof(short));
            }
        }
        /*
       * Write out the file header information
       */
        (void)fwrite((char *)&header, sizeof(header), 1, out_file_ptr);
        (void)fwrite(name_of_file, win_name_size, 1, out_file_ptr);

        /*
       * Write out the color cell RGB values
       */
        (void)fwrite((char *)colors, sizeof(XColor), ncolors, out_file_ptr);

        /*
       * Write out the buffer
       */
        (void)fwrite(ImagePix->data, (int)buffer_size, 1, out_file_ptr);

        if (ncolors > 0)
            free(colors); /* free the color buffer */

        fclose(out_file_ptr);
        XFlush(disp);
        XDestroyImage(ImagePix);
    }
    return 0;
}

/*
 * Image_Size - this routine takes an XImage and returns it's total byte count
 */
static int Image_Size(XImage *image, int format)
{
    if (format != ZPixmap)
        return (image->bytes_per_line * image->height * image->depth);
    else
        return (image->bytes_per_line * image->height);
}

/*
 * Get_Colors - takes a pointer to an XColor struct and returns the total
 *            number of cells in the current colormap, plus all of their
 *            RGB values.
 */
static int Get_Colors(XColor **colors, Display *disp, Colormap the_colormap)
{
    int i, ncolors;

    ncolors = DisplayCells(disp, DefaultScreen(disp));

    if ((*colors = (XColor *)malloc(sizeof(XColor) * ncolors)) == NULL)
        return (false);

    for (i = 0; i < ncolors; i++)
        (*colors)[i].pixel = i;

    XQueryColors(disp, the_colormap, *colors, ncolors);
    return (ncolors);
}

/*
 * _swapshort - this routine is stolen, and I don't know what it does
 */
static void _swapshort(char *bp, unsigned n)
{
    char c;
    /* char *ep = bp + n; */
    char *ep;

    ep = bp + n;
    while (bp < ep)
    {
        c = *bp;
        *bp = *(bp + 1);
        bp++;
        *bp++ = c;
    }
}

/*
 * _swaplong - this routine is stolen, and I don't know what it does
 */
static void _swaplong(char *bp, unsigned n)
{
    char c;
    /* char *ep = bp + n; */
    char *sp;
    char *ep;

    ep = bp + n;
    while (bp < ep)
    {
        sp = bp + 3;
        c = *sp;
        *sp = *bp;
        *bp++ = c;
        sp = bp + 1;
        c = *sp;
        *sp = *bp;
        *bp++ = c;
        bp += 2;
    }
}

/*
 * read_image_from_disk - this routine reads the file indicated and allocates
 *               and loads up and then finally returns the XImage structure
 *               ready to blow out to the indicated display.  If at all
 *               possible, it attempts to return an image with the
 *               depth equalling the depth of the disp passed in.  Either
 *               way, it will return the depth it managed to get.
 */
XImage *read_image_from_disk(Display *disp,
                             Window win_id,
                             char *name_of_file,
                             int *width,
                             int *height,
                             int *depth)
{
    XImage *ImagePix;
    unsigned long swaptest = true;
    unsigned buffer_size;
    char *buffer;
    char *win_name;
    int format;
    int i, name_size, ncolors;
    XColor *colors = NULL;
    FILE *in_file;
    XWDFileHeader header;

    /* open file for read */
    if ((in_file = fopen(name_of_file, "r")) == NULL)
    {
        printf("ERROR: could not open file %s.\n", name_of_file);
        return (0);
    }
    /* read header */
    if (fread((char *)&header, sizeof(header), 1, in_file) != 1)
    {
        printf("ERROR: unable to read imagefile header.\n");
        return (0);
    }
    if (*(char *)&swaptest)
        _swaplong((char *)&header, sizeof(header));

    /*
    * check to see if the dump file is in the proper format
    */
    if (header.file_version != XWD_FILE_VERSION)
    {
        printf("ERROR: Imagefile format version mismatch.\n");
        return (0);
    }
    if (header.header_size < sizeof(header))
    {
        printf("ERROR: Imagefile header is too small.\n");
        return (0);
    }
    /* space for window name */
    name_size = (int)(header.header_size - sizeof(header));

    if ((win_name = (char *)malloc((unsigned)name_size * sizeof(char))) == NULL)
    {
        printf("ERROR: Can't malloc window name storage.\n");
        return (0);
    }
    /*
    * Read in window name
    */
    if (fread(win_name, sizeof(char), name_size, in_file) != name_size)
    {
        printf("ERROR: Unable to read window name from file.\n");
        return (0);
    }
    /*
    * Malloc the image data space and initialize it
    */
    if ((ImagePix = (XImage *)malloc(sizeof(XImage))) == NULL)
    {
        printf("ERROR: Can't malloc space for the image.\n");
        return (0);
    }
    ImagePix->width = (int)header.pixmap_width;
    ImagePix->height = (int)header.pixmap_height;
    ImagePix->xoffset = (int)header.xoffset;
    ImagePix->format = (int)header.pixmap_format;
    ImagePix->byte_order = (int)header.byte_order;
    ImagePix->bitmap_unit = (int)header.bitmap_unit;
    ImagePix->bitmap_bit_order = (int)header.bitmap_bit_order;
    ImagePix->bitmap_pad = (int)header.bitmap_pad;
    ImagePix->depth = (int)header.pixmap_depth;
    ImagePix->bits_per_pixel = (int)header.bits_per_pixel;
    ImagePix->bytes_per_line = (int)header.bytes_per_line;
    ImagePix->red_mask = header.red_mask;
    ImagePix->green_mask = header.green_mask;
    ImagePix->blue_mask = header.blue_mask;
    ImagePix->obdata = NULL;
    _XInitImageFuncPtrs(ImagePix);

    format = ImagePix->format;

    /*
    * malloc memory for the RGB values from the old colormap and read in the
    * values
    */
    ncolors = (int)header.ncolors;
    if (ncolors)
    {
        if ((colors = (XColor *)malloc(ncolors * sizeof(XColor))) == NULL)
        {
            printf("ERROR: Can't malloc space for the image cmap.\n");
            if (win_name)
                free(win_name);
            if (ImagePix)
                XDestroyImage(ImagePix);
            return (0);
        }
        if (fread((char *)colors, sizeof(XColor), ncolors, in_file) != ncolors)
        {
            printf("ERROR: Unable to read cmap from imagefile.\n");
            if (win_name)
                free(win_name);
            if (ImagePix)
                XDestroyImage(ImagePix);
            if (colors)
                free((char *)colors);
            return (0);
        }
        if (*(char *)&swaptest)
        {
            for (i = 0; i < ncolors; i++)
            {
                _swaplong((char *)&colors[i].pixel, sizeof(long));
                _swapshort((char *)&colors[i].red, 3 * sizeof(short));
            }
        }
    }
    buffer_size = Image_Size(ImagePix, format); /* malloc the pixel buffer */
    if ((buffer = (char *)malloc(buffer_size * sizeof(char))) == NULL)
    {
        printf("ERROR: Can't malloc the data buffer.\n");
        if (win_name)
            free(win_name);
        if (ImagePix)
            XDestroyImage(ImagePix);
        if (colors)
            free((char *)colors);
        return (0);
    }
    ImagePix->data = buffer;

    /*
    * Read in the pixmap buffer
    */
    if (fread(buffer, sizeof(char), (int)buffer_size, in_file) != buffer_size)
    {
        printf("ERROR: Unable to read pixmap from imagefile.\n");
        if (win_name)
            free(win_name);
        if (ImagePix)
            XDestroyImage(ImagePix);
        if (colors)
            free((char *)colors);
        if (buffer)
            free((char *)buffer);
        return (0);
    }
    (void)fclose(in_file); /* we are done with the infile */

    *depth = ImagePix->depth;
    /* even if the rest fails, return the
    * dimensions */
    *width = ImagePix->width;
    *height = ImagePix->height;

    if (win_name && (name_size > 0))
        free(win_name);

    if (allocate_colors_and_assign_em(disp, win_id, &ImagePix, colors) != true)
    {
        /*
       * the above converts the pixels in the xwd file over to the current
       * colormap, and also swaps the image to a new depth if the depth the
       * image was stored on disk as disagrees with the depth of the window
       * it is to be blasted into.  This block is if that fails.
       */
        printf("ERROR: Can't convert xwd file to usuable format.\n");
        printf("       Probably because the display is a wimpy non-HP.\n");
        if (ImagePix)
            XDestroyImage(ImagePix);
        if (colors)
            free((char *)colors);
        return (0);
    }
    else if (colors)
        free((char *)colors);

    *depth = ImagePix->depth; /* the depth may have changed */
    return (ImagePix);
}

/*
 * allocate_colors_and_assign_em - this function takes an XImage and it's
 *            colormap as it was at the time the image was created, and
 *            allocates cells in the system colormap matching those old colors
 *            and shuffles the pixels in the image to point to our new color
 *            cells instead of those old color cells.
 */
static int allocate_colors_and_assign_em(Display *disp, Window win_id, XImage **image_ptr, XColor *cells)
{
    int width, height, i, j, num_cells_in_colormap;
    unsigned long *opv, *npv; /* old and new pixel values */
    unsigned long tmp_pixel_value;
    XColor screen_in_out;
    XImage *diff_depth_im, *image;
    XWindowAttributes win_attr;
    int buffer_size;
    char *buffer;

    image = *image_ptr; /* for my sanity */
    height = image->height;
    width = image->width;

    /*
    * determine the number of cells in the colormap by taking 2 to the power
    * of the number of bits of depth this display has.
    */
    for (i = 0, num_cells_in_colormap = 1; i < (*image_ptr)->depth; i++)
        num_cells_in_colormap *= 2;

    if (num_cells_in_colormap > 256)
    {
        printf("WARNING: This is a monster deep display image, dude.  This ");
        printf("image\n");
        printf("         will take %ld mega-bytes of free memory to process.\n",
               ((num_cells_in_colormap * (long)sizeof(unsigned long)) / 1024) / 1024);
    }
    /*
    * set up temporary storage for the old and new pixel values
    */
    opv = (unsigned long *)malloc(num_cells_in_colormap * sizeof(unsigned long));
    npv = (unsigned long *)malloc(num_cells_in_colormap * sizeof(unsigned long));

    if (!opv || !npv)
    {
        printf("ERROR: unable to malloc the temporary pixel storage.\n");
        if (opv)
            free((char *)opv);
        if (npv)
            free((char *)npv);
        return (0);
    }
    /*
    * Each opv value is a flag indicating whether or not that pixel is
    * present in the image.  Each npv value is it's replacement value.
    */
    for (i = 0; i < num_cells_in_colormap; i++)
    {
        opv[i] = false;
        npv[i] = (unsigned long)UNALLOCATED;
    }

    for (i = 0; i < height; i++)
        /* examine each pxl in image; recrd all used
       * values */
        for (j = 0; j < width; j++)
        {
            tmp_pixel_value = XGetPixel(image, j, i);
            if (tmp_pixel_value >= num_cells_in_colormap)
            {
                printf("ERROR: Bad pixel value at x=%d, y=%d, pixel=%ld\n",
                       j, i, tmp_pixel_value);
            }
            else
                opv[tmp_pixel_value] = true;
        }

    /*
    * For each true opv allocate the colors
    */
    for (i = 0; i < num_cells_in_colormap; i++)
    {
        if (opv[i])
        {
            screen_in_out.red = cells[i].red;
            screen_in_out.green = cells[i].green;
            screen_in_out.blue = cells[i].blue;
            screen_in_out.flags = DoRed | DoGreen | DoBlue;

            if (XAllocColor(disp,
                            XDefaultColormapOfScreen(XDefaultScreenOfDisplay(disp)),
                            &screen_in_out) == 0)
            {
                printf("ERROR: out of colors on this display. ");
                printf("Cannot import image.\n");
                return (0);
            }
            else
                npv[i] = screen_in_out.pixel;
        }
    }

    XGetWindowAttributes(disp, win_id, &win_attr);

    if (win_attr.depth == image->depth) /* cool, this is easy */
    {
        for (i = 0; i < height; i++)
            for (j = 0; j < width; j++)
                XPutPixel(image, j, i, npv[XGetPixel(image, j, i)]);
    } /* oh darn it, this is tough */
    else
    {
        /*
       * the concept here is to creat a new image that IS the correct depth
       * and then do "PutPixel" into it, filling it with the correct color
       * pixels.
       */
        diff_depth_im = XCreateImage(disp, win_attr.visual, win_attr.depth,
                                     image->format, image->xoffset, NULL, width,
                                     height, image->bitmap_pad, 0);

        buffer_size = Image_Size(diff_depth_im,
                                 diff_depth_im->format); /* malloc pixel buffer */
        if ((buffer = (char *)malloc(buffer_size * sizeof(char))) == NULL)
        {
            printf("ERROR: Can't malloc data buffer for differing depth.\n");
            return (0);
        }
        diff_depth_im->data = buffer;

        for (i = 0; i < height; i++)
            for (j = 0; j < width; j++)
                XPutPixel(diff_depth_im, j, i,
                          npv[XGetPixel(image, j, i)]);

        *image_ptr = diff_depth_im;
        XDestroyImage(image); /* free up space of wrong depth image */
    }

    if (opv)
        free((char *)opv);
    if (npv)
        free((char *)npv);

    return (1);
}
