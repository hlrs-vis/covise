/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef B2NMAP_H
#define B2NMAP_H

#include <png.h>
#include "glh_array.h"
#include "bumpmap_to_normalmap.h"
#include <iostream>

int write_png_normalmap(const char *filename, char *image, int imgsize)
{
    FILE *fp;
    png_uint_32 row_bytes;
    png_structp png_ptr = 0;
    png_infop info_ptr = 0;

    // open the PNG output file
    if (!filename)
        return -1;

    if (!(fp = fopen(filename, "wb")))
    {
        printf("Not able to open file for writing: %s\n", filename);
        return -1;
    }

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
    if (!png_ptr)
    {
        if (fp)
            fclose(fp);
        printf("Error while creating png-write-structure.\n");
        return -1;
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        png_destroy_write_struct(&png_ptr, 0);
        if (fp)
            fclose(fp);
        printf("Error while creating png-info-structure.\n");
        return -1;
    }

    // initialize the png structure
    png_init_io(png_ptr, fp);

    int h = imgsize;
    int w = imgsize;

    png_set_IHDR(png_ptr, info_ptr, w, h,
                 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    //  row_bytes = w * sizeof(short) * 2;
    row_bytes = w * 3;

    png_byte **row = new png_byte *[h];

    // set the individual row-pointers to point at the correct offsets

    png_byte *img = (png_byte *)image;

    for (unsigned int i = 0; i < h; i++)
        row[i] = img + i * row_bytes;

    //  png_set_rows(png_ptr, info_ptr, row);

    printf("Kurz vor dem Schreiben.\n");

    //  png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

    png_write_info(png_ptr, info_ptr);

    printf("Info geschrieben.\n");

    png_write_image(png_ptr, row);

    printf("Image geschrieben.\n");

    png_write_end(png_ptr, 0);

    printf("Nach dem Schreiben.\n");

    delete[] row;

    png_destroy_write_struct(&png_ptr, &info_ptr);

    if (fp)
        fclose(fp);
    return 0;
}

int read_png_grey(const char *filename, glh::array2<unsigned char> &image)
{
    FILE *fp;
    png_byte sig[8];
    int bit_depth, color_type;
    double gamma;
    png_uint_32 channels, row_bytes;
    png_structp png_ptr = 0;
    png_infop info_ptr = 0;

    // open the PNG input file
    if (!filename)
        return -1;

    if (!(fp = fopen(filename, "rb")))
    {
        printf("File not found: %s\n", filename);
        return -1;
    }

    // first check the eight byte PNG signature
    size_t retval;
    retval = fread(sig, 1, 8, fp);
    if (retval != 8)
    {
        std::cerr << "read_png_grey: fread failed" << std::endl;
        return -1;
    }
    if (!png_check_sig(sig, 8))
    {
        if (fp)
            fclose(fp);
        printf("File is not a png file: %s\n", filename);
        return -1;
    }

    // start back here!!!!

    // create the two png(-info) structures

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
    if (!png_ptr)
    {
        if (fp)
            fclose(fp);
        printf("Error while creating png-read-structure.\n");
        return -1;
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        png_destroy_read_struct(&png_ptr, 0, 0);
        if (fp)
            fclose(fp);
        printf("Error while creating png-info-structure.\n");
        return -1;
    }

    // initialize the png structure
    png_init_io(png_ptr, fp);

    png_set_sig_bytes(png_ptr, 8);

    // read all PNG info up to image data
    png_read_info(png_ptr, info_ptr);

    // get width, height, bit-depth and color-type
    png_uint_32 w, h;
    png_get_IHDR(png_ptr, info_ptr, &w, &h, &bit_depth, &color_type, 0, 0, 0);

    // expand images of all color-type and bit-depth to 3x8 bit RGB images
    // let the library process things like alpha, transparency, background

    if (bit_depth == 16)
        png_set_strip_16(png_ptr);
    if (color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_expand(png_ptr);
    if (bit_depth < 8)
        png_set_expand(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
        png_set_expand(png_ptr);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png_ptr);

    // if required set gamma conversion
    if (png_get_gAMA(png_ptr, info_ptr, &gamma))
        png_set_gamma(png_ptr, (double)2.2, gamma);

    // after the transformations have been registered update info_ptr data
    png_read_update_info(png_ptr, info_ptr);

    // get again width, height and the new bit-depth and color-type
    png_get_IHDR(png_ptr, info_ptr, &w, &h, &bit_depth, &color_type, 0, 0, 0);

    // row_bytes is the width x number of channels
    row_bytes = png_get_rowbytes(png_ptr, info_ptr);
    channels = png_get_channels(png_ptr, info_ptr);

    // now we can allocate memory to store the image

    png_byte *img = new png_byte[row_bytes * h];

    // and allocate memory for an array of row-pointers

    png_byte **row = new png_byte *[h];

    // set the individual row-pointers to point at the correct offsets

    for (unsigned int i = 0; i < h; i++)
        row[i] = img + i * row_bytes;

    // now we can go ahead and just read the whole image

    png_read_image(png_ptr, row);

    // read the additional chunks in the PNG file (not really needed)

    png_read_end(png_ptr, NULL);

    image = glh::array2<unsigned char>(w, h);
    {
        for (unsigned int i = 0; i < w; i++)
            for (unsigned int j = 0; j < h; j++)
            //	{ image(i,j) = *(img + ((h-j-1)*row_bytes + i * 3)); }
            {
                image(i, j) = *(img + (j * row_bytes + i * 3));
            }
    }

    delete[] row;
    delete[] img;

    png_destroy_read_struct(&png_ptr, &info_ptr, 0);

    if (fp)
        fclose(fp);
    return 0;
}
#endif
