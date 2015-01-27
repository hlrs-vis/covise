/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <math.h>

#include "ft.h"

#define FT_BYTES_PER_PIXEL 4
#define FT_FONT "/usr/share/fonts/bitstream-vera/Vera.ttf"
//#define FT_FONT "/usr/share/fonts/truetype/ttf-bitstream-vera/Vera.ttf"
#define FT_FONT_SIZE 64

ft_pixmap::ft_pixmap()
    : buffer(NULL)
{
}

ft_pixmap::~ft_pixmap()
{
    delete[] buffer;
    FT_Done_Face(face);
    FT_Done_FreeType(freetype);
}

ft_string::ft_string()
{
}

ft_pixmap *FT::createPixmap()
{
    ft_pixmap *pixmap = new ft_pixmap();
    int error;

    /* initialize font service */
    error = FT_Init_FreeType(&(pixmap->freetype));
    if (error)
    {
        fprintf(stderr, "couldn't initialize the font library\n");
        return NULL;
    }
    error = FT_New_Face(pixmap->freetype, FT_FONT, 0, &(pixmap->face));
    if (error)
    {
        fprintf(stderr, "couldn't load font \"%s\"", FT_FONT);
        return NULL;
    }

    error = FT_Set_Char_Size(pixmap->face, 0, FT_FONT_SIZE * 64, 72, 72);
    if (error)
    {
        fprintf(stderr, "couldn't set the font size to \"%d\"", FT_FONT_SIZE);
        return NULL;
    }

    return pixmap;
}

void FT::drawBitmap(ft_pixmap *pixmap,
                    FT_BitmapGlyph bit,
                    FT_Vector where)
{
    unsigned int grays;
    unsigned int row, column;

    /* nothing to do */
    if (bit->bitmap.rows == 0)
    {
        fprintf(stderr, "no bitmap rows\n");
        return;
    }

    grays = bit->bitmap.num_grays - 1;
    for (row = 0; row < bit->bitmap.rows; ++row)
    {
        int pixmap_offset_y = (int)((where.y) + 0.5 + row - bit->top);
        /* don't draw off top or bottom of image */
        if (pixmap_offset_y < 0 || pixmap_offset_y >= pixmap->height)
        {
            fprintf(stderr, "bitmap off top or bottom of image\n");
            continue;
        }
        pixmap_offset_y *= pixmap->rowstride;

        for (column = 0; column < bit->bitmap.width; ++column)
        {
            int pixmap_offset_x, pixmap_offset, bitmap_offset;
            unsigned char red, green, blue, newa;
            double old_level, new_level;

            pixmap_offset_x = (int)((where.x) - 0.5 + column + bit->left);
            /* don't draw off left or right of image */
            if (pixmap_offset_x < 0 || pixmap_offset_x >= pixmap->width)
            {
                fprintf(stderr, "bitmap off left or right of image\n");
                continue;
            }
            pixmap_offset_x *= FT_BYTES_PER_PIXEL;
            pixmap_offset = pixmap_offset_y + pixmap_offset_x;

            /* don't draw outside of allocated space */
            if (pixmap_offset < 0 || (pixmap_offset + 2) >= (pixmap->height * pixmap->rowstride))
            {
                fprintf(stderr, "bitmap outside bounds of image\n");
                continue;
            }

            red = pixmap->buffer[pixmap_offset + 0];
            green = pixmap->buffer[pixmap_offset + 1];
            blue = pixmap->buffer[pixmap_offset + 2];

            bitmap_offset = row * bit->bitmap.width + column;
            newa = *(bit->bitmap.buffer + bitmap_offset);

            /* apply color alpha */
            newa *= (unsigned char)((double)(0xFF) / (double)grays);

            new_level = (double)newa / (double)grays;
            old_level = 1.0 - new_level;
            red = (unsigned char)(double)(red * old_level + (double)(0x00) * new_level);
            green = (unsigned char)(double)(green * old_level + (double)(0x00) * new_level);
            blue = (unsigned char)(double)(blue * old_level + (double)(0x00) * new_level);

            pixmap->buffer[pixmap_offset + 0] = red;
            pixmap->buffer[pixmap_offset + 1] = green;
            pixmap->buffer[pixmap_offset + 2] = blue;

            if (newa)
                pixmap->buffer[pixmap_offset + 3] = newa; //
        }
    }
}

void FT::computeBBox(ft_string *string)
{
    int n;
    /* initialize string bbox to "empty" values */
    string->bbox.xMin = string->bbox.yMin = 32000;
    string->bbox.xMax = string->bbox.yMax = -32000;

    /* for each glyph image, compute its bounding box, */
    /* translate it, and grow the string bbox          */
    for (n = 0; n < string->num_glyphs; n++)
    {
        FT_BBox glyph_bbox;

        FT_Glyph_Get_CBox(string->glyphs[n].image, ft_glyph_bbox_pixels,
                          &glyph_bbox);
        if (glyph_bbox.xMin < string->bbox.xMin)
            string->bbox.xMin = glyph_bbox.xMin;

        if (glyph_bbox.yMin < string->bbox.yMin)
            string->bbox.yMin = glyph_bbox.yMin;

        if (glyph_bbox.xMax > string->bbox.xMax)
            string->bbox.xMax = glyph_bbox.xMax;

        if (glyph_bbox.yMax > string->bbox.yMax)
            string->bbox.yMax = glyph_bbox.yMax;
    }

    /* check that we really grew the string bbox */
    if (string->bbox.xMin > string->bbox.xMax)
    {
        string->bbox.xMin = 0;
        string->bbox.yMin = 0;
        string->bbox.xMax = 0;
        string->bbox.yMax = 0;
    }
}

ft_string *FT::createString(ft_pixmap *pixmap, const char *text)
{
    FT_GlyphSlot slot = pixmap->face->glyph; /* a small shortcut */
    FT_Bool use_kerning;
    FT_UInt previous;
    FT_Vector ft_pen;

    ft_string *string;
    ft_char *glyph; /* current glyph in table */
    int n, error;

    ft_pen.x = 0; /* start at (0,0) !! */
    ft_pen.y = 0;

    string = new ft_string();
    string->width = 0;
    string->height = 0;
    string->count = strlen(text);
    string->glyphs = new ft_char[string->count];
    string->num_glyphs = 0;

    string->transform.xx = (FT_Fixed)(0x10000);
    string->transform.xy = (FT_Fixed)(0);
    string->transform.yx = (FT_Fixed)(0);
    string->transform.yy = (FT_Fixed)(0x10000);

    use_kerning = FT_HAS_KERNING(pixmap->face);
    previous = 0;
    glyph = string->glyphs;
    for (n = 0; n < string->count; n++, glyph++)
    {
        FT_Vector vec;

        /* initialize each struct ft_char_s */
        glyph->index = 0;
        glyph->pos.x = 0;
        glyph->pos.y = 0;
        glyph->image = NULL;

        glyph->index = FT_Get_Char_Index(pixmap->face, text[n]);

        /* compute glyph origin */
        if (use_kerning && previous && glyph->index)
        {
            FT_Vector kerning;
            FT_Get_Kerning(pixmap->face, previous, glyph->index,
                           ft_kerning_default, &kerning);
            ft_pen.x += kerning.x;
            ft_pen.y += kerning.y;
        }

        /* store current pen position */
        glyph->pos.x = ft_pen.x;
        glyph->pos.y = ft_pen.y;

        /* load the glyph image (in its native format) */
        /* for now, we take a monochrome glyph bitmap */
        error = FT_Load_Glyph(pixmap->face, glyph->index, FT_LOAD_DEFAULT);
        if (error)
        {
            fprintf(stderr, "couldn't load glyph:  %c\n", text[n]);
            continue;
        }
        error = FT_Get_Glyph(slot, &glyph->image);
        if (error)
        {
            fprintf(stderr, "couldn't get glyph from slot:  %c\n", text[n]);
            continue;
        }

        ft_pen.x += slot->advance.x;
        ft_pen.y += slot->advance.y;

        /* translate glyph */
        vec = glyph->pos;
        FT_Vector_Transform(&vec, &string->transform);
        error = FT_Glyph_Transform(glyph->image, &string->transform, &vec);
        if (error)
        {
            fprintf(stderr, "couldn't transform glyph\n");
            continue;
        }

        /* convert to a bitmap - destroy native image */
        error = FT_Glyph_To_Bitmap(&glyph->image, ft_render_mode_normal, 0, 1);
        if (error)
        {
            fprintf(stderr, "couldn't convert glyph to bitmap\n");
            continue;
        }

        /* increment number of glyphs */
        previous = glyph->index;
        string->num_glyphs++;
    }

    computeBBox(string);

    pixmap->width = string->bbox.xMax - string->bbox.xMin;
    pixmap->height = string->bbox.yMax - string->bbox.yMin;
    pixmap->rowstride = pixmap->width * FT_BYTES_PER_PIXEL;

    int size = pixmap->height * pixmap->rowstride;
    pixmap->buffer = new unsigned char[size];
    printf("-----%d %d: %d\n", pixmap->width, pixmap->height, size);
    memset(pixmap->buffer, 0x00, size);

    return string;
}

void FT::drawString(ft_pixmap *pixmap, ft_string *string)
{
    FT_Vector where;
    ft_char *glyph;
    int n, error;

    where.x = -string->bbox.xMin + 1;
    where.y = string->bbox.yMax;
    glyph = string->glyphs;

    for (n = 0; n < string->num_glyphs; ++n, ++glyph)
    {
        FT_Glyph image;
        FT_Vector vec; /* 26.6 */
        FT_BitmapGlyph bit;

        // make copy to transform
        if (!glyph->image)
        {
            fprintf(stderr, "no image\n");
            continue;
        }
        error = FT_Glyph_Copy(glyph->image, &image);
        if (error)
        {
            fprintf(stderr, "couldn't copy image\n");
            continue;
        }

        /* transform it */
        vec = glyph->pos;
        FT_Vector_Transform(&vec, &string->transform);

        bit = (FT_BitmapGlyph)image;
        drawBitmap(pixmap, bit, where);
        FT_Done_Glyph(image);
    }
}

ft_string::~ft_string()
{
    int n;
    if (glyphs)
    {
        for (n = 0; n < num_glyphs; ++n)
            FT_Done_Glyph(glyphs[n].image);
        delete glyphs;
    }
}
/*
void FT::savePixmap(ft_pixmap *pixmap, const char *filename)
{
   FILE          *fp;
   png_structp   png_ptr;
   png_infop     info_ptr;
   unsigned char *image_data, **row_pointers;
   unsigned int  width, height, rowspan, row;

   fp = fopen(filename, "wb");
   if (!fp) {
      fprintf(stderr, "couldn't open file \"%s\":  %s", filename, strerror(errno));
      return;
   }

   png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

   if (!png_ptr) {
      fclose(fp);
      fprintf(stderr, "couldn't find memory for writing PNG image");
      return;
   }

   // allocate/initialize the image information data
   info_ptr = png_create_info_struct(png_ptr);
   if (!info_ptr) {
      fclose(fp);
      png_destroy_write_struct(&png_ptr, (png_infopp) NULL);
      fprintf(stderr, "couldn't find memory for writing PNG image");
      return;
   }

   if (setjmp(png_jmpbuf(png_ptr))) {
      fclose(fp);
      png_destroy_write_struct(&png_ptr, &info_ptr);
      fprintf(stderr, "error occured while writing to file \"%s\"", filename);
      return;
   }

   png_init_io(png_ptr, fp);
   width = pixmap->width;
   height = pixmap->height;
   png_set_IHDR(png_ptr, info_ptr, width, height,
                8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

   png_write_info(png_ptr, info_ptr);

   image_data = pixmap->buffer;
   rowspan = pixmap->rowstride;
   row_pointers = (unsigned char **) malloc(height * sizeof(png_bytep));
   for (row = 0; row < height; ++row)
      row_pointers[row] = &(image_data[row * rowspan]);

   png_set_rows(png_ptr, info_ptr, row_pointers);
   png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

   png_destroy_write_struct(&png_ptr, &info_ptr);
   free(row_pointers);
   fclose(fp);
}
*/
/*
int main(int argc, char * argv [])
{
   ft_pixmap pixmap;
   ft_string string;

   pixmap = ft_pixmap_create();

   string = ft_string_create(pixmap, argv[1]);

   ft_pixmap_draw_string(pixmap, string);
   ft_string_destroy(string);

   ft_pixmap_save(pixmap, FT_FILENAME);
   ft_pixmap_destroy(pixmap);

   return 0;
}
*/
