/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_GLYPH_H

class ft_pixmap
{

public:
    ft_pixmap();
    ~ft_pixmap();
    unsigned int width;
    unsigned int height;
    unsigned char *buffer;
    unsigned int rowstride;

    FT_Library freetype;
    FT_Face face;
};

class ft_char
{
public:
    FT_UInt index; /* glyph index */
    FT_Vector pos; /* location from baseline in 26.6 */
    FT_Glyph image; /* glyph bitmap */
};

class ft_string
{
public:
    ft_string();
    ~ft_string();
    unsigned int width;
    unsigned int height;
    size_t count; /* number of characters */
    ft_char *glyphs;
    size_t num_glyphs;
    FT_BBox bbox;
    FT_Matrix transform;
};

class FT
{

public:
    ft_pixmap *createPixmap();
    void drawBitmap(ft_pixmap *pixmap,
                    FT_BitmapGlyph bit,
                    FT_Vector where);

    void computeBBox(ft_string *string);
    ft_string *createString(ft_pixmap *pixmap, const char *text);
    void drawString(ft_pixmap *pixmap, ft_string *string);
    void destroyPixmap(ft_pixmap *pixmap);
    void destroyString(ft_string *string);
};
