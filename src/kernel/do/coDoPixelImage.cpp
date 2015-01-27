/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
 **                                                           (C)1996 RUS  **
 **                                                                        **
 ** Description: Constructors and Member-Functions of the new COVISE-OBJECT**
 **              for handling Image-Data                                   **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                               (C) 1996                                 **
 **                             Paul Benoelken                             **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 ** Author: Paul Benoelken                                                 **
 **                                                                        **
 ** Date:  04.12.96  V1.0                                                  **
 ****************************************************************************/

#include "coDoPixelImage.h"

using namespace covise;

coDistributedObject *coDoPixelImage::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;

    ret = new coDoPixelImage(coObjInfo(), arr);
    return ret;
}

/**************************************************************************
 *						 CONSTRUCTORS									  *
 **************************************************************************/

coDoPixelImage::coDoPixelImage(const coObjInfo &info)
    : coDistributedObject(info, "IMAGE")
{
    // retrieval of PixelImage by name from shared memory
    if (info.getName())
    {
        if (getShmArray() != 0)
        {
            if (rebuildFromShm() == 0)
                print_comment(__LINE__, __FILE__, "rebuildFromShm == 0");
        }
        else
        {
            print_comment(__LINE__, __FILE__, "object %s doesn't exist", info.getName());
            new_ok = 0;
        }
    }
}

coDoPixelImage::coDoPixelImage(const coObjInfo &info,
                               coShmArray *arr)
    : coDistributedObject(info, "IMAGE")
{
    // retrieval of PixelImage by arr from shared memory
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

coDoPixelImage::coDoPixelImage(const coObjInfo &info,
                               int x_size,
                               int y_size,
                               short psize,
                               unsigned form,
                               const char **buffer)
    : coDistributedObject(info, "IMAGE")
{
    // allocating memory in shared data space
    pixels.set_length(x_size * y_size * psize);

    covise_data_list dl[] = {
        { INTSHM, &width },
        { INTSHM, &height },
        { SHORTSHM, &pixelsize },
        { INTSHM, &format },
        { CHARSHMARRAY, &pixels }
    };

    new_ok = store_shared_dl(5, dl) != 0;

    if (!new_ok)
        return;
    // setting member variables
    width = x_size;
    height = y_size;
    pixelsize = psize;
    format = form;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            pixels[y * width + x] = buffer[y][x];
}

coDoPixelImage::coDoPixelImage(const coObjInfo &info,
                               int x_size,
                               int y_size,
                               short psize,
                               unsigned form,
                               const char *buffer)
    : coDistributedObject(info, "IMAGE")
{
    int size = x_size * y_size * psize;

    // allocating memory in shared data space
    pixels.set_length(size);

    covise_data_list dl[] = {
        { INTSHM, &width },
        { INTSHM, &height },
        { SHORTSHM, &pixelsize },
        { INTSHM, &format },
        { CHARSHMARRAY, &pixels }
    };

    new_ok = store_shared_dl(5, dl) != 0;
    if (!new_ok)
        return;
    // setting member variables
    width = x_size;
    height = y_size;
    pixelsize = psize;
    format = form;
    for (int i = 0; i < size; i++)
        pixels[i] = buffer[i];
}

coDoPixelImage::coDoPixelImage(const coObjInfo &info,
                               int x_size,
                               int y_size,
                               short psize,
                               unsigned form)
    : coDistributedObject(info, "IMAGE")
{
    // allocating memory in shared data space
    pixels.set_length(x_size * y_size * psize);
    covise_data_list dl[] = {
        { INTSHM, &width },
        { INTSHM, &height },
        { SHORTSHM, &pixelsize },
        { INTSHM, &format },
        { CHARSHMARRAY, &pixels }
    };
    new_ok = store_shared_dl(5, dl) != 0;
    if (!new_ok)
        return;
    // setting member variables
    width = x_size;
    height = y_size;
    pixelsize = psize;
    format = form;
}

coDoPixelImage *coDoPixelImage::cloneObject(const coObjInfo &newinfo) const
{
    return new coDoPixelImage(newinfo, getWidth(), getHeight(), getPixelsize(), getFormat(), getPixels());
}

/*****************************************************************************
 *							ACCESS-FUNCTIONS     *
 ****************************************************************************/

int coDoPixelImage::getWidth() const
{
    return width;
}

int coDoPixelImage::getHeight() const
{
    return height;
}

int coDoPixelImage::getPixelsize() const
{
    return pixelsize;
}

unsigned coDoPixelImage::getFormat() const
{
    return format;
}

char *coDoPixelImage::getPixels() const
{
    return (char *)pixels.getDataPtr();
}

const char &coDoPixelImage::operator()(int x, int y) const
{
    if (x < width && y < height)
        return pixels[y * width + x];
    else
    {
        cerr << "indices out of range check first pixel-byte !" << endl;
        return pixels[0];
    }
}

char &coDoPixelImage::operator()(int x, int y)
{
    if (x < width && y < height)
        return pixels[y * width + x];
    else
    {
        cerr << "indices out of range check first pixel-byte !" << endl;
        return pixels[0];
    }
}

const char &coDoPixelImage::operator[](int i) const
{
    if (i < width * height * pixelsize)
        return pixels[i];
    else
    {
        cerr << "index out of range check first pixel-byte !" << endl;
        return pixels[0];
    }
}

char &coDoPixelImage::operator[](int i)
{
    if (i < width * height * pixelsize)
        return pixels[i];
    else
    {
        cerr << "index out of range check first pixel-byte !" << endl;
        return pixels[0];
    }
}

/*****************************************************************************
 *							       Shm-FUNCTIONS *
 *****************************************************************************/

int coDoPixelImage::rebuildFromShm()
{
    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }
    covise_data_list dl[] = {
        { INTSHM, &width },
        { INTSHM, &height },
        { SHORTSHM, &pixelsize },
        { INTSHM, &format },
        { CHARSHMARRAY, &pixels }
    };
    return restore_shared_dl(5, dl);
}

int coDoPixelImage::getObjInfo(int no, coDoInfo **il) const
{
    if (no == 5)
    {
        (*il)[0].description = "Image Width";
        (*il)[1].description = "Image Height";
        (*il)[2].description = "Size of each Pixel";
        (*il)[3].description = "ID for Pixel Format";
        (*il)[4].description = "Pixel Values";

        return 5;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}
