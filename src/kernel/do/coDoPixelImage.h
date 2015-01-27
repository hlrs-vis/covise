/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_PIXELIMAGE_H
#define CO_DO_PIXELIMAGE_H

/***************************************************************************
 **                                                           (C)1996 RUS  **
 **                                                                        **
 ** Description: Class-Declaration of a new COVISE-OBJECT                  **
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

#include "coDistributedObject.h"

namespace covise
{

class DOEXPORT coDoPixelImage : public coDistributedObject
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);

private:
    coShortShm pixelsize; // number of bytes stored per pixels
    coIntShm width; // number of Pixels in X-direction
    coIntShm height; // number of Pixels in Y-direction
    coIntShm format; // id for pixelformat
    coCharShmArray pixels; // image data

protected:
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoPixelImage *cloneObject(const coObjInfo &newinfo) const;

public:
    coDoPixelImage(const coObjInfo &info);
    coDoPixelImage(const coObjInfo &info, coShmArray *arr);
    coDoPixelImage(const coObjInfo &info, int width, int height, short psize, unsigned form,
                   const char **buffer);
    coDoPixelImage(const coObjInfo &info, int width, int height, short psize, unsigned form,
                   const char *buffer);
    coDoPixelImage(const coObjInfo &info, int width, int height, short psize, unsigned form);

    int getWidth() const;
    int getHeight() const;
    int getPixelsize() const;
    unsigned getFormat() const;
    char *getPixels() const;
    char &operator()(int x, int y);
    const char &operator()(int x, int y) const;
    char &operator[](int i);
    const char &operator[](int i) const;
};
}
#endif
