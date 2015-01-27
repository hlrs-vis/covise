/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
 **                                                                        **
 ** Description: Class-Declaration of a new COVISE-OBJECT                  **
 **              for handling Texture-Data                                 **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                               (C) 1997                                 **
 **                             Paul Benoelken                             **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 ** Author: Paul Benoelken                                                 **
 **                                                                        **
 ** Date:  31.07.97  V1.0                                                  **
 ****************************************************************************/
#include "coDoTexture.h"

using namespace covise;

coDistributedObject *coDoTexture::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;
    ret = new coDoTexture(coObjInfo(), arr);
    return ret;
}

int coDoTexture::rebuildFromShm()
{
    covise_data_list dl[8];

    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }
    dl[0].type = UNKNOWN;
    dl[0].ptr = (void *)&buffer;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&border;
    dl[2].type = INTSHM;
    dl[2].ptr = (void *)&components;
    dl[3].type = INTSHM;
    dl[3].ptr = (void *)&level;
    dl[4].type = INTSHMARRAY;
    dl[4].ptr = (void *)&vertices;
    dl[5].type = INTSHM;
    dl[5].ptr = (void *)&num_coordinates;
    dl[6].type = FLOATSHMARRAY;
    dl[6].ptr = (void *)&x_coordinates;
    dl[7].type = FLOATSHMARRAY;
    dl[7].ptr = (void *)&y_coordinates;
    return restore_shared_dl(8, dl);
}

int coDoTexture::getObjInfo(int no, coDoInfo **il) const
{
    if (no == 8)
    {
        (*il)[0].description = "Texture Buffer";
        (*il)[1].description = "Border";
        (*il)[2].description = "Components";
        (*il)[3].description = "Level";
        (*il)[4].description = "Indices of Vertices";
        (*il)[5].description = "Number of Texture Coordinates",
        (*il)[6].description = "X-Coordinates of Texture";
        (*il)[7].description = "Y-Coordinates of Texture";
        return 8;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}

coDoTexture::coDoTexture(const coObjInfo &info)
    : coDistributedObject(info, "TEXTUR")
{
    // retrieval of Texture by name from shared memory
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

coDoTexture::coDoTexture(const coObjInfo &info,
                         coShmArray *arr)
    : coDistributedObject(info, "TEXTUR")
{ // retrieval of PixelImage by arr from shared memory
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

coDoTexture::coDoTexture(const coObjInfo &info,
                         coDoPixelImage *image,
                         int b,
                         int c,
                         int l,
                         int nv,
                         int *vi,
                         int nc,
                         float **coords)
    : coDistributedObject(info, "TEXTUR")
{
    // allocating memory in shared data space
    covise_data_list dl[8];

    vertices.set_length(nv);
    x_coordinates.set_length(nc);
    y_coordinates.set_length(nc);

    buffer = image;
    dl[0].type = DISTROBJ;
    dl[0].ptr = (void *)buffer;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&border;
    dl[2].type = INTSHM;
    dl[2].ptr = (void *)&components;
    dl[3].type = INTSHM;
    dl[3].ptr = (void *)&level;
    dl[4].type = INTSHMARRAY;
    dl[4].ptr = (void *)&vertices;
    dl[5].type = INTSHM;
    dl[5].ptr = (void *)&num_coordinates;
    dl[6].type = FLOATSHMARRAY;
    dl[6].ptr = (void *)&x_coordinates;
    dl[7].type = FLOATSHMARRAY;
    dl[7].ptr = (void *)&y_coordinates;
    new_ok = store_shared_dl(8, dl) != 0;
    if (!new_ok)
        return;
    // setting members in shared data space
    border = b;
    components = c;
    level = l;
    num_coordinates = nc;
    memcpy(vertices.getDataPtr(), vi, nv * sizeof(int));
    if (nc != 0)
    {
        memcpy(x_coordinates.getDataPtr(), coords[0], nc * sizeof(float));
        memcpy(y_coordinates.getDataPtr(), coords[1], nc * sizeof(float));
    }
}

coDoTexture *coDoTexture::cloneObject(const coObjInfo &newinfo) const
{
    return new coDoTexture(newinfo, getBuffer(), getBorder(), getComponents(), getLevel(), getNumVertices(), getVertices(), getNumCoordinates(), getCoordinates());
}

coDoPixelImage *coDoTexture::getBuffer() const
{
    return (coDoPixelImage *)buffer;
}

int coDoTexture::getBorder() const
{
    return border;
}

int coDoTexture::getComponents() const
{
    return components;
}

int coDoTexture::getLevel() const
{
    return level;
}

int coDoTexture::getNumVertices() const
{
    return vertices.get_length();
}

int *coDoTexture::getVertices() const
{
    return (int *)vertices.getDataPtr();
}

int coDoTexture::getNumCoordinates() const
{
    return num_coordinates;
}

float **coDoTexture::getCoordinates() const
{
    float **tmp = new float *[2];
    tmp[0] = (float *)x_coordinates.getDataPtr();
    tmp[1] = (float *)y_coordinates.getDataPtr();
    return tmp;
}
