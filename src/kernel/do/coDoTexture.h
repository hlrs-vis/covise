/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_TEXTURE_H
#define CO_DO_TEXTURE_H

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

#include "coDistributedObject.h"
namespace covise
{

class coDoPixelImage;

class DOEXPORT coDoTexture : public coDistributedObject
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);

private:
    coDoPixelImage *buffer;
    coIntShm border;
    coIntShm components;
    coIntShm level;
    coIntShm width;
    coIntShm height;
    coIntShmArray vertices;
    coIntShm num_coordinates;
    coFloatShmArray x_coordinates;
    coFloatShmArray y_coordinates;

protected:
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoTexture *cloneObject(const coObjInfo &newinfo) const;

public:
    coDoTexture(const coObjInfo &info);
    coDoTexture(const coObjInfo &info, coShmArray *arr);
    coDoTexture(const coObjInfo &info,
                coDoPixelImage *image, /* texture buffer        */
                int b, /* border                */
                int c, /* components            */
                int l, /* level                 */
                int nv, /* number of vertices    */
                int *vi, /* vertex-indices        */
                int nc, /* number of ccordinates */
                float **coords); /* x/y coordinates       */

    coDoPixelImage *getBuffer() const;
    int getBorder() const;
    int getComponents() const;
    int getLevel() const;
    int getWidth() const;
    int getHeight() const;
    int getNumVertices() const;
    int *getVertices() const;
    int getNumCoordinates() const;
    float **getCoordinates() const;
};
}
#endif
