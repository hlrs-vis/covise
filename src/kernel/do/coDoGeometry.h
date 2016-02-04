/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_GEOMETRY_H
#define CO_DO_GEOMETRY_H

/*
 $Log: covise_geometry.h,v $
 * Revision 1.1  1993/09/25  20:44:13  zrhk0125
 * Initial revision
 *
*/

#include "coDistributedObject.h"

/***********************************************************************\ 
 **                                                                     **
 **   Geometry class                                 Version: 1.0       **
 **                                                                     **
 **                                                                     **
 **   Description  : Classes for the handling of Geometry for the       **
 **                  renderer                   		               **
 **                                                                     **
 **   Classes      :                                                    **
 **                                                                     **
 **   Copyright (C) 1993     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                                                                     **
 **                                                                     **
 **   Author       : A. Wierse   (RUS)                                  **
 **                                                                     **
 **   History      :                                                    **
 **                  12.08.93  Ver 1.0                                  **
 **                                                                     **
 **                                                                     **
\***********************************************************************/
namespace covise
{

class DOEXPORT coDoGeometry : public coDistributedObject
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);

public:
    enum { NumChannels = 8 };

private:
    coIntShm geometry_type;
    const coDistributedObject *geometry;
    coIntShm color_attr;
    const coDistributedObject *colors[NumChannels];
    coIntShm normal_attr;
    const coDistributedObject *normals;
    coIntShm texture_attr;
    const coDistributedObject *texture;
    coIntShm vertexAttribute_attr;
    const coDistributedObject *vertexAttribute;

protected:
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoGeometry *cloneObject(const coObjInfo &newinfo) const;

public:
    coDoGeometry(const coObjInfo &info)
        : coDistributedObject(info, "GEOMET")
    {
        if (name != (char *)NULL)
        {
            if (getShmArray() != 0)
            {
                if (rebuildFromShm() == 0)
                {
                    print_comment(__LINE__, __FILE__, "rebuildFromShm == 0");
                }
            }
            else
            {
                print_comment(__LINE__, __FILE__, "object %s doesn't exist", name);
                new_ok = 0;
            }
        }
    };

    void setGeometry(int gtype, const coDistributedObject *geo);
    void setColors(int cattr, const coDistributedObject *c, size_t chan = 0);
    void setNormals(int nattr, const coDistributedObject *n);
    void setTexture(int tattr, const coDistributedObject *t);
    void setVertexAttribute(int vattr, const coDistributedObject *v);

    coDoGeometry(const coObjInfo &info, coShmArray *arr);
    coDoGeometry(const coObjInfo &info, const coDistributedObject *geo);
    const coDistributedObject *getGeometry() const
    {
        return geometry;
    };
    const coDistributedObject *getColors(size_t chan = 0) const
    {
        return colors[chan];
    };
    const coDistributedObject *getNormals() const
    {
        return normals;
    };
    const coDistributedObject *getTexture() const
    {
        return texture;
    };
    const coDistributedObject *getVertexAttribute() const
    {
        return vertexAttribute;
    };

    int getGeometryType() const
    {
        return geometry_type;
    }
    int getColorAttributes() const
    {
        return color_attr;
    }
    void setColorAttributes(int cattr)
    {
        color_attr = cattr;
    }
    int getNormalAttributes() const
    {
        return normal_attr;
    }
    void setNormalAttributes(int nattr)
    {
        normal_attr = nattr;
    }
    int getTextureAttributes() const
    {
        return texture_attr;
    }
    void setTextureAttributes(int nattr)
    {
        texture_attr = nattr;
    }
    int getVertexAttributeAttributes() const
    {
        return vertexAttribute_attr;
    }
    void setVertexAttributeAttributes(int vattr)
    {
        vertexAttribute_attr = vattr;
    }
};
}
#endif
