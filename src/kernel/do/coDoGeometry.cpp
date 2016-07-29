/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coDoGeometry.h"

/*
 $Log: covise_geometry.C,v $
Revision 1.1  1993/09/25  20:44:49  zrhk0125
Initial revision

*/

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

using namespace covise;

coDistributedObject *coDoGeometry::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;
    ret = new coDoGeometry(coObjInfo(), arr);
    return ret;
}

coDoGeometry::coDoGeometry(const coObjInfo &info, coShmArray *arr)
    : coDistributedObject(info, "GEOMET")
{
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

int coDoGeometry::getObjInfo(int no, coDoInfo **il) const
{
    if (no == 10 + NumChannels + NumColorMaps)
    {
        (*il)[0].description = "Geometry Type";
        (*il)[1].description = "Geometry";
        (*il)[2].description = "Color Attribute: not used";
        for (int c = 0; c < NumChannels; ++c)
            (*il)[3 + c].description = "Colors";
        (*il)[3 + NumChannels].description = "Normal Attribute: not used";
        (*il)[4 + NumChannels].description = "Normals";
        (*il)[5 + NumChannels].description = "Texture Type";
        (*il)[6 + NumChannels].description = "Texture";
        (*il)[7 + NumChannels].description = "Vertex Attribute Type";
        (*il)[8 + NumChannels].description = "Vertex Attribute";
        (*il)[9 + NumChannels].description = "Color Map Type";
        for (int c = 0; c < NumColorMaps; ++c)
            (*il)[10 + NumChannels + c].description = "ColorMap";
        return 10 + NumChannels + NumColorMaps;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}

coDoGeometry::coDoGeometry(const coObjInfo &info, const coDistributedObject *geo)
    : coDistributedObject(info, "GEOMET")
{

    geometry = geo;
    for (int c = 0; c < NumChannels; ++c)
    {
        colors[c] = NULL;
    }
    normals = NULL;
    texture = NULL;
    vertexAttribute = NULL;
    for (int c = 0; c < NumColorMaps; ++c)
    {
        colorMap[c] = NULL;
    }

    covise_data_list dl[10 + NumChannels + NumColorMaps];
    dl[0].type = INTSHM;
    dl[0].ptr  = (void *)&geometry_type;
    dl[1].type = DISTROBJ;
    dl[1].ptr  = (void *)geometry;
    dl[2].type = INTSHM;
    dl[2].ptr  = (void *)&color_attr;
    for (int c = 0; c < NumChannels; ++c)
    {
        dl[3 + c].type = DISTROBJ;
        dl[3 + c].ptr  = (void *)colors[c];
    }
    dl[3 + NumChannels].type = INTSHM;
    dl[3 + NumChannels].ptr  = (void *)&normal_attr;
    dl[4 + NumChannels].type = DISTROBJ;
    dl[4 + NumChannels].ptr  = (void *)normals;
    dl[5 + NumChannels].type = INTSHM;
    dl[5 + NumChannels].ptr  = (void *)&texture_attr;
    dl[6 + NumChannels].type = DISTROBJ;
    dl[6 + NumChannels].ptr  = (void *)texture;
    dl[7 + NumChannels].type = INTSHM;
    dl[7 + NumChannels].ptr  = (void *)&vertexAttribute_attr;
    dl[8 + NumChannels].type = DISTROBJ;
    dl[8 + NumChannels].ptr  = (void *)vertexAttribute;
    dl[9 + NumChannels].type = INTSHM;
    dl[9 + NumChannels].ptr  = (void *)&colorMap_attr;
    for (int c = 0; c < NumColorMaps; ++c)
    {
        dl[10 + NumChannels + c].type = DISTROBJ;
        dl[10 + NumChannels + c].ptr  = (void *)colorMap[c];
    }

    new_ok = store_shared_dl(10 + NumChannels + NumColorMaps, dl) != 0;
    if (!new_ok)
        return;
    geometry_type = geo->get_type_no();
    color_attr = NONE;
    normal_attr = NONE;
    texture_attr = NONE;
    vertexAttribute_attr = NONE;
    colorMap_attr = NONE;
}

coDoGeometry *coDoGeometry::cloneObject(const coObjInfo &newinfo) const
{
    coDoGeometry *geo = new coDoGeometry(newinfo);
    geo->setGeometry(getGeometryType(), getGeometry());
    for (int c = 0; c < NumChannels; ++c)
        geo->setColors(getColorAttributes(), getColors(c), c);
    geo->setNormals(getNormalAttributes(), getNormals());
    geo->setTexture(getTextureAttributes(), getTexture());
    geo->setVertexAttribute(getVertexAttributeAttributes(), getVertexAttribute());
    for (int c = 0; c < NumColorMaps; ++c)
        geo->setColorMap(getColorMapAttributes(), getColorMap(c), c);
    return geo;
}

int coDoGeometry::rebuildFromShm()
{
    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }

    covise_data_list dl[10 + NumChannels + NumColorMaps];
    dl[0].type = INTSHM;
    dl[0].ptr  = (void *)&geometry_type;
    dl[1].type = UNKNOWN;
    dl[1].ptr  = (void *)&geometry;
    dl[2].type = INTSHM;
    dl[2].ptr  = (void *)&color_attr;
    for (int c = 0; c < NumChannels; ++c)
    {
        dl[3 + c].type = COVISE_OPTIONAL;
        dl[3 + c].ptr  = (void *)&colors[c];
    }
    dl[3 + NumChannels].type = INTSHM;
    dl[3 + NumChannels].ptr  = (void *)&normal_attr;
    dl[4 + NumChannels].type = COVISE_OPTIONAL;
    dl[4 + NumChannels].ptr  = (void *)&normals;
    dl[5 + NumChannels].type = INTSHM;
    dl[5 + NumChannels].ptr  = (void *)&texture_attr;
    dl[6 + NumChannels].type = COVISE_OPTIONAL;
    dl[6 + NumChannels].ptr  = (void *)&texture;
    dl[7 + NumChannels].type = INTSHM;
    dl[7 + NumChannels].ptr  = (void *)&vertexAttribute_attr;
    dl[8 + NumChannels].type = COVISE_OPTIONAL;
    dl[8 + NumChannels].ptr  = (void *)&vertexAttribute;
    dl[9 + NumChannels].type = INTSHM;
    dl[9 + NumChannels].ptr  = (void *)&colorMap_attr;
    for (int c = 0; c < NumColorMaps; ++c)
    {
        dl[10 + NumChannels + c].type = COVISE_OPTIONAL;
        dl[10 + NumChannels + c].ptr  = (void *)&colorMap[c];
    }

    return restore_shared_dl(10 + NumChannels + NumColorMaps, dl);
}

void coDoGeometry::setGeometry(int gtype, const coDistributedObject *geo)
{
    geometry_type = gtype;
    geometry = geo;
    covise_data_list dl[10 + NumChannels + NumColorMaps];
    dl[0].type = INTSHM;
    dl[0].ptr  = (void *)&geometry_type;
    dl[1].type = DISTROBJ;
    dl[1].ptr  = (void *)geometry;
    dl[2].type = INTSHM;
    dl[2].ptr  = (void *)&color_attr;
    for (int c = 0; c < NumChannels; ++c)
    {
        dl[3 + c].type = DISTROBJ;
        dl[3 + c].ptr  = (void *)colors[c];
    }
    dl[3 + NumChannels].type = INTSHM;
    dl[3 + NumChannels].ptr  = (void *)&normal_attr;
    dl[4 + NumChannels].type = DISTROBJ;
    dl[4 + NumChannels].ptr  = (void *)normals;
    dl[5 + NumChannels].type = INTSHM;
    dl[5 + NumChannels].ptr  = (void *)&texture_attr;
    dl[6 + NumChannels].type = DISTROBJ;
    dl[6 + NumChannels].ptr  = (void *)texture;
    dl[7 + NumChannels].type = INTSHM;
    dl[7 + NumChannels].ptr  = (void *)&vertexAttribute_attr;
    dl[8 + NumChannels].type = DISTROBJ;
    dl[8 + NumChannels].ptr  = (void *)vertexAttribute;
    dl[9 + NumChannels].type = INTSHM;
    dl[9 + NumChannels].ptr  = (void *)&colorMap_attr;
    for (int c = 0; c < NumColorMaps; ++c)
    {
        dl[10 + NumChannels + c].type = DISTROBJ;
        dl[10 + NumChannels + c].ptr  = (void *)colorMap[c];
    }
    update_shared_dl(10 + NumChannels + NumColorMaps, dl);
}

void coDoGeometry::setColors(int cattr, const coDistributedObject *c, size_t chan)
{
    color_attr = cattr;
    colors[chan] = c;
    covise_data_list dl[10 + NumChannels + NumColorMaps];
    dl[0].type = INTSHM;
    dl[0].ptr  = (void *)&geometry_type;
    dl[1].type = DISTROBJ;
    dl[1].ptr  = (void *)geometry;
    dl[2].type = INTSHM;
    dl[2].ptr  = (void *)&color_attr;
    for (int c = 0; c < NumChannels; ++c)
    {
        dl[3 + c].type = DISTROBJ;
        dl[3 + c].ptr  = (void *)colors[c];
    }
    dl[3 + NumChannels].type = INTSHM;
    dl[3 + NumChannels].ptr  = (void *)&normal_attr;
    dl[4 + NumChannels].type = DISTROBJ;
    dl[4 + NumChannels].ptr  = (void *)normals;
    dl[5 + NumChannels].type = INTSHM;
    dl[5 + NumChannels].ptr  = (void *)&texture_attr;
    dl[6 + NumChannels].type = DISTROBJ;
    dl[6 + NumChannels].ptr  = (void *)texture;
    dl[7 + NumChannels].type = INTSHM;
    dl[7 + NumChannels].ptr  = (void *)&vertexAttribute_attr;
    dl[8 + NumChannels].type = DISTROBJ;
    dl[8 + NumChannels].ptr  = (void *)vertexAttribute;
    dl[9 + NumChannels].type = INTSHM;
    dl[9 + NumChannels].ptr  = (void *)&colorMap_attr;
    for (int c = 0; c < NumColorMaps; ++c)
    {
        dl[10 + NumChannels + c].type = DISTROBJ;
        dl[10 + NumChannels + c].ptr  = (void *)colorMap[c];
    }
    update_shared_dl(10 + NumChannels + NumColorMaps, dl);
}

void coDoGeometry::setNormals(int nattr, const coDistributedObject *n)
{
    normal_attr = nattr;
    normals = n;
    covise_data_list dl[10 + NumChannels + NumColorMaps];
    dl[0].type = INTSHM;
    dl[0].ptr  = (void *)&geometry_type;
    dl[1].type = DISTROBJ;
    dl[1].ptr  = (void *)geometry;
    dl[2].type = INTSHM;
    dl[2].ptr  = (void *)&color_attr;
    for (int c = 0; c < NumChannels; ++c)
    {
        dl[3 + c].type = DISTROBJ;
        dl[3 + c].ptr  = (void *)colors[c];
    }
    dl[3 + NumChannels].type = INTSHM;
    dl[3 + NumChannels].ptr  = (void *)&normal_attr;
    dl[4 + NumChannels].type = DISTROBJ;
    dl[4 + NumChannels].ptr  = (void *)normals;
    dl[5 + NumChannels].type = INTSHM;
    dl[5 + NumChannels].ptr  = (void *)&texture_attr;
    dl[6 + NumChannels].type = DISTROBJ;
    dl[6 + NumChannels].ptr  = (void *)texture;
    dl[7 + NumChannels].type = INTSHM;
    dl[7 + NumChannels].ptr  = (void *)&vertexAttribute_attr;
    dl[8 + NumChannels].type = DISTROBJ;
    dl[8 + NumChannels].ptr  = (void *)vertexAttribute;
    dl[9 + NumChannels].type = INTSHM;
    dl[9 + NumChannels].ptr  = (void *)&colorMap_attr;
    for (int c = 0; c < NumColorMaps; ++c)
    {
        dl[10 + NumChannels + c].type = DISTROBJ;
        dl[10 + NumChannels + c].ptr  = (void *)colorMap[c];
    }
    update_shared_dl(10 + NumChannels + NumColorMaps, dl);
}

void coDoGeometry::setTexture(int tattr, const coDistributedObject *t)
{
    texture_attr = tattr;
    texture = t;
    covise_data_list dl[10 + NumChannels + NumColorMaps];
    dl[0].type = INTSHM;
    dl[0].ptr  = (void *)&geometry_type;
    dl[1].type = DISTROBJ;
    dl[1].ptr  = (void *)geometry;
    dl[2].type = INTSHM;
    dl[2].ptr  = (void *)&color_attr;
    for (int c = 0; c < NumChannels; ++c)
    {
        dl[3 + c].type = DISTROBJ;
        dl[3 + c].ptr  = (void *)colors[c];
    }
    dl[3 + NumChannels].type = INTSHM;
    dl[3 + NumChannels].ptr  = (void *)&normal_attr;
    dl[4 + NumChannels].type = DISTROBJ;
    dl[4 + NumChannels].ptr  = (void *)normals;
    dl[5 + NumChannels].type = INTSHM;
    dl[5 + NumChannels].ptr  = (void *)&texture_attr;
    dl[6 + NumChannels].type = DISTROBJ;
    dl[6 + NumChannels].ptr  = (void *)texture;
    dl[7 + NumChannels].type = INTSHM;
    dl[7 + NumChannels].ptr  = (void *)&vertexAttribute_attr;
    dl[8 + NumChannels].type = DISTROBJ;
    dl[8 + NumChannels].ptr  = (void *)vertexAttribute;
    dl[9 + NumChannels].type = INTSHM;
    dl[9 + NumChannels].ptr  = (void *)&colorMap_attr;
    for (int c = 0; c < NumColorMaps; ++c)
    {
        dl[10 + NumChannels + c].type = DISTROBJ;
        dl[10 + NumChannels + c].ptr  = (void *)colorMap[c];
    }
    update_shared_dl(10 + NumChannels + NumColorMaps, dl);
}

void coDoGeometry::setVertexAttribute(int vattr, const coDistributedObject *v)
{
    vertexAttribute_attr = vattr;
    vertexAttribute = v;
    covise_data_list dl[10 + NumChannels + NumColorMaps];
    dl[0].type = INTSHM;
    dl[0].ptr  = (void *)&geometry_type;
    dl[1].type = DISTROBJ;
    dl[1].ptr  = (void *)geometry;
    dl[2].type = INTSHM;
    dl[2].ptr  = (void *)&color_attr;
    for (int c = 0; c < NumChannels; ++c)
    {
        dl[3 + c].type = DISTROBJ;
        dl[3 + c].ptr  = (void *)colors[c];
    }
    dl[3 + NumChannels].type = INTSHM;
    dl[3 + NumChannels].ptr  = (void *)&normal_attr;
    dl[4 + NumChannels].type = DISTROBJ;
    dl[4 + NumChannels].ptr  = (void *)normals;
    dl[5 + NumChannels].type = INTSHM;
    dl[5 + NumChannels].ptr  = (void *)&texture_attr;
    dl[6 + NumChannels].type = DISTROBJ;
    dl[6 + NumChannels].ptr  = (void *)texture;
    dl[7 + NumChannels].type = INTSHM;
    dl[7 + NumChannels].ptr  = (void *)&vertexAttribute_attr;
    dl[8 + NumChannels].type = DISTROBJ;
    dl[8 + NumChannels].ptr  = (void *)vertexAttribute;
    dl[9 + NumChannels].type = INTSHM;
    dl[9 + NumChannels].ptr  = (void *)&colorMap_attr;
    for (int c = 0; c < NumColorMaps; ++c)
    {
        dl[10 + NumChannels + c].type = DISTROBJ;
        dl[10 + NumChannels + c].ptr  = (void *)colorMap[c];
    }
    update_shared_dl(10 + NumChannels + NumColorMaps, dl);
}

void coDoGeometry::setColorMap(int cmattr, const coDistributedObject *cm, size_t chan)
{
    colorMap_attr = cmattr;
    colorMap[chan] = cm;
    covise_data_list dl[10 + NumChannels + NumColorMaps];
    dl[0].type = INTSHM;
    dl[0].ptr  = (void *)&geometry_type;
    dl[1].type = DISTROBJ;
    dl[1].ptr  = (void *)geometry;
    dl[2].type = INTSHM;
    dl[2].ptr  = (void *)&color_attr;
    for (int c = 0; c < NumChannels; ++c)
    {
        dl[3 + c].type = DISTROBJ;
        dl[3 + c].ptr  = (void *)colors[c];
    }
    dl[3 + NumChannels].type = INTSHM;
    dl[3 + NumChannels].ptr  = (void *)&normal_attr;
    dl[4 + NumChannels].type = DISTROBJ;
    dl[4 + NumChannels].ptr  = (void *)normals;
    dl[5 + NumChannels].type = INTSHM;
    dl[5 + NumChannels].ptr  = (void *)&texture_attr;
    dl[6 + NumChannels].type = DISTROBJ;
    dl[6 + NumChannels].ptr  = (void *)texture;
    dl[7 + NumChannels].type = INTSHM;
    dl[7 + NumChannels].ptr  = (void *)&vertexAttribute_attr;
    dl[8 + NumChannels].type = DISTROBJ;
    dl[8 + NumChannels].ptr  = (void *)vertexAttribute;
    dl[9 + NumChannels].type = INTSHM;
    dl[9 + NumChannels].ptr  = (void *)&colorMap_attr;
    for (int c = 0; c < NumColorMaps; ++c)
    {
        dl[10 + NumChannels + c].type = DISTROBJ;
        dl[10 + NumChannels + c].ptr  = (void *)colorMap[c];
    }
    update_shared_dl(10 + NumChannels + NumColorMaps, dl);
}
