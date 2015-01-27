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
    if (no == 10)
    {
        (*il)[0].description = "Geometry Type";
        (*il)[1].description = "Geometry";
        (*il)[2].description = "Color Attribute: not used";
        (*il)[3].description = "Colors";
        (*il)[4].description = "Normal Attribute: not used";
        (*il)[5].description = "Normals";
        (*il)[6].description = "Texture Type";
        (*il)[7].description = "Texture";
        (*il)[8].description = "Vertex Attribute Type";
        (*il)[9].description = "Vertex Attribute";
        return 10;
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
    colors = NULL;
    normals = NULL;
    texture = NULL;
    vertexAttribute = NULL;

    covise_data_list dl[]
        = {
            { INTSHM, &geometry_type },
            { DISTROBJ, geometry },
            { INTSHM, &color_attr },
            { DISTROBJ, colors },
            {
              INTSHM, &normal_attr,
            },
            { DISTROBJ, normals },
            {
              INTSHM, &texture_attr,
            },
            { DISTROBJ, texture },
            {
              INTSHM, &vertexAttribute_attr,
            },
            { DISTROBJ, vertexAttribute }
        };

    new_ok = store_shared_dl(10, dl) != 0;
    if (!new_ok)
        return;
    geometry_type = geo->get_type_no();
    color_attr = NONE;
    normal_attr = NONE;
    texture_attr = NONE;
    vertexAttribute_attr = NONE;
}

coDoGeometry *coDoGeometry::cloneObject(const coObjInfo &newinfo) const
{
    coDoGeometry *geo = new coDoGeometry(newinfo);
    geo->setGeometry(getGeometryType(), getGeometry());
    geo->setColors(getColorAttributes(), getColors());
    geo->setNormals(getNormalAttributes(), getNormals());
    geo->setTexture(getTextureAttributes(), getTexture());
    geo->setVertexAttribute(getVertexAttributeAttributes(), getVertexAttribute());
    return geo;
}

int coDoGeometry::rebuildFromShm()
{
    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }

    covise_data_list dl[]
        = {
            { INTSHM, &geometry_type },
            { UNKNOWN, &geometry },
            { INTSHM, &color_attr },
            { COVISE_OPTIONAL, &colors },
            {
              INTSHM, &normal_attr,
            },
            { COVISE_OPTIONAL, &normals },
            {
              INTSHM, &texture_attr,
            },
            { COVISE_OPTIONAL, &texture },
            {
              INTSHM, &vertexAttribute_attr,
            },
            { COVISE_OPTIONAL, &vertexAttribute }
        };

    return restore_shared_dl(10, dl);
}

void coDoGeometry::setGeometry(int gtype, const coDistributedObject *geo)
{
    geometry_type = gtype;
    geometry = geo;
    covise_data_list dl[] = {
        { INTSHM, &geometry_type },
        { DISTROBJ, geometry },
        { INTSHM, &color_attr },
        { DISTROBJ, colors },
        {
          INTSHM, &normal_attr,
        },
        { DISTROBJ, normals },
        {
          INTSHM, &texture_attr,
        },
        { DISTROBJ, texture },
        {
          INTSHM, &vertexAttribute_attr,
        },
        { DISTROBJ, vertexAttribute }
    };
    update_shared_dl(10, dl);
}

void coDoGeometry::setColors(int cattr, const coDistributedObject *c)
{
    color_attr = cattr;
    colors = c;
    covise_data_list dl[] = {
        { INTSHM, &geometry_type },
        { DISTROBJ, geometry },
        { INTSHM, &color_attr },
        { DISTROBJ, colors },
        {
          INTSHM, &normal_attr,
        },
        { DISTROBJ, normals },
        {
          INTSHM, &texture_attr,
        },
        { DISTROBJ, texture },
        {
          INTSHM, &vertexAttribute_attr,
        },
        { DISTROBJ, vertexAttribute }
    };
    update_shared_dl(10, dl);
}

void coDoGeometry::setNormals(int nattr, const coDistributedObject *n)
{
    normal_attr = nattr;
    normals = n;
    covise_data_list dl[] = {
        { INTSHM, &geometry_type },
        { DISTROBJ, geometry },
        { INTSHM, &color_attr },
        { DISTROBJ, colors },
        {
          INTSHM, &normal_attr,
        },
        { DISTROBJ, normals },
        {
          INTSHM, &texture_attr,
        },
        { DISTROBJ, texture },
        {
          INTSHM, &vertexAttribute_attr,
        },
        { DISTROBJ, vertexAttribute }
    };
    update_shared_dl(10, dl);
}

void coDoGeometry::setTexture(int tattr, const coDistributedObject *t)
{
    texture_attr = tattr;
    texture = t;
    covise_data_list dl[] = {
        { INTSHM, &geometry_type },
        { DISTROBJ, geometry },
        { INTSHM, &color_attr },
        { DISTROBJ, colors },
        {
          INTSHM, &normal_attr,
        },
        { DISTROBJ, normals },
        {
          INTSHM, &texture_attr,
        },
        { DISTROBJ, texture },
        {
          INTSHM, &vertexAttribute_attr,
        },
        { DISTROBJ, vertexAttribute }
    };
    update_shared_dl(10, dl);
}

void coDoGeometry::setVertexAttribute(int vattr, const coDistributedObject *v)
{
    vertexAttribute_attr = vattr;
    vertexAttribute = v;
    covise_data_list dl[] = {
        { INTSHM, &geometry_type },
        { DISTROBJ, geometry },
        { INTSHM, &color_attr },
        { DISTROBJ, colors },
        {
          INTSHM, &normal_attr,
        },
        { DISTROBJ, normals },
        {
          INTSHM, &texture_attr,
        },
        { DISTROBJ, texture },
        {
          INTSHM, &vertexAttribute_attr,
        },
        { DISTROBJ, vertexAttribute }
    };
    update_shared_dl(10, dl);
}
