/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: geometry manager class for COVISE renderer modules        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Dirk Rantzau                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  11.09.95  V1.0                                                  **
\**************************************************************************/

#include "GeometryManager.h"
//#include <Covise_Util.h>
#include "ObjectManager.h"
#include "ObjectList.h"

#include <float.h>


//================================================================
// GeometryManager methods
//================================================================

void GeometryManager::boundingBox(float *x, float *y, float *z, int num, float *bbox)
{

    for (int i = 0; i < num; i++)
    {
        if (x[i] < bbox[0])
            bbox[0] = x[i];
        else if (x[i] > bbox[3])
            bbox[3] = x[i];

        if (y[i] < bbox[1])
            bbox[1] = y[i];
        else if (y[i] > bbox[4])
            bbox[4] = y[i];

        if (z[i] < bbox[2])
            bbox[2] = z[i];
        else if (z[i] > bbox[5])
            bbox[5] = z[i];
    }
}

void GeometryManager::boundingBox(float *x, float *y, float *z, float *r, int num, float *bbox)
{

    for (int i = 0; i < num; i++)
    {
        if (x[i] - r[i] < bbox[0])
            bbox[0] = x[i] - r[i];
        else if (x[i] + r[i] > bbox[3])
            bbox[3] = x[i] + r[i];

        if (y[i] - r[i] < bbox[1])
            bbox[1] = y[i] - r[i];
        else if (y[i] + r[i] > bbox[4])
            bbox[4] = y[i] + r[i];

        if (z[i] - r[i] < bbox[2])
            bbox[2] = z[i] - r[i];
        else if (z[i] + r[i] > bbox[5])
            bbox[5] = z[i] + r[i];
    }
}

void unpackRGBA(int *pc, int pos, float *r, float *g, float *b, float *a);

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::addGroup(const char *object, const char *rootName, int is_timestep, int minTimeStep, int maxTimeStep)
{
    (void)object;
    (void)rootName;
    (void)is_timestep;
    (void)minTimeStep;
    (void)maxTimeStep;

    CoviseRender::sendInfo("Adding a new set hierarchy");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::addUGrid(const char *object, const char *rootName, int xsize, int ysize,
                               int zsize, float xmin, float xmax,
                               float ymin, float ymax, float zmin, float zmax,
                               int no_of_colors, int colorbinding, int colorpacking,
                               float *r, float *g, float *b, int *pc, int no_of_normals,
                               int normalbinding,
                               float *nx, float *ny, float *nz, float transparency, coMaterial *material)

{
    (void)object;
    (void)rootName;
    (void)xsize;
    (void)ysize;
    (void)zsize;
    (void)xmin;
    (void)xmax;
    (void)ymin;
    (void)ymax;
    (void)zmin;
    (void)zmax;
    (void)no_of_colors;
    (void)colorbinding;
    (void)colorpacking;
    (void)r;
    (void)g;
    (void)b;
    (void)pc;
    (void)no_of_normals;
    (void)normalbinding;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)transparency;
    (void)material;

    CoviseRender::sendInfo("Adding a uniform grid");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::addRGrid(const char *object, const char *rootName, int xsize, int ysize,
                               int zsize, float *x_c, float *y_c,
                               float *z_c,
                               int no_of_colors, int colorbinding, int colorpacking,
                               float *r, float *g, float *b, int *pc, int no_of_normals,
                               int normalbinding,
                               float *nx, float *ny, float *nz, float transparency, coMaterial *material)

{
    (void)object;
    (void)rootName;
    (void)xsize;
    (void)ysize;
    (void)zsize;
    (void)x_c;
    (void)y_c;
    (void)z_c;
    (void)no_of_colors;
    (void)colorbinding;
    (void)colorpacking;
    (void)r;
    (void)g;
    (void)b;
    (void)pc;
    (void)no_of_normals;
    (void)normalbinding;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)transparency;
    (void)material;

    CoviseRender::sendInfo("Adding a rectilinear grid");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::addSGrid(const char *object, const char *rootName, int xsize, int ysize,
                               int zsize, float *x_c, float *y_c,
                               float *z_c,
                               int no_of_colors, int colorbinding, int colorpacking,
                               float *r, float *g, float *b, int *pc, int no_of_normals,
                               int normalbinding,
                               float *nx, float *ny, float *nz, float transparency, coMaterial *material)
{
    (void)object;
    (void)rootName;
    (void)xsize;
    (void)ysize;
    (void)zsize;
    (void)x_c;
    (void)y_c;
    (void)z_c;
    (void)no_of_colors;
    (void)colorbinding;
    (void)colorpacking;
    (void)r;
    (void)g;
    (void)b;
    (void)pc;
    (void)no_of_normals;
    (void)normalbinding;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)transparency;
    (void)material;

    CoviseRender::sendInfo("Adding a structured grid");
}


void GeometryManager::addMaterial(coMaterial* material, int colorbinding, int colorpacking, float* r, float* g, float* b, int* pc, NewCharBuffer& buf)
{
    char line[500];
    if (material)
    {
        if (objlist->outputMode == OutputMode::VRML97)
        {
            buf += "Shape{  appearance Appearance { material Material {diffuseColor ";
            sprintf(line, " %1g %1g %1g ", material->diffuseColor[0], material->diffuseColor[1], material->diffuseColor[2]);
            buf += line;
            buf += "emissiveColor ";
            sprintf(line, " %1g %1g %1g ", material->emissiveColor[0], material->emissiveColor[1], material->emissiveColor[2]);
            buf += line;
            buf += "ambientIntensity ";
            sprintf(line, " %1g ", (material->ambientColor[0] + material->ambientColor[1] + material->ambientColor[2]) / 3.0);
            buf += line;
            buf += "specularColor ";
            sprintf(line, " %1g %1g %1g ", material->specularColor[0], material->specularColor[1], material->specularColor[2]);
            buf += line;
            buf += "transparency ";
            sprintf(line, " %1g ", material->transparency);
            buf += line;
            buf += "shininess ";
            sprintf(line, " %1g ", material->shininess);
            buf += line;
            buf += "}}";
        }
        else
        {
            buf += "<shape>\n<appearance>\n<material diffuseColor='";
            sprintf(line, "%1g %1g %1g'", material->diffuseColor[0], material->diffuseColor[1], material->diffuseColor[2]);
            buf += line;
            buf += " emissiveColor='";
            sprintf(line, "%1g %1g %1g'", material->emissiveColor[0], material->emissiveColor[1], material->emissiveColor[2]);
            buf += line;
            buf += " ambientIntensity='";
            sprintf(line, "%1g'", (material->ambientColor[0] + material->ambientColor[1] + material->ambientColor[2]) / 3.0);
            buf += line;
            buf += " specularColor='";
            sprintf(line, "%1g %1g %1g'", material->specularColor[0], material->specularColor[1], material->specularColor[2]);
            buf += line;
            buf += " transparency='";
            sprintf(line, "%1g'", material->transparency);
            buf += line;
            buf += " shininess='";
            sprintf(line, "%1g'", material->shininess);
            buf += line;
            buf += "></material>\n</appearance>";
        }
    }
    else if (colorbinding == CO_PER_VERTEX)
    {
        if (objlist->outputMode == OutputMode::VRML97)
        {
            buf += "Shape{ appearance Appearance { material Material { }}\n";
        }
        else
        {
            buf += "<shape>  <appearance> <material> </material> </appearance> ";
        }
    }
    else
    {
        if (objlist->outputMode == OutputMode::VRML97)
        {
            buf += "Shape{  appearance Appearance { material Material {diffuseColor ";
            if (colorpacking == CO_RGBA && pc)
            {
                float r, g, b, a;

                unpackRGBA(pc, 0, &r, &g, &b, &a);
                sprintf(line, " %1g %1g %1g ", r, g, b);
            }
            else if (colorbinding != CO_NONE && r)
            {
                sprintf(line, " %1g %1g %1g ", r[0], g[0], b[0]);
            }
            else
            {
                sprintf(line, " 1 1 1 ");
            }
            buf += line;
            buf += "}}";
        }
        else
        {
            buf += "<shape>  <appearance> <material diffuseColor='";
            if (colorpacking == CO_RGBA && pc)
            {
                float r, g, b, a;

                unpackRGBA(pc, 0, &r, &g, &b, &a);
                sprintf(line, "%1g %1g %1g'", r, g, b);
            }
            else if (colorbinding != CO_NONE && r)
            {
                sprintf(line, "%1g %1g %1g'", r[0], g[0], b[0]);
            }
            else
            {
                sprintf(line, "1 1 1'");
            }
            buf += line;
            buf += "></material> </appearance>";
        }
    }

}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::addPolygon(const char *object, const char *rootName, int no_of_polygons, int no_of_vertices,
                                 int no_of_coords, float *x_c, float *y_c,
                                 float *z_c,
                                 int *v_l, int *l_l,
                                 int no_of_colors, int colorbinding, int colorpacking,
                                 float *r, float *g, float *b, int *pc, int no_of_normals,
                                 int normalbinding,
                                 float *nx, float *ny, float *nz, float transparency,
                                 int vertexOrder, coMaterial *material)
{
    (void)vertexOrder;
    (void)transparency;
    float bbox[6] = { FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };

    NewCharBuffer buf(10000);
    char line[500];
    int i;
    int n = 0;

    addMaterial(material, colorbinding, colorpacking,r,g,b,pc, buf);

    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "geometry IndexedFaceSet {\n coord Coordinate{\npoint[";
    }
    else
    {
        buf += "<indexedFaceSet\n";
    }


    if (objlist->outputMode == OutputMode::VRML97)
    {
        for (i = 0; i < no_of_coords; i++)
        {
            sprintf(line, "%1g %1g %1g,", x_c[i], y_c[i], z_c[i]);
            buf += line;
            if ((i % 10) == 0)
                buf += '\n';
        }
        buf += "]}\ncoordIndex [";
        n = 0;
        for (i = 0; i < no_of_vertices; i++)
        {
            if ((n < (no_of_polygons - 1)) && (l_l[n + 1] == i))
            {
                buf += "-1 ";
                n++;
            }
            sprintf(line, "%d ", v_l[i]);
            buf += line;
            if ((i % 10) == 0)
                buf += '\n';
        }
        buf += ']';
        if (normalbinding == CO_PER_VERTEX)
        {
            buf += "\n normal Normal {\nvector[";
            for (i = 0; i < no_of_normals; i++)
            {
                sprintf(line, "%1g %1g %1g,", nx[i], ny[i], nz[i]);
                buf += line;
                if ((i % 10) == 0)
                    buf += '\n';
            }
            buf += "]}\nnormalPerVertex TRUE\n";
        }
        if (colorbinding == CO_PER_VERTEX)
        {
            buf += "\n color Color {\ncolor[";
            if (colorpacking == CO_RGBA && pc)
            {
                float r, g, b, a;

                for (i = 0; i < no_of_colors; i++)
                {
                    unpackRGBA(pc, i, &r, &g, &b, &a);
                    sprintf(line, "%1g %1g %1g,", r, g, b);
                    buf += line;
                    if ((i % 10) == 0)
                        buf += '\n';
                }
            }
            else
            {
                for (i = 0; i < no_of_colors; i++)
                {
                    sprintf(line, "%1g %1g %1g,", r[i], g[i], b[i]);
                    buf += line;
                    if ((i % 10) == 0)
                        buf += '\n';
                }
            }
            buf += "]}\ncolorPerVertex TRUE\n";
        }
        buf += "solid FALSE\nccw TRUE\nconvex TRUE}}\n";
    }
    else
    {
        buf += "coordIndex='";
        n = 0;
        for (i = 0; i < no_of_vertices; i++)
        {
            if ((n < (no_of_polygons - 1)) && (l_l[n + 1] == i))
            {
                buf += "-1 ";
                n++;
            }
            sprintf(line, "%d ", v_l[i]);
            buf += line;
        }
        buf += "'\n";

        buf += "solid='false'\nccw='true'\nconvex='true'\n";
        if (colorbinding == CO_PER_VERTEX)
        {
            buf += "colorPerVertex = 'true'\n";
        }

        buf += ">\n";

        buf += "<coordinate point = '";
        for (i = 0; i < no_of_coords; i++)
        {
            sprintf(line, "%1g %1g %1g ", x_c[i], y_c[i], z_c[i]);
            buf += line;
        }
        buf += "'>\n</coordinate>\n ";
        if (normalbinding == CO_PER_VERTEX)
        {
            buf += "\n <Normal vector=";
            for (i = 0; i < no_of_normals; i++)
            {
                sprintf(line, "%1g %1g %1g ", nx[i], ny[i], nz[i]);
                buf += line;
            }
            buf += "'\n";
            buf += "><Normal>\nnormalPerVertex='true'\n";
        }
        if (colorbinding == CO_PER_VERTEX)
        {
            buf += "\n <Color color='";
            if (colorpacking == CO_RGBA && pc)
            {
                float r, g, b, a;

                for (i = 0; i < no_of_colors; i++)
                {
                    unpackRGBA(pc, i, &r, &g, &b, &a);
                    sprintf(line, "%1g %1g %1g ", r, g, b);
                    buf += line;
                }
            }
            else
            {
                for (i = 0; i < no_of_colors; i++)
                {
                    sprintf(line, "%1g %1g %1g,", r[i], g[i], b[i]);
                    buf += line;
                    if ((i % 10) == 0)
                        buf += '\n';
                }
            }
            buf += "'>\n";
            buf += "</Color>\n";
        }
        buf += "</indexedFaceSet> </shape>\n";
    }

    boundingBox(x_c, y_c, z_c, no_of_coords, bbox);

    NewCharBuffer *newbuf = new NewCharBuffer(&buf);
    LObject *lob = new LObject();
    lob->set_boundingbox(bbox);
    lob->set_name((char *)object);
    lob->set_timestep(object);
    lob->set_rootname((char *)rootName);
    lob->set_objPtr((void *)newbuf);
    objlist->push_back(std::unique_ptr<LObject>(lob));
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::addTriangleStrip(const char *object, const char *rootName, int no_of_strips, int no_of_vertices,
                                       int no_of_coords, float *x_c, float *y_c,
                                       float *z_c,
                                       int *v_l, int *s_l,
                                       int no_of_colors, int colorbinding, int colorpacking,
                                       float *r, float *g, float *b, int *pc, int no_of_normals,
                                       int normalbinding,
                                       float *nx, float *ny, float *nz, float transparency,
                                       int vertexOrder, coMaterial *material)
{

    (void)transparency;
    (void)vertexOrder;
    float bbox[6] = { FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };

    NewCharBuffer buf(10000);
    char line[500];
    int i;
    int n = 0;


    addMaterial(material, colorbinding, colorpacking, r, g, b, pc, buf);




    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "geometry IndexedFaceSet {\n coord Coordinate{\npoint[";
    }
    else
    {
        buf += "<indexedFaceSet\n";
    }


    if (objlist->outputMode == OutputMode::VRML97)
    {
        for (i = 0; i < no_of_coords; i++)
        {
            sprintf(line, "%1g %1g %1g,", x_c[i], y_c[i], z_c[i]);
            buf += line;
            if ((i % 10) == 0)
                buf += '\n';
        }
        buf += "]}\ncoordIndex [";
        n = 0;
        for (i = 0; i < no_of_strips; i++)
        {

            if (i == (no_of_strips - 1))
            {
                for (n = 2; n < (no_of_vertices - s_l[i]); n++)
                {
                    if (n % 2)
                        sprintf(line, "%d %d %d -1 ", v_l[s_l[i] + n - 1], v_l[s_l[i] + n - 2], v_l[s_l[i] + n]);
                    else
                        sprintf(line, "%d %d %d -1 ", v_l[s_l[i] + n - 2], v_l[s_l[i] + n - 1], v_l[s_l[i] + n]);
                    buf += line;
                }
            }
            else
            {
                for (n = 2; n < (s_l[i + 1] - s_l[i]); n++)
                {
                    if (n % 2)
                        sprintf(line, "%d %d %d -1 ", v_l[s_l[i] + n - 1], v_l[s_l[i] + n - 2], v_l[s_l[i] + n]);
                    else
                        sprintf(line, "%d %d %d -1 ", v_l[s_l[i] + n - 2], v_l[s_l[i] + n - 1], v_l[s_l[i] + n]);
                    buf += line;
                }
            }
            if ((i % 5) == 0)
                buf += '\n';
        }
        buf += ']';
        if (normalbinding == CO_PER_VERTEX)
        {
            buf += "\n normal Normal {\nvector[";
            for (i = 0; i < no_of_normals; i++)
            {
                sprintf(line, "%1g %1g %1g,", nx[i], ny[i], nz[i]);
                buf += line;
                if ((i % 10) == 0)
                    buf += '\n';
            }
            buf += "]}\nnormalPerVertex TRUE\n";
        }
        if (colorbinding == CO_PER_VERTEX)
        {
            buf += "\n color Color {\ncolor[";
            if (colorpacking == CO_RGBA && pc)
            {
                float r, g, b, a;

                for (i = 0; i < no_of_colors; i++)
                {
                    unpackRGBA(pc, i, &r, &g, &b, &a);
                    sprintf(line, "%1g %1g %1g,", r, g, b);
                    buf += line;
                    if ((i % 10) == 0)
                        buf += '\n';
                }
            }
            else
            {
                for (i = 0; i < no_of_colors; i++)
                {
                    sprintf(line, "%1g %1g %1g,", r[i], g[i], b[i]);
                    buf += line;
                    if ((i % 10) == 0)
                        buf += '\n';
                }
            }
            buf += "]}\ncolorPerVertex TRUE\n";
        }
        buf += "solid FALSE\nccw TRUE\nconvex TRUE}}\n";
    }
    else
    {
        buf += "coordIndex='";
        n = 0;
        for (i = 0; i < no_of_strips; i++)
        {

            if (i == (no_of_strips - 1))
            {
                for (n = 2; n < (no_of_vertices - s_l[i]); n++)
                {
                    if (n % 2)
                        sprintf(line, "%d %d %d -1 ", v_l[s_l[i] + n - 1], v_l[s_l[i] + n - 2], v_l[s_l[i] + n]);
                    else
                        sprintf(line, "%d %d %d -1 ", v_l[s_l[i] + n - 2], v_l[s_l[i] + n - 1], v_l[s_l[i] + n]);
                    buf += line;
                }
            }
            else
            {
                for (n = 2; n < (s_l[i + 1] - s_l[i]); n++)
                {
                    if (n % 2)
                        sprintf(line, "%d %d %d -1 ", v_l[s_l[i] + n - 1], v_l[s_l[i] + n - 2], v_l[s_l[i] + n]);
                    else
                        sprintf(line, "%d %d %d -1 ", v_l[s_l[i] + n - 2], v_l[s_l[i] + n - 1], v_l[s_l[i] + n]);
                    buf += line;
                }
            }
        }
        buf += "'\n";

        buf += "solid='false'\nccw='true'\nconvex='true'\n";
        if (colorbinding == CO_PER_VERTEX)
        {
            buf += "colorPerVertex = 'true'\n";
        }

        buf += ">\n";

        buf += "<coordinate point = '";
        for (i = 0; i < no_of_coords; i++)
        {
            sprintf(line, "%1g %1g %1g ", x_c[i], y_c[i], z_c[i]);
            buf += line;
        }
        buf += "'>\n</coordinate>\n ";
        if (normalbinding == CO_PER_VERTEX)
        {
            buf += "\n <Normal vector=";
            for (i = 0; i < no_of_normals; i++)
            {
                sprintf(line, "%1g %1g %1g ", nx[i], ny[i], nz[i]);
                buf += line;
            }
            buf += "'\n";
            buf += "><Normal>\nnormalPerVertex='true'\n";
        }
        if (colorbinding == CO_PER_VERTEX)
        {
            buf += "\n <Color color='";
            if (colorpacking == CO_RGBA && pc)
            {
                float r, g, b, a;

                for (i = 0; i < no_of_colors; i++)
                {
                    unpackRGBA(pc, i, &r, &g, &b, &a);
                    sprintf(line, "%1g %1g %1g ", r, g, b);
                    buf += line;
                }
            }
            else
            {
                for (i = 0; i < no_of_colors; i++)
                {
                    sprintf(line, "%1g %1g %1g,", r[i], g[i], b[i]);
                    buf += line;
                    if ((i % 10) == 0)
                        buf += '\n';
                }
            }
            buf += "'>\n";
            buf += "</Color>\n";
        }
        buf += "</indexedFaceSet> </shape>\n";
    }
    boundingBox(x_c, y_c, z_c, no_of_coords, bbox);

    NewCharBuffer *newbuf = new NewCharBuffer(&buf);
    LObject *lob = new LObject(object, rootName, newbuf);
    lob->set_boundingbox(bbox);
    lob->set_timestep(object);

    objlist->push_back(std::unique_ptr<LObject>(lob));
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::addLine(const char *object, const char *rootName, int no_of_lines, int no_of_vertices,
                              int no_of_coords, float *x_c, float *y_c,
                              float *z_c,
                              int *v_l, int *l_l,
                              int no_of_colors, int colorbinding, int colorpacking,
                              float *r, float *g, float *b, int *pc, int no_of_normals,
                              int normalbinding,
                              float *nx, float *ny, float *nz, int isTrace, coMaterial *material)

{
    (void)no_of_normals;
    (void)normalbinding;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)isTrace;

    float bbox[6] = { FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };

    NewCharBuffer buf(10000);
    char line[500];
    int i;
    int n = 0;


    addMaterial(material, colorbinding, colorpacking, r, g, b, pc, buf);

    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "geometry IndexedLineSet {\n coord Coordinate{\npoint[";
    }
    else
    {
        buf += "<indexedLineSet lit='false' solid='false'\n";
    }


    if (objlist->outputMode == OutputMode::VRML97)
    {
        for (i = 0; i < no_of_coords; i++)
        {
            sprintf(line, "%1g %1g %1g,", x_c[i], y_c[i], z_c[i]);
            buf += line;
            if ((i % 10) == 0)
                buf += '\n';
        }
        buf += "]}\ncoordIndex [";
        n = 0;
        for (i = 0; i < no_of_vertices; i++)
        {
            if ((n < (no_of_lines - 1)) && (l_l[n + 1] == i))
            {
                buf += "-1 ";
                n++;
            }
            sprintf(line, "%d ", v_l[i]);
            buf += line;
            if ((i % 10) == 0)
                buf += '\n';
        }
        buf += ']';
        if (normalbinding == CO_PER_VERTEX)
        {
            buf += "\n normal Normal {\nvector[";
            for (i = 0; i < no_of_normals; i++)
            {
                sprintf(line, "%1g %1g %1g,", nx[i], ny[i], nz[i]);
                buf += line;
                if ((i % 10) == 0)
                    buf += '\n';
            }
            buf += "]}\nnormalPerVertex TRUE\n";
        }
        if (colorbinding == CO_PER_VERTEX)
        {
            buf += "\n color Color {\ncolor[";
            if (colorpacking == CO_RGBA && pc)
            {
                float r, g, b, a;

                for (i = 0; i < no_of_colors; i++)
                {
                    unpackRGBA(pc, i, &r, &g, &b, &a);
                    sprintf(line, "%1g %1g %1g,", r, g, b);
                    buf += line;
                    if ((i % 10) == 0)
                        buf += '\n';
                }
            }
            else
            {
                for (i = 0; i < no_of_colors; i++)
                {
                    sprintf(line, "%1g %1g %1g,", r[i], g[i], b[i]);
                    buf += line;
                    if ((i % 10) == 0)
                        buf += '\n';
                }
            }
            buf += "]}\ncolorPerVertex TRUE\n";
        }
        buf += "solid FALSE\n\nccw TRUE\nconvex TRUE}}\n";
    }
    else
    {
        buf += "coordIndex='";
        n = 0;
        for (i = 0; i < no_of_vertices; i++)
        {
            if ((n < (no_of_lines - 1)) && (l_l[n + 1] == i))
            {
                buf += "-1 ";
                n++;
            }
            sprintf(line, "%d ", v_l[i]);
            buf += line;
        }
        buf += "'\n";

        if (colorbinding == CO_PER_VERTEX)
        {
            buf += "colorPerVertex = 'true'\n";
        }

        buf += ">\n";

        buf += "<coordinate point = '";
        for (i = 0; i < no_of_coords; i++)
        {
            sprintf(line, "%1g %1g %1g ", x_c[i], y_c[i], z_c[i]);
            buf += line;
        }
        buf += "'>\n</coordinate>\n ";
        if (normalbinding == CO_PER_VERTEX)
        {
            buf += "\n <Normal vector=";
            for (i = 0; i < no_of_normals; i++)
            {
                sprintf(line, "%1g %1g %1g ", nx[i], ny[i], nz[i]);
                buf += line;
            }
            buf += "'\n";
            buf += "><Normal>\nnormalPerVertex='true'\n";
        }
        if (colorbinding == CO_PER_VERTEX)
        {
            buf += "\n <Color color='";
            if (colorpacking == CO_RGBA && pc)
            {
                float r, g, b, a;

                for (i = 0; i < no_of_colors; i++)
                {
                    unpackRGBA(pc, i, &r, &g, &b, &a);
                    sprintf(line, "%1g %1g %1g ", r, g, b);
                    buf += line;
                }
            }
            else
            {
                for (i = 0; i < no_of_colors; i++)
                {
                    sprintf(line, "%1g %1g %1g,", r[i], g[i], b[i]);
                    buf += line;
                    if ((i % 10) == 0)
                        buf += '\n';
                }
            }
            buf += "'>\n";
            buf += "</Color>\n";
        }
        buf += "</indexedLineSet> </shape>\n";
    }


    NewCharBuffer *newbuf = new NewCharBuffer(&buf);
    boundingBox(x_c, y_c, z_c, no_of_coords, bbox);
    LObject *lob = new LObject(object, rootName, newbuf);
    lob->set_boundingbox(bbox);
    lob->set_timestep(object);
    objlist->push_back(std::unique_ptr<LObject>(lob));
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::addPoint(const char *object, const char *rootName, int no_of_points,
                               float *x_c, float *y_c, float *z_c,
                               int colorbinding, int colorpacking,
                               float *r, float *g, float *b, int *pc, coMaterial *material)

{
    NewCharBuffer buf(10000);
    char line[500];
    int i;

    float bbox[6] = { FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };

    addMaterial(material, colorbinding, colorpacking, r, g, b, pc, buf);

    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "geometry PointSet {\n coord Coordinate{\npoint[";
    }
    else
    {
        buf += "<pointSet\n";
    }


    if (objlist->outputMode == OutputMode::VRML97)
    {
        for (i = 0; i < no_of_points; i++)
        {
            sprintf(line, "%1g %1g %1g,", x_c[i], y_c[i], z_c[i]);
            buf += line;
            if ((i % 10) == 0)
                buf += '\n';
        }
        buf += "]}\n";
        if (colorbinding == CO_PER_VERTEX)
        {
            buf += "\n color Color {\ncolor[";
            if (colorpacking == CO_RGBA && pc)
            {
                float r, g, b, a;

                for (i = 0; i < no_of_points; i++)
                {
                    unpackRGBA(pc, i, &r, &g, &b, &a);
                    sprintf(line, "%1g %1g %1g,", r, g, b);
                    buf += line;
                    if ((i % 10) == 0)
                        buf += '\n';
                }
            }
            else
            {
                for (i = 0; i < no_of_points; i++)
                {
                    sprintf(line, "%1g %1g %1g,", r[i], g[i], b[i]);
                    buf += line;
                    if ((i % 10) == 0)
                        buf += '\n';
                }
            }
            buf += "]}\ncolorPerVertex TRUE\n";
        }
        buf += "}}\n";
    }
    else
    {
        
        buf += "\n";
        if (colorbinding == CO_PER_VERTEX)
        {
            buf += "colorPerVertex = 'true'\n";
        }

        buf += ">\n";

        buf += "<coordinate point = '";
        for (i = 0; i < no_of_points; i++)
        {
            sprintf(line, "%1g %1g %1g ", x_c[i], y_c[i], z_c[i]);
            buf += line;
        }
        buf += "'>\n</coordinate>\n ";
        if (colorbinding == CO_PER_VERTEX)
        {
            buf += "\n <Color color='";
            if (colorpacking == CO_RGBA && pc)
            {
                float r, g, b, a;

                for (i = 0; i < no_of_points; i++)
                {
                    unpackRGBA(pc, i, &r, &g, &b, &a);
                    sprintf(line, "%1g %1g %1g ", r, g, b);
                    buf += line;
                }
            }
            else
            {
                for (i = 0; i < no_of_points; i++)
                {
                    sprintf(line, "%1g %1g %1g,", r[i], g[i], b[i]);
                    buf += line;
                    if ((i % 10) == 0)
                        buf += '\n';
                }
            }
            buf += "'>\n";
            buf += "</Color>\n";
        }
        buf += "</pointSet> </shape>\n";
    }


    boundingBox(x_c, y_c, z_c, no_of_points, bbox);
    NewCharBuffer *newbuf = new NewCharBuffer(&buf);
    LObject *lob = new LObject(object, rootName, newbuf);
    lob->set_boundingbox(bbox);
    lob->set_timestep(object);

    objlist->push_back(std::unique_ptr<LObject>(lob));
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::addSphere(const char *object, const char *rootName, int no_of_points,
                                float *x_c, float *y_c, float *z_c, float *r_c,
                                int colorbinding, int colorpacking,
                                float *r, float *g, float *b, int *pc, coMaterial *material)

{
    NewCharBuffer buf(10000);
    char line[500];
    float bbox[6] = { FLT_MAX, FLT_MAX, FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX };

    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "Group { children [\n";
    }
    else
    {
        buf += "<transform>\n";
    }
    for (int i = 0; i < no_of_points; ++i)
    {
        if (objlist->outputMode == OutputMode::VRML97)
        {
            buf += "Transform { translation ";
            sprintf(line, "%g %g %g", x_c[i], y_c[i], z_c[i]);
            buf += line;
            buf += " children [";
        }
        else
        {
            buf += "<transform translation='\n";
            sprintf(line, "%g %g %g '>", x_c[i], y_c[i], z_c[i]);
            buf += line;
        }



        addMaterial(material, colorbinding, colorpacking, r, g, b, pc, buf);

        if (objlist->outputMode == OutputMode::VRML97)
        {
            buf += " geometry Sphere { radius ";
            sprintf(line, "%g", r_c[i]);
            buf += line;
            buf += " }";
            buf += "} ] }\n"; // Shape + Transform
        }
        else
        {
            buf += "<sphere radius='";
            sprintf(line, "%g'>", r_c[i]);
            buf += line;
            buf += "</shape> </transform>";
        }


    }
    if (objlist->outputMode == OutputMode::VRML97)
    {
        buf += "] }\n";
    }
    else
    {
        buf += "</transform>";
    }

    boundingBox(x_c, y_c, z_c, r_c, no_of_points, bbox);
    NewCharBuffer *newbuf = new NewCharBuffer(&buf);
    LObject *lob = new LObject(object, rootName, newbuf);
    lob->set_boundingbox(bbox);
    lob->set_timestep(object);

    objlist->push_back(std::unique_ptr<LObject>(lob));
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::replaceUGrid(const char *, int, int,
                                   int, float, float,
                                   float, float, float, float,
                                   int, int, int,
                                   float *, float *, float *, int *, int,
                                   int,
                                   float *, float *, float *, float)

{

    CoviseRender::sendInfo("Replacing uniform grid");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::replaceRGrid(const char *, int, int,
                                   int, float *, float *,
                                   float *,
                                   int, int, int,
                                   float *, float *, float *, int *, int,
                                   int,
                                   float *, float *, float *, float)

{

    CoviseRender::sendInfo("Replacing rectilinear grid");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------

void GeometryManager::replaceSGrid(const char *, int, int,
                                   int, float *, float *,
                                   float *,
                                   int, int, int,
                                   float *, float *, float *, int *, int,
                                   int,
                                   float *, float *, float *, float)
{

    CoviseRender::sendInfo("Replacing structured grid");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::replacePolygon(const char *, int, int,
                                     int, float *, float *,
                                     float *,
                                     int *, int *,
                                     int, int, int,
                                     float *, float *, float *, int *, int,
                                     int,
                                     float *, float *, float *, float,
                                     int)
{

    CoviseRender::sendInfo("Replacing polygons");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::replaceTriangleStrip(const char *, int, int,
                                           int, float *, float *,
                                           float *,
                                           int *, int *,
                                           int, int, int,
                                           float *, float *, float *, int *, int,
                                           int,
                                           float *, float *, float *, float,
                                           int)
{

    CoviseRender::sendInfo("Replacing triangle strips");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::replaceLine(const char *, int, int,
                                  int, float *, float *,
                                  float *,
                                  int *, int *,
                                  int, int, int,
                                  float *, float *, float *, int *, int,
                                  int,
                                  float *, float *, float *)

{

    CoviseRender::sendInfo("Replacing lines");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::replacePoint(const char *, int,
                                   float *, float *, float *,
                                   int, int,
                                   float *, float *, float *, int *)

{

    CoviseRender::sendInfo("Replacing points");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::replaceSphere(const char *, int,
                                    float *, float *, float *, float *,
                                    int, int,
                                    float *, float *, float *, int *)

{

    CoviseRender::sendInfo("Replacing spheres");
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void GeometryManager::remove_geometry(const char *name)
{
    objlist->removeone(name);
}

void unpackRGBA(int *pc, int pos, float *r, float *g, float *b, float *a)
{

    unsigned char *chptr;

    chptr = (unsigned char *)&pc[pos];
// RGBA switched 12.03.96 due to color bug in Inventor Renderer
// D. Rantzau

#ifdef BYTESWAP
    *a = ((float)(*chptr)) / 255.0f;
    chptr++;
    *b = ((float)(*chptr)) / 255.0f;
    chptr++;
    *g = ((float)(*chptr)) / 255.0f;
    chptr++;
    *r = ((float)(*chptr)) / 255.0f;
#else
    *r = ((float)(*chptr)) / 255.0;
    chptr++;
    *g = ((float)(*chptr)) / 255.0;
    chptr++;
    *b = ((float)(*chptr)) / 255.0;
    chptr++;
    *a = ((float)(*chptr)) / 255.0;
#endif
}
