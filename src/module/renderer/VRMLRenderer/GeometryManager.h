/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GEOMETRY_MANAGER
#define _GEOMETRY_MANAGER

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

#include <appl/RenderInterface.h>
#include "ObjectList.h"
#include <util/coMaterial.h>
#include <do/coDoTexture.h>

extern ObjectList *objlist;

class NewCharBuffer
{
    char* buf;
    int len;
    int incSize;

public:
    int cur_len;
    NewCharBuffer()
    {
        incSize = 1000;
        cur_len = 0;
        len = 0;
        buf = NULL;
    };
    NewCharBuffer(NewCharBuffer* obuf)
    {
        incSize = 1000;
        cur_len = obuf->cur_len;
        len = cur_len + 1;
        buf = new char[len];
        strcpy(buf, obuf->getbuf());
    };
    NewCharBuffer(int def)
    {
        incSize = def;
        cur_len = 0;
        len = def;
        buf = new char[len];
    };
    ~NewCharBuffer() { delete[] buf; };
    char* return_data()
    {
        char* tmp = buf;
        buf = NULL;
        cur_len = 0;
        len = 0;
        return (tmp);
    };
    int strlen() { return (cur_len); };
    void operator+=(const char* const s)
    {
        int l = (int)::strlen(s);
        if (cur_len + l >= len)
        {
            len += incSize;
            char* nbuf = new char[len];
            strcpy(nbuf, buf);
            delete[] buf;
            buf = nbuf;
        }
        strcpy(buf + cur_len, s);
        cur_len += l;
    };
    void operator+=(char c)
    {
        if (cur_len + 1 >= len)
        {
            len += incSize;
            char* nbuf = new char[len];
            strcpy(nbuf, buf);
            delete[] buf;
            buf = nbuf;
        }
        buf[cur_len] = c;
        cur_len++;
        buf[cur_len] = 0;
    };
    void operator+=(int n)
    {
        CharNum s(n);
        int l = (int)::strlen(s);
        if (cur_len + l >= len)
        {
            len += incSize;
            char* nbuf = new char[len];
            strcpy(nbuf, buf);
            delete[] buf;
            buf = nbuf;
        }
        strcpy(buf + cur_len, s);
        cur_len += l;
    };
    void operator+=(float n)
    {
        CharNum s(n);
        int l = (int)::strlen(s);
        if (cur_len + l >= len)
        {
            len += incSize;
            char* nbuf = new char[len];
            strcpy(nbuf, buf);
            delete[] buf;
            buf = nbuf;
        }
        strcpy(buf + cur_len, s);
        cur_len += l;
    };
    operator const char* () const { return (buf); };
    const char* getbuf() { return (buf); };
};
//================================================================
// GeoemtryManager
//================================================================

class GeometryManager
{
private:
    void boundingBox(float *x, float *y, float *z, int num, float *bbox);
    void boundingBox(float *x, float *y, float *z, float *r, int num, float *bbox);

public:
    GeometryManager(){};

    void addGroup(const char *object, const char *rootName, int is_timestep, int minTimeStep, int maxTimeStep);
    void addUGrid(const char *object, const char *rootName, int xsize, int ysize,
                  int zsize, float xmin, float xmax,
                  float ymin, float ymax, float zmin, float zmax,
                  int no_of_colors, int colorbinding, int colorpacking,
                  float *r, float *g, float *b, int *pc, int no_of_normals,
                  int normalbinding,
                  float *nx, float *ny, float *nz, float transparency, coMaterial *material, coDoTexture *texture);

    void addRGrid(const char *object, const char *rootName, int xsize, int ysize,
                  int zsize, float *x_c, float *y_c,
                  float *z_c,
                  int no_of_colors, int colorbinding, int colorpacking,
                  float *r, float *g, float *b, int *pc, int no_of_normals,
                  int normalbinding,
                  float *nx, float *ny, float *nz, float transparency, coMaterial *material, coDoTexture *texture);

    void addSGrid(const char *object, const char *rootName, int xsize, int ysize,
                  int zsize, float *x_c, float *y_c,
                  float *z_c,
                  int no_of_colors, int colorbinding, int colorpacking,
                  float *r, float *g, float *b, int *pc, int no_of_normals,
                  int normalbinding,
                  float *nx, float *ny, float *nz, float transparency, coMaterial *material, coDoTexture *texture);

    void addPolygon(const char *object, const char *rootName, int no_of_polygons, int no_of_vertices,
                    int no_of_coords, float *x_c, float *y_c,
                    float *z_c,
                    int *v_l, int *i_l,
                    int no_of_colors, int colorbinding, int colorpacking,
                    float *r, float *g, float *b, int *pc, int no_of_normals,
                    int normalbinding,
                    float *nx, float *ny, float *nz, float transparency,
                    int vertexOrder, coMaterial *material, coDoTexture *texture);

    void addTriangleStrip(const char *object, const char *rootName, int no_of_strips, int no_of_vertices,
                          int no_of_coords, float *x_c, float *y_c,
                          float *z_c,
                          int *v_l, int *i_l,
                          int no_of_colors, int colorbinding, int colorpacking,
                          float *r, float *g, float *b, int *pc, int no_of_normals,
                          int normalbinding,
                          float *nx, float *ny, float *nz, float transparency,
                          int vertexOrder, coMaterial *material, coDoTexture *texture);

    void addLine(const char *object, const char *rootName, int no_of_lines, int no_of_vertices,
                 int no_of_coords, float *x_c, float *y_c,
                 float *z_c,
                 int *v_l, int *i_l,
                 int no_of_colors, int colorbinding, int colorpacking,
                 float *r, float *g, float *b, int *pc, int no_of_normals,
                 int normalbinding,
                 float *nx, float *ny, float *nz, int isTrace, coMaterial *material, coDoTexture *texture);

    void addPoint(const char *object, const char *rootName, int no_of_points,
                  float *x_c, float *y_c, float *z_c,
                  int colorbinding, int colorpacking, float *r, float *g, float *b, int *pc, coMaterial *material, coDoTexture *texture);

    void addSphere(const char *object, const char *rootName, int no_of_points,
                   float *x_c, float *y_c, float *z_c, float *r_c,
                   int colorbinding, int colorpacking, float *r, float *g, float *b, int *pc, coMaterial *material, coDoTexture *texture);

    // void addIv(char *object,char *rootName, char *IvDescription, int size);

    void replaceUGrid(const char *object, int xsize, int ysize,
                      int zsize, float xmin, float xmax,
                      float ymin, float ymax, float zmin, float zmax,
                      int no_of_colors, int colorbinding, int colorpacking,
                      float *r, float *g, float *b, int *pc, int no_of_normals,
                      int normalbinding,
                      float *nx, float *ny, float *nz, float transparency);

    void replaceRGrid(const char *object, int xsize, int ysize,
                      int zsize, float *x_c, float *y_c,
                      float *z_c,
                      int no_of_colors, int colorbinding, int colorpacking,
                      float *r, float *g, float *b, int *pc, int no_of_normals,
                      int normalbinding,
                      float *nx, float *ny, float *nz, float transparency);

    void replaceSGrid(const char *object, int xsize, int ysize,
                      int zsize, float *x_c, float *y_c,
                      float *z_c,
                      int no_of_colors, int colorbinding, int colorpacking,
                      float *r, float *g, float *b, int *pc, int no_of_normals,
                      int normalbinding,
                      float *nx, float *ny, float *nz, float transparency);

    void replacePolygon(const char *object, int no_of_polygons, int no_of_vertices,
                        int no_of_coords, float *x_c, float *y_c,
                        float *z_c,
                        int *v_l, int *i_l,
                        int no_of_colors, int colorbinding, int colorpacking,
                        float *r, float *g, float *b, int *pc, int no_of_normals,
                        int normalbinding,
                        float *nx, float *ny, float *nz, float transparency,
                        int vertexOrder);

    void replaceTriangleStrip(const char *object, int no_of_strips, int no_of_vertices,
                              int no_of_coords, float *x_c, float *y_c,
                              float *z_c,
                              int *v_l, int *i_l,
                              int no_of_colors, int colorbinding, int colorpacking,
                              float *r, float *g, float *b, int *pc, int no_of_normals,
                              int normalbinding,
                              float *nx, float *ny, float *nz, float transparency,
                              int vertexOrder);

    void replaceLine(const char *object, int no_of_lines, int no_of_vertices,
                     int no_of_coords, float *x_c, float *y_c,
                     float *z_c,
                     int *v_l, int *i_l,
                     int no_of_colors, int colorbinding, int colorpacking,
                     float *r, float *g, float *b, int *pc, int no_of_normals,
                     int normalbinding,
                     float *nx, float *ny, float *nz);

    void replacePoint(const char *object, int no_of_points,
                      float *x_c, float *y_c, float *z_c,
                      int colorbinding, int colorpacking, float *r, float *g,
                      float *b, int *pc);

    void replaceSphere(const char *object, int no_of_points,
                       float *x_c, float *y_c, float *z_c, float *r_c,
                       int colorbinding, int colorpacking, float *r, float *g,
                       float *b, int *pc);
    // void replaceIv(char *object, char *IvDescription);

    void remove_geometry(const char *name);

    void addMaterial(coMaterial* material, int colorbinding, int colorpacking, float* r, float* g, float* b, int* pc, coDoTexture *, NewCharBuffer& buf, const char* object);

    ~GeometryManager()
    {
    }
};

#endif
