/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_OBJECT_MANAGER_EXTERN_H
#define _INV_OBJECT_MANAGER_EXTERN_H

#include <covise/covise.h>

void om_timestepCB(void *userData, void *callbackData);
void om_addObjectCB(const char *name, SoGroup *ptr);
void om_deleteObjectCB(const char *name);
void om_deleteColormapCB(const char *name);
void om_deletePartCB(const char *name);
void om_deleteTimePartCB(const char *name);

void om_addSeparatorCB(const char *object, const char *rootName, int is_timestep, int min, int max);
SoGroup *om_addUGridCB(const char *object, const char *rootName, int xsize, int ysize,
                       int zsize, float xmin, float xmax,
                       float ymin, float ymax, float zmin, float zmax,
                       int no_of_colors, int colorbinding, int colorpacking,
                       float *r, float *g, float *b, uint32_t *pc, unsigned char *byteData,
                       int no_of_normals, int normalbinding,
                       float *nx, float *ny, float *nz, float transparency);

SoGroup *om_addRGridCB(const char *object, const char *rootName, int xsize, int ysize,
                       int zsize, float *x_c, float *y_c,
                       float *z_c,
                       int no_of_colors, int colorbinding, int colorpacking,
                       float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                       int normalbinding,
                       float *nx, float *ny, float *nz, float transparency);

SoGroup *om_addSGridCB(const char *object, const char *rootName, int xsize, int ysize,
                       int zsize, float *x_c, float *y_c,
                       float *z_c,
                       int no_of_colors, int colorbinding, int colorpacking,
                       float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                       int normalbinding,
                       float *nx, float *ny, float *nz, float transparency);

SoGroup *om_addPolygonCB(const char *object, const char *rootName, int no_of_polygons, int no_of_vertices,
                         int no_of_coords, float *x_c, float *y_c, float *z_c,
                         int *v_l, int *l_l, int no_of_colors,
                         int colorbinding, int colorpacking,
                         float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                         int normalbinding,
                         float *nx, float *ny, float *nz, float transparency,
                         int vertexOrder,
                         int texWidth, int texHeight, int pixelSize, unsigned char *image,
                         int no_of_texCoords, float *tx, float *ty, coMaterial *,
                         char *rName /*=NULL*/, char *label /*=NULL*/);

SoGroup *om_addTriangleStripCB(const char *object, const char *rootName, int no_of_strips, int no_of_vertices,
                               int no_of_coords, float *x_c, float *y_c, float *z_c,
                               int *v_l, int *l_l, int no_of_colors,
                               int colorbinding, int colorpacking,
                               float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                               int normalbinding,
                               float *nx, float *ny, float *nz, float transparency,
                               int vertexOrder,
                               int texWidth, int texHeight, int pixelSize, unsigned char *image,
                               int no_of_texCoords, float *tx, float *ty, coMaterial *,
                               char *rName /*=NULL*/, char *label /*=NULL*/);

SoGroup *om_addLineCB(const char *object, const char *rootName, int no_of_lines, int no_of_vertices,
                      int no_of_coords, float *x_c, float *y_c,
                      float *z_c,
                      int *v_l, int *i_l,
                      int no_of_colors, int colorbinding, int colorpacking,
                      float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                      int normalbinding,
                      float *nx, float *ny, float *nz, coMaterial *material,
                      char *rName /*=NULL*/, char *label /*=NULL*/);

SoGroup *om_addPointCB(const char *object, const char *rootName, int no_of_points,
                       float *x_c, float *y_c, float *z_c,
                       int numColors, int colorbinding, int colorpacking,
                       float *r, float *g, float *b,
                       uint32_t *packedColor, float pointsize);

void om_addIvCB(const char *object, const char *rootName, char *IvDescription, int size);
void om_addColormapCB(const char *object, const char *colormap);
void om_addPartCB(const char *object, int part_id, SoSwitch *s);
void om_addTimePartCB(const char *object, int timeStep, int part_id, SoSwitch *s);

SoGroup *om_replaceUGridCB(const char *object, int xsize, int ysize,
                           int zsize, float xmin, float xmax,
                           float ymin, float ymax, float zmin, float zmax,
                           int no_of_colors, int colorbinding, int colorpacking,
                           float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                           int normalbinding,
                           float *nx, float *ny, float *nz, float transparency);

SoGroup *om_replaceRGridCB(const char *object, int xsize, int ysize,
                           int zsize, float *x_c, float *y_c,
                           float *z_c,
                           int no_of_colors, int colorbinding, int colorpacking,
                           float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                           int normalbinding,
                           float *nx, float *ny, float *nz, float transparency);

SoGroup *om_replaceSGridCB(const char *object, int xsize, int ysize,
                           int zsize, float *x_c, float *y_c,
                           float *z_c,
                           int no_of_colors, int colorbinding, int colorpacking,
                           float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                           int normalbinding,
                           float *nx, float *ny, float *nz, float transparency);

SoGroup *om_replacePolygonCB(const char *object, int no_of_polygons, int no_of_vertices,
                             int no_of_coords, float *x_c, float *y_c,
                             float *z_c,
                             int *v_l, int *i_l,
                             int no_of_colors, int colorbinding, int colorpacking,
                             float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                             int normalbinding,
                             float *nx, float *ny, float *nz, float transparency, int vertexOrder,
                             int texWidth, int texHeight, int pixelSize, unsigned char *image,
                             int no_of_texCoords, float *tx, float *ty, coMaterial *);

SoGroup *om_replaceTriangleStripCB(const char *object, int no_of_strips, int no_of_vertices,
                                   int no_of_coords, float *x_c, float *y_c,
                                   float *z_c,
                                   int *v_l, int *i_l,
                                   int no_of_colors, int colorbinding, int colorpacking,
                                   float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                                   int normalbinding,
                                   float *nx, float *ny, float *nz, float transparency, int vertexOrder,
                                   int texWidth, int texHeight, int pixelSize, unsigned char *image,
                                   int no_of_texCoords, float *tx, float *ty, coMaterial *);

SoGroup *om_replaceLineCB(const char *object, int no_of_lines, int no_of_vertices,
                          int no_of_coords, float *x_c, float *y_c,
                          float *z_c,
                          int *v_l, int *i_l,
                          int no_of_colors, int colorbinding, int colorpacking,
                          float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                          int normalbinding,
                          float *nx, float *ny, float *nz);

SoGroup *om_replacePointCB(const char *object, int no_of_points,
                           float *x_c, float *y_c, float *z_c,
                           int colorbinding, int colorpacking, float *r, float *g, float *b, uint32_t *pc);
void om_replaceIvCB(const char *object, char *IvDescription, int size);
void om_replaceColormapCB(const char *object, const char *colormap);
void om_replacePartCB(const char *object, int part_id, SoSwitch *s);
void om_replaceTimePartCB(const char *object, int timeStep, int part_id, SoSwitch *s);

void om_addInteractor(coDistributedObject *obj);
#endif
