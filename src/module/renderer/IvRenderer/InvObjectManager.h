/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_OBJECT_MANAGER_H
#define _INV_OBJECT_MANAGER_H

/* $Id: InvObjectManager.h /main/vir_main/1 12-Dec-2001.12:16:59 sergio_te $ */

/* $Log:  $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

//**************************************************************************
//
// * Description    :  the object manager for the renderer
//
//
// * Class(es)      : InvObjectManager
//
//
// * inherited from : none
//
//
// * Author  : Dirk Rantzau
//
//
// * History : 17.08.93 V 1.0
//
//
//
//**************************************************************************
//
//
//

#include <covise/covise.h>
#include <util/covise_list.h>

//
// other classes
//

#include "InvRenderManager.h"
#include "InvObjectList.h"
#include "InvObjects.h"
#include "InvError.h"
#include "InvSequencer.h"
#include <util/coMaterial.h>

//
// Inventor stuff
//

#include <Inventor/Xt/SoXt.h>
#include <Inventor/Xt/viewers/SoXtExaminerViewer.h>
#include <Inventor/nodes/SoGroup.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoNode.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoTransform.h>
#include <Inventor/SoInput.h>

//
// Covise List class AW
//
//
// ec stuff
//
#include <covise/covise_process.h>

namespace covise
{
class coDistributedObject;
}
using namespace covise;

//================================================================
// InvObjectManager
//================================================================

class InvObjectManager
{

private:
    static List<SoSwitch> timestepSwitchList;

    InvObjectList *list;
    SoNode *root;

    friend void om_timestepCB(void *userData, void *callbackData);
    friend void om_addObjectCB(const char *name, SoGroup *ptr);
    friend void om_deleteObjectCB(const char *name);
    friend void om_deleteColormapCB(const char *name);
    friend void om_deletePartCB(const char *name);
    friend void om_deleteTimePartCB(const char *name);

    friend void om_addSeparatorCB(const char *object, const char *rootName, int is_timestep, int min, int max);
    friend SoGroup *om_addUGridCB(const char *object, const char *rootName, int xsize, int ysize,
                                  int zsize, float xmin, float xmax,
                                  float ymin, float ymax, float zmin, float zmax,
                                  int no_of_colors, int colorbinding, int colorpacking,
                                  float *r, float *g, float *b, uint32_t *pc, unsigned char *byteData,
                                  int no_of_normals, int normalbinding,
                                  float *nx, float *ny, float *nz, float transparency);

    friend SoGroup *om_addRGridCB(const char *object, const char *rootName, int xsize, int ysize,
                                  int zsize, float *x_c, float *y_c,
                                  float *z_c,
                                  int no_of_colors, int colorbinding, int colorpacking,
                                  float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                                  int normalbinding,
                                  float *nx, float *ny, float *nz, float transparency);

    friend SoGroup *om_addSGridCB(const char *object, const char *rootName, int xsize, int ysize,
                                  int zsize, float *x_c, float *y_c,
                                  float *z_c,
                                  int no_of_colors, int colorbinding, int colorpacking,
                                  float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                                  int normalbinding,
                                  float *nx, float *ny, float *nz, float transparency);
    friend SoGroup *om_addPolygonCB(const char *object, const char *rootName, int no_of_polygons, int no_of_vertices,
                                    int no_of_coords, float *x_c, float *y_c, float *z_c,
                                    int *v_l, int *l_l, int no_of_colors,
                                    int colorbinding, int colorpacking,
                                    float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                                    int normalbinding,
                                    float *nx, float *ny, float *nz, float transparency,
                                    int vertexOrder,
                                    int texWidth, int texHeight, int pixelSize, unsigned char *image,
                                    int no_of_texCoords, float *tx, float *ty, coMaterial *,
                                    char *rName, char *label);

    friend SoGroup *om_addTriangleStripCB(const char *object, const char *rootName, int no_of_strips, int no_of_vertices,
                                          int no_of_coords, float *x_c, float *y_c, float *z_c,
                                          int *v_l, int *l_l, int no_of_colors,
                                          int colorbinding, int colorpacking,
                                          float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                                          int normalbinding,
                                          float *nx, float *ny, float *nz, float transparency,
                                          int vertexOrder,
                                          int texWidth, int texHeight, int pixelSize, unsigned char *image,
                                          int no_of_texCoords, float *tx, float *ty, coMaterial *,
                                          char *rName, char *label);

    friend SoGroup *om_addLineCB(const char *object, const char *rootName, int no_of_lines, int no_of_vertices,
                                 int no_of_coords, float *x_c, float *y_c,
                                 float *z_c,
                                 int *v_l, int *i_l,
                                 int no_of_colors, int colorbinding, int colorpacking,
                                 float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                                 int normalbinding,
                                 float *nx, float *ny, float *nz, coMaterial *material,
                                 char *rName, char *label);

    friend SoGroup *om_addPointCB(const char *object, const char *rootName, int no_of_points,
                                  float *x_c, float *y_c, float *z_c,
                                  int numColors, int colorbinding, int colorpacking,
                                  float *r, float *g, float *b,
                                  uint32_t *packedColor, float pointsize);

    friend void om_addIvCB(const char *object, const char *rootName, char *IvDescription, int size);
    friend void om_addColormapCB(const char *object, const char *colormap);
    friend void om_addPartCB(const char *object, int part_id, SoSwitch *s);
    friend void om_addTimePartCB(const char *object, int timeStep, int part_id, SoSwitch *s);

    friend SoGroup *om_replaceUGridCB(const char *object, int xsize, int ysize,
                                      int zsize, float xmin, float xmax,
                                      float ymin, float ymax, float zmin, float zmax,
                                      int no_of_colors, int colorbinding, int colorpacking,
                                      float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                                      int normalbinding,
                                      float *nx, float *ny, float *nz, float transparency);

    friend SoGroup *om_replaceRGridCB(const char *object, int xsize, int ysize,
                                      int zsize, float *x_c, float *y_c,
                                      float *z_c,
                                      int no_of_colors, int colorbinding, int colorpacking,
                                      float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                                      int normalbinding,
                                      float *nx, float *ny, float *nz, float transparency);

    friend SoGroup *om_replaceSGridCB(const char *object, int xsize, int ysize,
                                      int zsize, float *x_c, float *y_c,
                                      float *z_c,
                                      int no_of_colors, int colorbinding, int colorpacking,
                                      float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                                      int normalbinding,
                                      float *nx, float *ny, float *nz, float transparency);

    friend SoGroup *om_replacePolygonCB(const char *object, int no_of_polygons, int no_of_vertices,
                                        int no_of_coords, float *x_c, float *y_c,
                                        float *z_c,
                                        int *v_l, int *i_l,
                                        int no_of_colors, int colorbinding, int colorpacking,
                                        float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                                        int normalbinding,
                                        float *nx, float *ny, float *nz, float transparency, int vertexOrder,
                                        int texWidth, int texHeight, int pixelSize, unsigned char *image,
                                        int no_of_texCoords, float *tx, float *ty, coMaterial *);

    friend SoGroup *om_replaceTriangleStripCB(const char *object, int no_of_strips, int no_of_vertices,
                                              int no_of_coords, float *x_c, float *y_c,
                                              float *z_c,
                                              int *v_l, int *i_l,
                                              int no_of_colors, int colorbinding, int colorpacking,
                                              float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                                              int normalbinding,
                                              float *nx, float *ny, float *nz, float transparency, int vertexOrder,
                                              int texWidth, int texHeight, int pixelSize, unsigned char *image,
                                              int no_of_texCoords, float *tx, float *ty, coMaterial *);

    friend SoGroup *om_replaceLineCB(const char *object, int no_of_lines, int no_of_vertices,
                                     int no_of_coords, float *x_c, float *y_c,
                                     float *z_c,
                                     int *v_l, int *i_l,
                                     int no_of_colors, int colorbinding, int colorpacking,
                                     float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                                     int normalbinding,
                                     float *nx, float *ny, float *nz);

    friend SoGroup *om_replacePointCB(const char *object, int no_of_points,
                                      float *x_c, float *y_c, float *z_c,
                                      int colorbinding, int colorpacking, float *r, float *g, float *b, uint32_t *pc);
    friend void om_replaceIvCB(const char *object, char *IvDescription, int size);
    friend void om_replaceColormapCB(const char *object, char *colormap);
    friend void om_replacePartCB(const char *object, int part_id, SoSwitch *s);
    friend void om_replaceTimePartCB(const char *object, int timeStep, int part_id, SoSwitch *s);

    friend void om_addInteractor(coDistributedObject *obj);

public:
    InvObjectManager();
    SoNode *createInventorGraph();
    ~InvObjectManager();
    static InvSequencer *timeStepper;
    void receiveSequencer(const char *message);
};
#endif
