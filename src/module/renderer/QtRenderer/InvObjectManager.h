/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_OBJECT_MANAGER_H
#define _INV_OBJECT_MANAGER_H

#include <covise/covise.h>

//
// other classes
//

#include "InvObjectList.h"
#include "InvObjects.h"
#include "InvError.h"
#include "InvSequencer.h"

//
// Inventor stuff
//

#include <Inventor/Qt/SoQt.h>
#include <Inventor/Qt/viewers/SoQtExaminerViewer.h>
#include <Inventor/nodes/SoGroup.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoNode.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoTransform.h>
#include <Inventor/SoInput.h>

//

// Covise List class AW
//
#include <util/covise_list.h>
#include <covise/covise_process.h>
#include <do/coDistributedObject.h>

namespace covise
{
class coMaterial;
}

//================================================================
// InvObjectManager
//================================================================

class InvObjectManager
{

private:
    static covise::List<SoSwitch> timestepSwitchList;

    SoNode *root;

public:
    const char *getname(const char *file);
    InvObjectList *list;
    InvObjectManager();
    SoNode *createInventorGraph();
    ~InvObjectManager();

    void setMeshCoords(SoCoordinate3 *coord, SoQuadMesh *quadmesh, int VerticesPerRow, int VerticesPerColumn,
                       float *x_c, float *y_c, float *z_c);

    void receiveSequencer(const char *message);
    void addObjectCB(const char *rootname, const char *object, SoGroup *ptr, bool);
    void deleteObjectCB(const char *name);
    void updateObjectList(const char *name, bool);
    void addColormap(const char *name, const char *colormap);
    void deleteColormap(const char *name);

    void addIvCB(const char *object, const char *rootName, char *IvDescription, int size);
    void replaceIvCB(const char *object, char *IvDescription, int size);
    void addSeparatorCB(const char *object, const char *rootName, int is_timestep, int min, int max);

    SoGroup *addUGridCB(const char *object, const char *rootName, int xsize, int ysize,
                        int zsize, float xmin, float xmax,
                        float ymin, float ymax, float zmin, float zmax,
                        int no_of_colors, int colorbinding, int colorpacking,
                        float *r, float *g, float *b, uint32_t *pc, uchar *byteData,
                        int no_of_normals, int normalbinding,
                        float *nx, float *ny, float *nz, float transparency,
                        int no_of_lut_entries, uchar *rgba_lut);

    SoGroup *addRGridCB(const char *object, const char *rootName, int xsize, int ysize,
                        int zsize, float *x_c, float *y_c,
                        float *z_c,
                        int no_of_colors, int colorbinding, int colorpacking,
                        float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                        int normalbinding,
                        float *nx, float *ny, float *nz, float transparency);

    SoGroup *addSGridCB(const char *object, const char *rootName, int xsize, int ysize,
                        int zsize, float *x_c, float *y_c,
                        float *z_c,
                        int no_of_colors, int colorbinding, int colorpacking,
                        float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                        int normalbinding,
                        float *nx, float *ny, float *nz, float transparency);

    SoGroup *addPolygonCB(const char *object, const char *rootName, int no_of_polygons, int no_of_vertices,
                          int no_of_coords, float *x_c, float *y_c, float *z_c,
                          int *v_l, int *l_l, int no_of_colors,
                          int colorbinding, int colorpacking,
                          float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                          int normalbinding,
                          float *nx, float *ny, float *nz, float transparency,
                          int vertexOrder,
                          int texWidth, int texHeight, int pixelSize, unsigned char *image,
                          int no_of_texCoords, float *tx, float *ty, covise::coMaterial *,
                          const char *rName = NULL, const char *label = NULL);

    SoGroup *addTriangleStripCB(const char *object, const char *rootName, int no_of_strips, int no_of_vertices,
                                int no_of_coords, float *x_c, float *y_c, float *z_c,
                                int *v_l, int *l_l, int no_of_colors,
                                int colorbinding, int colorpacking,
                                float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                                int normalbinding,
                                float *nx, float *ny, float *nz, float transparency,
                                int vertexOrder,
                                int texWidth, int texHeight, int pixelSize, unsigned char *image,
                                int no_of_texCoords, float *tx, float *ty, covise::coMaterial *,
                                const char *rName = NULL, const char *label = NULL);

    SoGroup *addLineCB(const char *object, const char *rootName, int no_of_lines, int no_of_vertices,
                       int no_of_coords, float *x_c, float *y_c,
                       float *z_c,
                       int *v_l, int *i_l,
                       int no_of_colors, int colorbinding, int colorpacking,
                       float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                       int normalbinding,
                       float *nx, float *ny, float *nz, covise::coMaterial *material,
                       const char *rName = NULL, const char *label = NULL);

    SoGroup *addPointCB(const char *object, const char *rootName, int no_of_points,
                        float *x_c, float *y_c, float *z_c,
                        int numColors, int colorbinding, int colorpacking,
                        float *r, float *g, float *b,
                        uint32_t *pc, float pointsize);

    SoGroup *addSphereCB(const char *object, const char *rootName, int no_of_points,
                         float *x_c, float *y_c, float *z_c, float *radii_c,
                         int numColors, int colorbinding, int colorpacking,
                         float *r, float *g, float *b,
                         uint32_t *pc, int iRenderMethod, const char *rname);

    SoGroup *replaceUGridCB(const char *object, int xsize, int ysize,
                            int zsize, float xmin, float xmax,
                            float ymin, float ymax, float zmin, float zmax,
                            int no_of_colors, int colorbinding, int colorpacking,
                            float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                            int normalbinding,
                            float *nx, float *ny, float *nz, float transparency);

    SoGroup *replaceRGridCB(const char *object, int xsize, int ysize,
                            int zsize, float *x_c, float *y_c,
                            float *z_c,
                            int no_of_colors, int colorbinding, int colorpacking,
                            float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                            int normalbinding,
                            float *nx, float *ny, float *nz, float transparency);

    SoGroup *replaceSGridCB(const char *object, int xsize, int ysize,
                            int zsize, float *x_c, float *y_c,
                            float *z_c,
                            int no_of_colors, int colorbinding, int colorpacking,
                            float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                            int normalbinding,
                            float *nx, float *ny, float *nz, float transparency);

    SoGroup *replacePolygonCB(const char *object, int no_of_polygons, int no_of_vertices,
                              int no_of_coords, float *x_c, float *y_c,
                              float *z_c,
                              int *v_l, int *i_l,
                              int no_of_colors, int colorbinding, int colorpacking,
                              float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                              int normalbinding,
                              float *nx, float *ny, float *nz, float transparency, int vertexOrder,
                              int texWidth, int texHeight, int pixelSize, unsigned char *image,
                              int no_of_texCoords, float *tx, float *ty, covise::coMaterial *);

    SoGroup *replaceTriangleStripCB(const char *object, int no_of_strips, int no_of_vertices,
                                    int no_of_coords, float *x_c, float *y_c,
                                    float *z_c,
                                    int *v_l, int *i_l,
                                    int no_of_colors, int colorbinding, int colorpacking,
                                    float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                                    int normalbinding,
                                    float *nx, float *ny, float *nz, float transparency, int vertexOrder,
                                    int texWidth, int texHeight, int pixelSize, unsigned char *image,
                                    int no_of_texCoords, float *tx, float *ty, covise::coMaterial *);

    SoGroup *replaceLineCB(const char *object, int no_of_lines, int no_of_vertices,
                           int no_of_coords, float *x_c, float *y_c,
                           float *z_c,
                           int *v_l, int *i_l,
                           int no_of_colors, int colorbinding, int colorpacking,
                           float *r, float *g, float *b, uint32_t *pc, int no_of_normals,
                           int normalbinding,
                           float *nx, float *ny, float *nz);

    SoGroup *replacePointCB(const char *object, int no_of_points,
                            float *x_c, float *y_c, float *z_c,
                            int colorbinding, int colorpacking, float *r, float *g, float *b, uint32_t *pc);

    SoGroup *replaceSphereCB(const char *object, int no_of_points,
                             float *x_c, float *y_c, float *z_c, float *radii_c,
                             int no_of_colors,
                             int colorbinding, int colorpacking, float *r, float *g, float *b, uint32_t *pc,
                             int iRenderMethod);

    void addInteractor(const covise::coDistributedObject *obj);
};
#endif
