/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*! \file
 \brief unpack COVISE data objects

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C) 1996
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   20.08.1997
 */

#ifndef GEOMETRY_MANAGER
#define GEOMETRY_MANAGER

#include <util/coMaterial.h>
using covise::coMaterial;

#include <osg/Texture>
#include <osg/Material>
#include <osg/ref_ptr>

namespace osgUtil
{
class TriStripVisitor;
}

namespace osg
{
class DrawElementsUShort;
class Geode;
class Geometry;
class StateSet;
class Node;
class Group;
}

namespace opencover
{

class COVISEPLUGINEXPORT GeometryManager
{
private:
    bool backfaceCulling;
    bool genStrips;
    osgUtil::TriStripVisitor *d_stripper;

    int sequential;

    osg::ref_ptr<osg::Material> globalDefaultMaterial;

    void setDefaultMaterial(osg::StateSet *geoState, bool transparent, coMaterial *material = NULL, bool isLightingOn = true);

    void setTexture(const unsigned char *image, int pixelSize, int width, int height, osg::StateSet *geoState, osg::Texture::WrapMode wm, osg::Texture::FilterMode minfm, osg::Texture::FilterMode magfm);

    osg::Vec4 coviseGeometryDefaultColor;

public:
    static GeometryManager *instance();
    GeometryManager();

    osg::Group *addGroup(const char *object, bool is_timestep);

    osg::Node *addUGrid(const char *object, int xsize, int ysize,
                        int zsize, float xmin, float xmax,
                        float ymin, float ymax, float zmin, float zmax,
                        int no_of_colors, int colorbinding, int colorpacking,
                        float *r, float *g, float *b, int *pc, int no_of_normals,
                        int normalbinding,
                        float *nx, float *ny, float *nz, float &transparency);

    osg::Node *addRGrid(const char *object, int xsize, int ysize,
                        int zsize, float *x_c, float *y_c,
                        float *z_c,
                        int no_of_colors, int colorbinding, int colorpacking,
                        float *r, float *g, float *b, int *pc, int no_of_normals,
                        int normalbinding,
                        float *nx, float *ny, float *nz, float &transparency);

    osg::Node *addSGrid(const char *object, int xsize, int ysize,
                        int zsize, float *x_c, float *y_c,
                        float *z_c,
                        int no_of_colors, int colorbinding, int colorpacking,
                        float *r, float *g, float *b, int *pc, int no_of_normals,
                        int normalbinding,
                        float *nx, float *ny, float *nz, float &transparency);

    osg::Node *addPolygon(const char *object_name,
                          int no_of_polygons, int no_of_vertices, int no_of_coords,
                          float *x_c, float *y_c, float *z_c,
                          int *v_l, int *i_l,
                          int no_of_colors, int colorbinding, int colorpacking,
                          float *r, float *g, float *b, int *pc,
                          int no_of_normals, int normalbinding,
                          float *nx, float *ny, float *nz,
                          float &transparency, int vertexOrder,
                          coMaterial *, int texWidth, int texHeight, int pixelSize, unsigned char *image,
                          int no_of_texCoords, float *tx, float *ty, osg::Texture::WrapMode wm, osg::Texture::FilterMode minfm, osg::Texture::FilterMode magfm,
                          int no_of_vertexAttributes,
                          float *vax, float *vay, float *vaz, bool cullBackfaces);

    osg::Node *addTriangles(const char *object_name,
                            int no_of_vertices, int no_of_coords,
                            float *x_c, float *y_c, float *z_c,
                            int *v_l,
                            int no_of_colors, int colorbinding, int colorpacking,
                            float *r, float *g, float *b, int *pc,
                            int no_of_normals, int normalbinding,
                            float *nx, float *ny, float *nz,
                            float &transparency, int, coMaterial *material, int texWidth, int texHeight, int pixelSize, unsigned char *image,
                            int no_of_texCoords, float *tx, float *ty, osg::Texture::WrapMode wm, osg::Texture::FilterMode minfm, osg::Texture::FilterMode magfm,
                            int no_of_vertexAttributes,
                            float *vax, float *vay, float *vaz, bool cullBackfaces);

    osg::Node *addQuads(const char *object_name,
                        int no_of_vertices, int no_of_coords,
                        float *x_c, float *y_c, float *z_c,
                        int *v_l,
                        int no_of_colors, int colorbinding, int colorpacking,
                        float *r, float *g, float *b, int *pc,
                        int no_of_normals, int normalbinding,
                        float *nx, float *ny, float *nz,
                        float &transparency, int, coMaterial *material, int texWidth, int texHeight, int pixelSize, unsigned char *image,
                        int no_of_texCoords, float *tx, float *ty, osg::Texture::WrapMode wm, osg::Texture::FilterMode minfm, osg::Texture::FilterMode magfm,
                        int no_of_vertexAttributes,
                        float *vax, float *vay, float *vaz, bool cullBackfaces);

    osg::Node *addTriangleStrip(const char *object, int no_of_strips, int no_of_vertices,
                                int no_of_coords, float *x_c, float *y_c,
                                float *z_c,
                                int *v_l, int *i_l,
                                int no_of_colors, int colorbinding, int colorpacking,
                                float *r, float *g, float *b, int *pc, int no_of_normals,
                                int normalbinding,
                                float *nx, float *ny, float *nz, float &transparency,
                                int vertexOrder,
                                coMaterial *, int texWidth, int texHeight, int pixelSize, unsigned char *image,
                                int no_of_texCoords, float *tx, float *ty, osg::Texture::WrapMode wm, osg::Texture::FilterMode minfm, osg::Texture::FilterMode magfm,
                                int no_of_vertexAttributes,
                                float *vax, float *vay, float *vaz, bool cullBackfaces);

    osg::Node *addLine(const char *object, int no_of_lines, int no_of_vertices,
                       int no_of_coords, float *x_c, float *y_c,
                       float *z_c,
                       int *v_l, int *i_l,
                       int no_of_colors, int colorbinding, int colorpacking,
                       float *r, float *g, float *b, int *pc, int no_of_normals,
                       int normalbinding,
                       float *nx, float *ny, float *nz, int is_trace, coMaterial *,
                       int texWidth, int texHeight, int pixelSize, unsigned char *image,
                       int no_of_texCoords, float *tx, float *ty, osg::Texture::WrapMode wm, osg::Texture::FilterMode minfm, osg::Texture::FilterMode magfm,
                       float linewidth);

    osg::Node *addPoint(const char *object, int no_of_points,
                        float *x_c, float *y_c, float *z_c,
                        int colorbinding, int colorpacking, float *r, float *g, float *b, int *pc,
                        coMaterial *, float pointsize);

    osg::Node *addSphere(const char *object_name, int no_of_points,
                         float *x_c, float *y_c, float *z_c,
                         int iRenderMethod,
                         float *radii_c, int colorbinding, float *rgbR_c, float *rgbG_c, float *rgbB_c, int *type_c,
                         int no_of_normals, float *nx, float *ny, float *nz, int no_of_vertexAttributes,
                         float *vax, float *vay, float *vaz,
                         coMaterial *material = NULL);

    ~GeometryManager()
    {
    }
};
}
#endif
