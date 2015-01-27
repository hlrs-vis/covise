/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef HFT_OSG_STATESET_H
#define HFT_OSG_STATESET_H

#include <osg/StateSet>
#include <osg/Material>
#include <osg/Point>
#include <osg/Image>
#include <osg/LineWidth>
#include <osg/PolygonMode>
#include <osg/Texture2D>
#include <osg/Vec4>

enum SurfMode
{
    SPOINTS,
    SLINES,
    STRIANGLES,
    SQUADS,
    SSHADE,
    STEXTURE,
    STRANSPARENT,
    SGAUSS,
    SMEAN
};
using namespace osg;

class HfT_osg_StateSet : public StateSet
{
public:
    // KONSTRUKTOR:
    HfT_osg_StateSet();
    HfT_osg_StateSet(SurfMode mode);
    HfT_osg_StateSet(SurfMode mode, Vec4 color);
    HfT_osg_StateSet(SurfMode mode, Image *image);
    HfT_osg_StateSet(SurfMode mode, Image *image, int anz);

    //destructor
    virtual ~HfT_osg_StateSet();

    void setSurfMode(SurfMode mode);
    SurfMode getSurfMode();
    void setImage(Image *image);
    Image *getImage();
    void setMaterialShade(Material *mat);
    Material *getMaterialShade();
    void setMaterialTransparency(Material *mat);
    Material *getMaterialTransparency();

    void setColorAndMaterial_Triangle(Vec4 Color, Material *Mat);
    void setColorAndMaterial(Vec4 ioColor, Material *ioMat);
    Vec4 getColor();
    void setColor(SurfMode mode, Vec4 ioColor);

    // createMode Methoden beinhalten die intialize Methoden
    void createMode();
    void createMode(Vec4 color);

    void initializePointMembers();
    void createPointMode();
    void createPointMode(Vec4 color);

    void initializeLineMembers();
    void createLineMode();
    void createLineMode(Vec4 color);

    void initializeTriangleMembers();
    void createTriangleMode();
    void createTriangleMode(Vec4 color);

    void initializeQuadMembers();
    void createQuadMode();
    void createQuadMode(Vec4 color);

    void initializeShadeMembers();
    void createShadeMode();
    void createShadeMode(Vec4 color);

    void initializeTextureMembers();
    void createTextureMode();
    void createTextureMode(Vec4 color);

    void initializeTransparentMembers();
    void createTransparentMode();
    void createTransparentMode(Vec4 color);

    void initializeGaussCurvatureMembers();
    void createGaussCurvatureMode();
    void initializeMeanCurvatureMembers();
    void createMeanCurvatureMode();

    void clearMode();
    void clearPointMode();
    void clearLineMode();

    void clearTriangleMode();
    void clearQuadMode();
    void clearShadeMode();
    void clearTextureMode();
    void clearTransparentMode();
    void clearGaussCurvatureMode();
    void clearMeanCurvatureMode();
    void recomputeMode(SurfMode iSurfMode);
    void recomputeMode(SurfMode iSurfMode, Vec4 color);
    void recomputeMode(SurfMode iSurfMode, Vec4 color, Image *image);

protected:
    SurfMode m_Surfmode;
    Image *m_Image;
    int m_Panz;

    ref_ptr<Material> m_MaterialPoint;
    ref_ptr<osg::Point> m_Point;
    Vec4d m_ColorPoint;

    ref_ptr<Material> m_MaterialLine;
    ref_ptr<LineWidth> m_Line;
    Vec4d m_ColorLine;

    ref_ptr<PolygonMode> m_Triangle;
    ref_ptr<Material> m_MaterialTriangle;
    Vec4d m_ColorTriangle;

    ref_ptr<PolygonMode> m_Quad;
    ref_ptr<Material> m_MaterialQuad;
    Vec4d m_ColorQuad;

    ref_ptr<Material> m_MaterialShade;
    Vec4d m_ColorShade;

    ref_ptr<Texture2D> m_Texture;
    ref_ptr<Material> m_MaterialTexture;
    Vec4d m_ColorTexture;

    ref_ptr<Texture2D> m_TransparentTexture;
    ref_ptr<Material> m_MaterialTransparent;
    Vec4d m_ColorTransparent;

    ref_ptr<Material> m_MaterialCurvature;
    Vec4d m_ColorCurvature;
};
#endif // HFT_OSG_STATESET_H
