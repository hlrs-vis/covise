/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "HfT_osg_StateSet.h"

using namespace osg;

HfT_osg_StateSet::HfT_osg_StateSet()
{
    m_Surfmode = SPOINTS;
    m_Image = 0L;
    this->createMode();
}
HfT_osg_StateSet::HfT_osg_StateSet(SurfMode iSurfMode)
{
    m_Surfmode = iSurfMode;
    m_Image = 0L;
    m_Panz = 0;
    this->createMode();
}
HfT_osg_StateSet::HfT_osg_StateSet(SurfMode iSurfMode, Vec4 color)
{
    m_Surfmode = iSurfMode;
    m_Image = 0L;
    m_Panz = 0;
    this->createMode(color);
}
HfT_osg_StateSet::HfT_osg_StateSet(SurfMode iSurfMode, Image *image)
{
    m_Surfmode = iSurfMode;
    m_Image = image;
    m_Panz = 0;
    this->createMode();
}
HfT_osg_StateSet::HfT_osg_StateSet(SurfMode iSurfMode, Image *image, int anz)
{
    m_Surfmode = iSurfMode;
    m_Image = image;
    m_Panz = anz;
    this->createMode();
}
HfT_osg_StateSet::~HfT_osg_StateSet()
{
    m_Surfmode = SPOINTS;
    m_Image = 0L;
    m_Panz = 0;
}
void HfT_osg_StateSet::setImage(Image *image)
{
    m_Image = image;
}
Image *HfT_osg_StateSet::getImage()
{
    return m_Image;
}
void HfT_osg_StateSet::setColor(SurfMode mode, Vec4 color)
{
    switch (mode)
    {
    case SPOINTS:
    {
        m_ColorPoint = color;
    }
    break;
    case SLINES:
    {
        m_ColorLine = color;
    }
    break;
    case STRIANGLES:
    {
        m_ColorTriangle = color;
    }
    break;
    case SQUADS:
    {
        m_ColorQuad = color;
    }
    break;
    case SSHADE:
    {
        m_ColorShade = color;
    }
    break;
    case STEXTURE:
    {
        m_ColorTexture = color;
    }
    break;
    case STRANSPARENT:
    {
        m_ColorTransparent = color;
    }
    break;
    case SGAUSS:
    {
        m_ColorCurvature = color;
    }
    break;
    case SMEAN:
    {
        m_ColorCurvature = color;
    }
    break;
    default:
        break;
    }
}

Vec4 HfT_osg_StateSet::getColor()
{
    Vec4 color = Vec4(1., 1., 1., 1.);
    switch (this->m_Surfmode)
    {
    //Pointcloud
    case SPOINTS:
    {
        color = m_ColorPoint;
    }
    break;
    //Paralines
    case SLINES:
    {
        color = m_ColorLine;
    }
    break;
    //Triangles
    case STRIANGLES:
    {
        color = m_ColorTriangle;
    }
    break;
    //Quads
    case SQUADS:
    {
        color = m_ColorQuad;
    }
    break;
    //Shade mode
    case SSHADE:
    {
        color = m_ColorShade;
    }
    break;
    //Texture mode
    case STEXTURE:
    {
        color = m_ColorTexture;
    }
    break;
    //Transparent mode
    case STRANSPARENT:
    {
        color = m_ColorTransparent;
    }
    break;
    case SGAUSS:
    {
        color = m_ColorCurvature;
    }
    break;
    case SMEAN:
    {
        color = m_ColorCurvature;
    }
    break;
    default:
        break;
    }
    return color;
}
void HfT_osg_StateSet::createMode()
{
    switch (this->m_Surfmode)
    {
    //Pointcloud
    case SPOINTS:
    {
        createPointMode();
    }
    break;
    //Paralines
    case SLINES:
    {
        createLineMode();
    }
    break;
    //Triangles
    case STRIANGLES:
    {
        createTriangleMode();
    }
    break;
    //Quads
    case SQUADS:
    {
        createQuadMode();
    }
    break;
    //Shade mode
    case SSHADE:
    {
        createShadeMode();
    }
    break;
    //Texture mode
    case STEXTURE:
    {
        createTextureMode();
    }
    break;
    //Transparent mode
    case STRANSPARENT:
    {
        createTransparentMode();
    }
    break;
    case SGAUSS:
    {
        createGaussCurvatureMode();
    }
    break;
    case SMEAN:
    {
        createMeanCurvatureMode();
    }
    break;
    default:
        break;
    }
}

void HfT_osg_StateSet::createMode(Vec4 color)
{
    switch (this->m_Surfmode)
    {
    //Pointcloud
    case SPOINTS:
    {
        createPointMode(color);
    }
    break;
    //Paralines
    case SLINES:
    {
        createLineMode(color);
    }
    break;
    //Triangles
    case STRIANGLES:
    {
        createTriangleMode(color);
    }
    break;
    //Quads
    case SQUADS:
    {
        createQuadMode(color);
    }
    break;
    //Shade mode
    case SSHADE:
    {
        createShadeMode(color);
    }
    break;
    //Texture mode
    case STEXTURE:
    {
        createTextureMode(color);
    }
    break;
    //Transparent mode
    case STRANSPARENT:
    {
        createTransparentMode(color);
    }
    break;
    case SGAUSS:
    {
        createGaussCurvatureMode();
    }
    break;
    case SMEAN:
    {
        createMeanCurvatureMode();
    }
    break;
    default:
        break;
    }
}
void HfT_osg_StateSet::initializePointMembers()
{
    m_MaterialPoint = new Material();
    m_Point = new Point(3.f);
}
void HfT_osg_StateSet::createPointMode()
{
    initializePointMembers();
    this->setAttributeAndModes(m_Point, StateAttribute::PROTECTED);
    this->setAttributeAndModes(m_MaterialPoint, StateAttribute::PROTECTED);
}
void HfT_osg_StateSet::createPointMode(Vec4 color)
{
    initializePointMembers();
    m_ColorPoint = color;
    this->setAttributeAndModes(m_Point, StateAttribute::PROTECTED);
    this->setAttributeAndModes(m_MaterialPoint, StateAttribute::PROTECTED);
}

void HfT_osg_StateSet::initializeLineMembers()
{
    m_MaterialLine = new Material();
    m_Line = new LineWidth(2.f);
    m_ColorLine = Vec4(1.f, 1.f, 0.f, 1.f);
}
void HfT_osg_StateSet::createLineMode(Vec4 color)
{
    initializeLineMembers();
    m_ColorLine = color;
    setColorAndMaterial(m_ColorLine, m_MaterialLine);
    this->setAttributeAndModes(m_Line, StateAttribute::PROTECTED);
    this->setAttributeAndModes(m_MaterialLine, StateAttribute::PROTECTED);
}
void HfT_osg_StateSet::createLineMode()
{
    initializeLineMembers();
    setColorAndMaterial(m_ColorLine, m_MaterialLine);
    this->setAttributeAndModes(m_Line, StateAttribute::PROTECTED);
    this->setAttributeAndModes(m_MaterialLine, StateAttribute::PROTECTED);
}

void HfT_osg_StateSet::initializeTriangleMembers()
{
    m_Triangle = new PolygonMode(PolygonMode::FRONT_AND_BACK, PolygonMode::LINE);
    m_MaterialTriangle = new Material();
    m_ColorTriangle = Vec4(0.f, 1.f, 1.f, 1.f);
}
void HfT_osg_StateSet::createTriangleMode(Vec4 color)
{
    initializeTriangleMembers();
    m_ColorTriangle = color;
    setColorAndMaterial(m_ColorTriangle, m_MaterialTriangle);
    this->setAttributeAndModes(m_Triangle, StateAttribute::PROTECTED);
    this->setAttributeAndModes(m_MaterialTriangle, StateAttribute::PROTECTED);
}
void HfT_osg_StateSet::createTriangleMode()
{
    initializeTriangleMembers();
    setColorAndMaterial(m_ColorTriangle, m_MaterialTriangle);
    this->setAttributeAndModes(m_Triangle, StateAttribute::PROTECTED);
    this->setAttributeAndModes(m_MaterialTriangle, StateAttribute::PROTECTED);
}
void HfT_osg_StateSet::initializeQuadMembers()
{
    m_Quad = new PolygonMode(PolygonMode::FRONT_AND_BACK, PolygonMode::LINE);
    m_MaterialQuad = new Material();
    m_ColorQuad = Vec4(0.f, 0.f, 1.f, 1.f);
}
void HfT_osg_StateSet::createQuadMode(Vec4 color)
{
    initializeQuadMembers();
    m_ColorQuad = color;
    setColorAndMaterial(m_ColorQuad, m_MaterialQuad);
    this->setAttributeAndModes(m_Quad, StateAttribute::PROTECTED);
    this->setAttributeAndModes(m_MaterialQuad, StateAttribute::PROTECTED);
}
void HfT_osg_StateSet::createQuadMode()
{
    initializeQuadMembers();
    setColorAndMaterial(m_ColorQuad, m_MaterialQuad);
    this->setAttributeAndModes(m_Quad, StateAttribute::PROTECTED);
    this->setAttributeAndModes(m_MaterialQuad, StateAttribute::PROTECTED);
}
void HfT_osg_StateSet::initializeShadeMembers()
{
    m_MaterialShade = new Material();
    m_ColorShade = Vec4(1.f, 0.5f, 0.f, 1.f);
    initializeLineMembers();
}
void HfT_osg_StateSet::createShadeMode(Vec4 color)
{
    initializeShadeMembers();
    m_ColorShade = color;
    setColorAndMaterial(m_ColorShade, m_MaterialShade);
    this->setMode(GL_BLEND, StateAttribute::PROTECTED);
    this->setAttributeAndModes(m_MaterialShade, StateAttribute::PROTECTED);
}
void HfT_osg_StateSet::createShadeMode()
{
    initializeShadeMembers();
    setColorAndMaterial(m_ColorShade, m_MaterialShade);
    this->setMode(GL_BLEND, StateAttribute::PROTECTED);
    this->setAttributeAndModes(m_MaterialShade, StateAttribute::PROTECTED);
}
void HfT_osg_StateSet::initializeTextureMembers()
{
    m_Texture = new Texture2D();
    if (m_Image)
        m_Texture->setImage(m_Image);
    m_MaterialTexture = new Material();
    m_ColorTexture = Vec4(0.6f, 0.6f, 0.6f, 1.f);
}
void HfT_osg_StateSet::createTextureMode(Vec4 color)
{
    initializeTextureMembers();
    m_ColorTexture = color;
    setColorAndMaterial(m_ColorTexture, m_MaterialTexture);
    this->setAttributeAndModes(m_MaterialTexture, StateAttribute::ON);
    this->setTextureAttributeAndModes(0, m_Texture, StateAttribute::ON);
}
void HfT_osg_StateSet::createTextureMode()
{
    initializeTextureMembers();
    setColorAndMaterial(m_ColorTexture, m_MaterialTexture);
    this->setAttributeAndModes(m_MaterialTexture, StateAttribute::ON);
    this->setTextureAttributeAndModes(0, m_Texture, StateAttribute::ON);
}
void HfT_osg_StateSet::initializeTransparentMembers()
{
    m_TransparentTexture = new Texture2D();
    if (m_Image)
        m_TransparentTexture->setImage(m_Image);
    m_MaterialTransparent = new Material();
    m_ColorTransparent = Vec4(0.6f, 0.6f, 0.6f, 1.f);
}
void HfT_osg_StateSet::createTransparentMode(Vec4 color)
{
    initializeTransparentMembers();
    m_ColorTransparent = color;
    this->setMode(GL_BLEND, StateAttribute::ON);
    setColorAndMaterial(m_ColorTransparent, m_MaterialTransparent);
    this->setAttributeAndModes(m_MaterialTransparent, StateAttribute::ON);

    this->setRenderingHint(StateSet::TRANSPARENT_BIN);
    m_MaterialTransparent->setTransparency(Material::FRONT_AND_BACK, 0.7f);
    this->setTextureAttributeAndModes(0, m_TransparentTexture, StateAttribute::ON);
}
void HfT_osg_StateSet::createTransparentMode()
{
    initializeTransparentMembers();
    this->setMode(GL_BLEND, StateAttribute::ON);
    setColorAndMaterial(m_ColorTransparent, m_MaterialTransparent);
    this->setAttributeAndModes(m_MaterialTransparent, StateAttribute::ON);

    this->setRenderingHint(StateSet::TRANSPARENT_BIN);
    m_MaterialTransparent->setTransparency(Material::FRONT_AND_BACK, 0.7f);
    this->setTextureAttributeAndModes(0, m_TransparentTexture, StateAttribute::ON);
}
void HfT_osg_StateSet::initializeGaussCurvatureMembers()
{
    m_MaterialCurvature = new Material();
    // Es wird ColorArray benutzt, daher keine spezielle Farbe zuweisen
    m_ColorCurvature = Vec4(1.f, 1.f, 1.f, 1.f);
}
void HfT_osg_StateSet::initializeMeanCurvatureMembers()
{
    m_MaterialCurvature = new Material();
    // Es wird ColorArray benutzt, daher keine spezielle Farbe zuweisen
    m_ColorCurvature = Vec4(1.f, 1.f, 1.f, 1.f);
}
void HfT_osg_StateSet::createGaussCurvatureMode()
{
    initializeGaussCurvatureMembers();
    m_MaterialCurvature->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    this->setAttributeAndModes(m_MaterialCurvature, StateAttribute::PROTECTED);
}
void HfT_osg_StateSet::createMeanCurvatureMode()
{
    initializeMeanCurvatureMembers();
    m_MaterialCurvature->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    this->setAttributeAndModes(m_MaterialCurvature, StateAttribute::PROTECTED);
}
void HfT_osg_StateSet::setColorAndMaterial(Vec4 Color, Material *Mat)
{
    //Je stärker eine Seite in das Licht gehalten/abgewendet wird, desto heller/dunkler erscheint die Oberfläche.
    //Mat->setDiffuse(Material::FRONT_AND_BACK, Vec4f(1.,1.,1.,1.));
    // Weißes Umgebungslicht
    //Mat->setAmbient(Material::FRONT_AND_BACK, Vec4f(1.,1.,1.,1.));

    // Glanz des Materials: Metall glänzend, Holz nicht glänzend
    Mat->setSpecular(Material::FRONT_AND_BACK, Color);

    // Die Emission von sichtbarem Licht macht ein Objekt zu einer selbstleuchtenden Lichtquelle.
    Mat->setEmission(Material::FRONT_AND_BACK, Color);

    // Shininess: Lichtreflexion Werte von 0 --> 128
    Mat->setShininess(Material::FRONT_AND_BACK, 64.);
}
void HfT_osg_StateSet::recomputeMode(SurfMode iSurfMode)
{
    this->clearMode();
    m_Surfmode = iSurfMode;
    this->createMode();
}
void HfT_osg_StateSet::recomputeMode(SurfMode iSurfMode, Vec4 Color)
{
    this->clearMode();
    m_Surfmode = iSurfMode;
    this->createMode(Color);
}
void HfT_osg_StateSet::recomputeMode(SurfMode iSurfMode, Vec4 Color, Image *image)
{
    this->clearMode();
    m_Image = image;
    m_Surfmode = iSurfMode;
    this->createMode(Color);
}
void HfT_osg_StateSet::clearMode()
{
    // Hier: StateSet-Attribute mit remove abtrennen
    switch (this->m_Surfmode)
    {
    case SPOINTS:
    {
        clearPointMode();
    }
    break;
    case SLINES:
    {
        clearLineMode();
    }
    break;
    case STRIANGLES:
    {
        clearTriangleMode();
    }
    break;
    case SQUADS:
    {
        clearQuadMode();
    }
    break;
    case SSHADE:
    {
        clearShadeMode();
    }
    break;
    case STEXTURE:
    {
        clearTextureMode();
    }
    break;
    case STRANSPARENT:
    {
        clearTransparentMode();
    }
    break;
    case SGAUSS:
    {
        clearGaussCurvatureMode();
    }
    break;
    case SMEAN:
    {
        clearMeanCurvatureMode();
    }
    break;
    default:
        break;
    }
}
void HfT_osg_StateSet::clearPointMode()
{
    this->removeAttribute(m_Point);
    this->removeAttribute(m_MaterialPoint);
}

void HfT_osg_StateSet::clearLineMode()
{
    this->removeAttribute(m_Line);
    this->removeAttribute(m_MaterialLine);
}
void HfT_osg_StateSet::clearTriangleMode()
{
    this->removeAttribute(m_Triangle);
    this->removeAttribute(m_MaterialTriangle);
}
void HfT_osg_StateSet::clearQuadMode()
{
    this->removeAttribute(m_Quad);
    this->removeAttribute(m_MaterialQuad);
}
void HfT_osg_StateSet::clearShadeMode()
{
    this->removeMode(GL_BLEND);
    this->removeMode(StateSet::TRANSPARENT_BIN);
    this->removeAttribute(m_MaterialShade);
}
void HfT_osg_StateSet::clearTextureMode()
{
    //m_Image = 0L;
    this->removeAttribute(m_MaterialTexture);
    this->removeTextureAttribute(0, m_Texture);
}
void HfT_osg_StateSet::clearTransparentMode()
{
    //m_Image = 0L;
    this->removeMode(GL_BLEND);
    this->removeMode(StateSet::TRANSPARENT_BIN);
    this->removeAttribute(m_MaterialTransparent);
    this->removeTextureAttribute(0, m_TransparentTexture);
}
void HfT_osg_StateSet::clearGaussCurvatureMode()
{
    this->removeAttribute(m_MaterialCurvature);
}
void HfT_osg_StateSet::clearMeanCurvatureMode()
{
    this->removeAttribute(m_MaterialCurvature);
}
void HfT_osg_StateSet::setSurfMode(SurfMode mode)
{
    m_Surfmode = mode;
}
SurfMode HfT_osg_StateSet::getSurfMode()
{
    return m_Surfmode;
}
void HfT_osg_StateSet::setMaterialShade(Material *mat)
{
    m_MaterialShade = mat;
}
void HfT_osg_StateSet::setMaterialTransparency(Material *mat)
{
    m_MaterialTransparent = mat;
}
Material *HfT_osg_StateSet::getMaterialShade()
{
    return m_MaterialShade;
}
Material *HfT_osg_StateSet::getMaterialTransparency()
{
    return m_MaterialTransparent;
}
