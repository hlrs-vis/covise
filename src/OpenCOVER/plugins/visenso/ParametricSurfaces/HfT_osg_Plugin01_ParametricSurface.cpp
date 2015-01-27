/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * HfT_osg_Plugin01_ParametricSurface .cpp
 *
 *  Created on: Nov 4, 2010
 *      Author: ac_te
 */

//Abstrakte Basisklasse
#include "HfT_osg_Plugin01_ParametricSurface.h"
#include "HfT_string.h"

using namespace osg;

HfT_osg_Plugin01_ParametricSurface::HfT_osg_Plugin01_ParametricSurface()
    : Geode()
{
    m_Surftype = 0;
    m_n = 1;
    m_m = 1;
    m_Surfmode = SPOINTS;
    m_su = 20;
    m_sv = 20;
    mp_Image_Texture = 0L;
    m_Pointanz = m_n * (m_su + 1) * m_m * (m_sv + 1);
    m_Pointanz_B = 2 * (m_n * (m_su + 1) + m_m * (m_sv + 1));
    m_type_C = CNOTHING;
    m_Pointanz_C = 0;

    this->initializeMembers();
}
HfT_osg_Plugin01_ParametricSurface::
    HfT_osg_Plugin01_ParametricSurface(double a, double b, double c, int iPatU, int iPatV,
                                       int iSegmU, int iSegmV,
                                       double iLowU, double iUpU,
                                       double iLowV, double iUpV,
                                       SurfMode iSurfMode, ConsType iConsType, int iConsSPOINTS)
    : Geode()
    , m_Surftype(0)
    , m_n(iPatU)
    , m_m(iPatV)
    , m_su(iSegmU)
    , m_sv(iSegmV)
    , m_Pointanz(iPatU * (iSegmU + 1) * iPatV * (iSegmV + 1))
    , m_Surfmode(iSurfMode)
    , m_cua(iLowU)
    , m_cue(iUpU)
    , m_cva(iLowV)
    , m_cve(iUpV)
    , mp_Image_Texture(0L)
    , m_Pointanz_B(iConsSPOINTS)
    , m_type_C(iConsType)
    , m_Pointanz_C(iConsSPOINTS)
    , m_a(a)
    , m_b(b)
    , m_c(c)
{
    this->initializeMembers();
}
HfT_osg_Plugin01_ParametricSurface::
    HfT_osg_Plugin01_ParametricSurface(double a, int iPatU, int iPatV,
                                       int iSegmU, int iSegmV,
                                       double iLowU, double iUpU,
                                       double iLowV, double iUpV,
                                       SurfMode iSurfMode, ConsType iConsType,
                                       int iConsSPOINTS, Image *image)
    : Geode()
    , m_Surftype(0)
    , m_n(iPatU)
    , m_m(iPatV)
    , m_su(iSegmU)
    , m_sv(iSegmV)
    , m_Pointanz(iPatU * (iSegmU + 1) * iPatV * (iSegmV + 1))
    , m_Surfmode(iSurfMode)
    , m_cua(iLowU)
    , m_cue(iUpU)
    , m_cva(iLowV)
    , m_cve(iUpV)
    , mp_Image_Texture(image)
    , m_Pointanz_B(iConsSPOINTS)
    , m_type_C(iConsType)
    , m_Pointanz_C(iConsSPOINTS)
    ,
    // Default ist Ebene
    m_a(a)
    , m_b(0)
    , m_c(0)
{
    this->initializeMembers();
}
HfT_osg_Plugin01_ParametricSurface::
    HfT_osg_Plugin01_ParametricSurface(double a, double b, int iPatU, int iPatV,
                                       int iSegmU, int iSegmV,
                                       double iLowU, double iUpU,
                                       double iLowV, double iUpV,
                                       SurfMode iSurfMode, ConsType iConsType,
                                       int iConsSPOINTS, Image *image)
    : Geode()
    , m_Surftype(0)
    , m_n(iPatU)
    , m_m(iPatV)
    , m_su(iSegmU)
    , m_sv(iSegmV)
    , m_Pointanz(iPatU * (iSegmU + 1) * iPatV * (iSegmV + 1))
    , m_Surfmode(iSurfMode)
    , m_cua(iLowU)
    , m_cue(iUpU)
    , m_cva(iLowV)
    , m_cve(iUpV)
    , mp_Image_Texture(image)
    , m_Pointanz_B(iConsSPOINTS)
    , m_type_C(iConsType)
    , m_Pointanz_C(iConsSPOINTS)
    , m_a(a)
    , m_b(b)
    , m_c(0)
{
    this->initializeMembers();
}
HfT_osg_Plugin01_ParametricSurface::
    HfT_osg_Plugin01_ParametricSurface(double a, double b, double c, int iPatU, int iPatV,
                                       int iSegmU, int iSegmV,
                                       double iLowU, double iUpU,
                                       double iLowV, double iUpV,
                                       SurfMode iSurfMode, ConsType iConsType,
                                       int iConsSPOINTS, Image *image)
    : Geode()
    , m_Surftype(0)
    , m_n(iPatU)
    , m_m(iPatV)
    , m_su(iSegmU)
    , m_sv(iSegmV)
    , m_Pointanz(iPatU * (iSegmU + 1) * iPatV * (iSegmV + 1))
    , m_Surfmode(iSurfMode)
    , m_cua(iLowU)
    , m_cue(iUpU)
    , m_cva(iLowV)
    , m_cve(iUpV)
    , mp_Image_Texture(image)
    , m_Pointanz_B(iConsSPOINTS)
    , m_type_C(iConsType)
    , m_Pointanz_C(iConsSPOINTS)
    , m_a(a)
    , m_b(b)
    , m_c(c)
{
    this->initializeMembers();
}
HfT_osg_Plugin01_ParametricSurface::~HfT_osg_Plugin01_ParametricSurface()
{

    m_Surftype = 0;
    m_m = 0;
    m_su = 0;
    m_sv = 0;
    m_Pointanz = 0;
    m_Surfmode = SPOINTS;
    m_cua = 0.0;
    m_cue = 0.0;
    m_cva = 0.0;
    m_cve = 0.0;
    m_a = 0;
    m_b = 0;
    m_c = 0;
    m_Pointanz_B = 0;
    ;
    m_type_C = CNOTHING;
    m_Pointanz_C = 0;
    delete mp_ParserSurface;
}

//getter----------------------------------------------------------------
int HfT_osg_Plugin01_ParametricSurface::getSurfType()
{
    return m_Surftype;
}
int HfT_osg_Plugin01_ParametricSurface::getPatchesU()
{
    return m_n;
}

int HfT_osg_Plugin01_ParametricSurface::getPatchesV()
{
    return m_m;
}
int HfT_osg_Plugin01_ParametricSurface::getSegmentsU()
{
    return m_su;
}
int HfT_osg_Plugin01_ParametricSurface::getSegmentsV()
{
    return m_sv;
}

double HfT_osg_Plugin01_ParametricSurface::getLowerBoundU()
{
    return m_cua;
}

double HfT_osg_Plugin01_ParametricSurface::getUpperBoundU()
{
    return m_cue;
}

double HfT_osg_Plugin01_ParametricSurface::getLowerBoundV()
{
    return m_cva;
}

double HfT_osg_Plugin01_ParametricSurface::getUpperBoundV()
{
    return m_cve;
}
double HfT_osg_Plugin01_ParametricSurface::getRadian()
{
    return (m_a);
}
double HfT_osg_Plugin01_ParametricSurface::getLength()
{
    return (m_b);
}
double HfT_osg_Plugin01_ParametricSurface::getHeight()
{
    return (m_c);
}
string HfT_osg_Plugin01_ParametricSurface::getXpara()
{
    return (m_xstr);
}
string HfT_osg_Plugin01_ParametricSurface::getYpara()
{
    return (m_ystr);
}
string HfT_osg_Plugin01_ParametricSurface::getZpara()
{
    return (m_zstr);
}
SurfMode HfT_osg_Plugin01_ParametricSurface::getSurfMode()
{
    return m_Surfmode;
}
Vec2Array *HfT_osg_Plugin01_ParametricSurface::getParameterValues()
{
    return mp_Coords_Geom;
}

Image *HfT_osg_Plugin01_ParametricSurface::getImage()
{
    return mp_Image_Texture;
}
HlParametricSurface3d *HfT_osg_Plugin01_ParametricSurface::getParserSurface()
{
    return (mp_ParserSurface);
}
int HfT_osg_Plugin01_ParametricSurface::getBoundaryPointanz()
{
    return m_Pointanz_B;
}

Vec2Array *HfT_osg_Plugin01_ParametricSurface::getBoundaryParameterValues()
{
    return mp_Geom_B->mp_Coords_Geom;
}

int HfT_osg_Plugin01_ParametricSurface::getConsPointanz()
{
    return m_Pointanz_C;
}

Vec2Array *HfT_osg_Plugin01_ParametricSurface::getConsParameterValues()
{
    return mp_Geom_C->mp_Coords_Geom;
}

ConsType HfT_osg_Plugin01_ParametricSurface::getConsType()
{
    return m_type_C;
}

Image *HfT_osg_Plugin01_ParametricSurface::getAllParam(int &Surftype, SurfMode &smode, int &pU, int &pV,
                                                       int &Su, int &Sv, ConsType &ctype, int &canz)
{
    Surftype = m_Surftype;
    smode = m_Surfmode;
    pU = m_n;
    pV = m_m;
    Su = m_su;
    Sv = m_sv;
    ctype = m_type_C;
    canz = m_Pointanz_C;

    return (mp_Image_Texture);
}
HfT_osg_Plugin01_Cons *HfT_osg_Plugin01_ParametricSurface::getBoundary()
{
    return mp_Geom_B;
}
HfT_osg_Plugin01_Cons *HfT_osg_Plugin01_ParametricSurface::getCons()
{
    return mp_Geom_C;
}
Vec3Array *HfT_osg_Plugin01_ParametricSurface::getPointArray()
{
    return mp_Points_Geom;
}
Vec2Array *HfT_osg_Plugin01_ParametricSurface::getPointArray2()
{
    return mp_Points_Geom2;
}
Vec4Array *HfT_osg_Plugin01_ParametricSurface::getPointArray4()
{
    return mp_Points_Geom4;
}
Vec3Array *HfT_osg_Plugin01_ParametricSurface::getNormalArray()
{
    return mp_Normals_Geom;
}

//setter---------------------------------------------------------------------
void HfT_osg_Plugin01_ParametricSurface::setSurfType(int iSurfType)
{
    m_Surftype = iSurfType;
}

void HfT_osg_Plugin01_ParametricSurface::setPatchesU(int iNumPatches)
{
    if (iNumPatches > 0)
    {
        m_n = iNumPatches;
    }
}

void HfT_osg_Plugin01_ParametricSurface::setPatchesV(int iNumPatches)
{
    if (iNumPatches > 0)
    {
        m_m = iNumPatches;
    }
}
void HfT_osg_Plugin01_ParametricSurface::setSegmentsU(int iNumSegments)
{
    if (iNumSegments > 0)
    {
        m_su = iNumSegments;
    }
}

void HfT_osg_Plugin01_ParametricSurface::setSegmentsV(int iNumSegments)
{
    if (iNumSegments > 0)
    {
        m_sv = iNumSegments;
    }
}

void HfT_osg_Plugin01_ParametricSurface::setLowerBoundU(double iLowU)
{
    m_cua = iLowU;
}

void HfT_osg_Plugin01_ParametricSurface::setUpperBoundU(double iUpU)
{
    m_cue = iUpU;
}

void HfT_osg_Plugin01_ParametricSurface::setLowerBoundV(double iLowV)
{
    m_cva = iLowV;
}
void HfT_osg_Plugin01_ParametricSurface::setUpperBoundV(double iUpV)
{
    m_cve = iUpV;
}
void HfT_osg_Plugin01_ParametricSurface::setRadian(const double &iRadian) //a-Wert
{
    m_a = iRadian;
}
void HfT_osg_Plugin01_ParametricSurface::setLength(const double &iLength) //b-Wert
{
    m_b = iLength;
}
void HfT_osg_Plugin01_ParametricSurface::setHeight(const double &iheight) //c-Wert
{
    m_c = iheight;
}
void HfT_osg_Plugin01_ParametricSurface::setParserSurface(HlParametricSurface3d *parsersurface)
{
    mp_ParserSurface = parsersurface;
}

void HfT_osg_Plugin01_ParametricSurface::setXpara(string xpstr)
{
    m_xstr = xpstr;
    if (mp_ParserSurface)
        mp_ParserSurface->SetFunktionX(xpstr);
}
void HfT_osg_Plugin01_ParametricSurface::setYpara(string ypstr)
{
    m_ystr = ypstr;
    if (mp_ParserSurface)
        mp_ParserSurface->SetFunktionY(ypstr);
}
void HfT_osg_Plugin01_ParametricSurface::setZpara(string zpstr)
{
    m_zstr = zpstr;
    if (mp_ParserSurface)
        mp_ParserSurface->SetFunktionZ(zpstr);
}
void HfT_osg_Plugin01_ParametricSurface::setParametrization(double A, double B, double C, double ua, double ue, double va, double ve, string xpstr, string ypstr, string zpstr)
{
    this->setRadian(A);
    this->setLength(B);
    this->setHeight(C);
    this->setXpara(xpstr);
    this->setYpara(ypstr);
    this->setZpara(zpstr);
    this->setLowerBoundU(ua);
    this->setUpperBoundU(ue);
    this->setLowerBoundV(va);
    this->setUpperBoundV(ve);
}

void HfT_osg_Plugin01_ParametricSurface::setSurfMode(SurfMode iSurfMode)
{
    if (iSurfMode != m_Surfmode)
    {
        m_Surfmode = iSurfMode;
    }
}
void HfT_osg_Plugin01_ParametricSurface::setImage(Image *image)
{
    mp_Image_Texture = image;
    // Gleich noch Image an Statset zuweisen
    mp_StateSet_Geom->setImage(image);
}
void HfT_osg_Plugin01_ParametricSurface::setColor(SurfMode surfmode, Vec4d color)
{
    // color ändern
    m_Surfmode = surfmode;
    if (mp_StateSet_Geom)
    {
        mp_StateSet_Geom->setSurfMode(surfmode);
        mp_StateSet_Geom->createMode(color);
    }
}
Vec4 HfT_osg_Plugin01_ParametricSurface::getColor()
{
    if (mp_StateSet_Geom)
        return mp_StateSet_Geom->getColor();
    else
        return Vec4(1, 1, 1, 1);
}

void HfT_osg_Plugin01_ParametricSurface::setParameterValues(Vec2Array *iPaV)
{
    for (unsigned int i = 0; i < iPaV->size(); i++)
    {
        Vec2d p = (*iPaV)[i];
        (*mp_Coords_Geom)[i].set(p[0], p[1]);
    }
}
void HfT_osg_Plugin01_ParametricSurface::setBoundaryPointanz(int iCp)
{
    m_Pointanz_B = iCp;
}

void HfT_osg_Plugin01_ParametricSurface::setBoundaryParameterValues(Vec2Array *iPaV)
{
    if (mp_Geom_B)
    {
        for (unsigned int i = 0; i < iPaV->size(); i++)
        {
            Vec2d p = (*iPaV)[i];
            (*mp_Geom_B->mp_Coords_Geom)[i].set(p[0], p[1]);
        }
    }
}

void HfT_osg_Plugin01_ParametricSurface::setColorBoundary(Vec4d color)
{
    mp_Geom_B->m_Color = color;
}
void HfT_osg_Plugin01_ParametricSurface::setBoundary(HfT_osg_Plugin01_Cons *boundary)
{
    // Neue Cons wird hinzugefügt und mp_Geom_C = Cons gesetzt
    if (boundary)
    {
        this->computeCons(boundary);
        this->addDrawable(boundary);
    }
    mp_Geom_B = boundary;
}

void HfT_osg_Plugin01_ParametricSurface::replaceBoundary(HfT_osg_Plugin01_Cons *boundary)
{
    if (boundary)
    {
        this->computeCons(boundary);
        if (mp_Geom_B)
            this->replaceDrawable(mp_Geom_B, boundary);
        else
            this->addDrawable(boundary);
    }
    mp_Geom_B = boundary;
}
void HfT_osg_Plugin01_ParametricSurface::setConsPointanz(int iCp)
{
    m_Pointanz_C = iCp;
}

void HfT_osg_Plugin01_ParametricSurface::setConsParameterValues(Vec2Array *iPaV)
{
    if (mp_Geom_C)
    {
        for (unsigned int i = 0; i < iPaV->size(); i++)
        {
            Vec2d p = (*iPaV)[i];
            (*mp_Geom_C->mp_Coords_Geom)[i].set(p[0], p[1]);
        }
    }
}
void HfT_osg_Plugin01_ParametricSurface::setConsType(ConsType iCm)
{
    m_type_C = iCm;
}
//wird nicht benutzt:(?)
void HfT_osg_Plugin01_ParametricSurface::setCons(HfT_osg_Plugin01_Cons *Cons)
{
    // Neue Cons wird hinzugefügt und mp_Geom_C = Cons gesetzt
    if (Cons)
    {
        this->computeCons(Cons);
        this->addDrawable(Cons);
        m_type_C = Cons->m_type;
    }
    else
        m_type_C = CNOTHING;

    mp_Geom_C = Cons;
}
void HfT_osg_Plugin01_ParametricSurface::replaceCons(HfT_osg_Plugin01_Cons *Cons)
{
    if (Cons)
    {
        this->computeCons(Cons);
        if (mp_Geom_C)
            this->replaceDrawable(mp_Geom_C, Cons);
        else
            this->addDrawable(Cons);
        m_type_C = Cons->m_type;
    }
    else
        m_type_C = CNOTHING;

    mp_Geom_C = Cons;
}
// Ende Setter

void HfT_osg_Plugin01_ParametricSurface::initializeMembers()
{
    // Geoms initalisieren und Geode = this zuordnen
    initializeGeoms();
    // Vertex und Normalen Arrays initalisieren und Geoms zuordnen
    // Indexlisten initalisieren und Geoms zuordnen
    initializeGeomsArraysandDrawables();

    // Statsets initialisieren, Zuordnung in
    // den createObjectGeometrieandSurfMode Routinen
    initializeStateSets();

    // Kernel Klasse zur Digitalisierung der Flächen
    initializeParser();

    //insertArray4();
}
void HfT_osg_Plugin01_ParametricSurface::initializeParser()
{
    // Parser Objekt erstellen und Parametrisierungen zuweisen

    mp_ParserSurface = new HlParametricSurface3d();
    mp_ParserSurface->SetFunktionX(m_xstr);
    mp_ParserSurface->SetFunktionY(m_ystr);
    mp_ParserSurface->SetFunktionZ(m_zstr);
}

void HfT_osg_Plugin01_ParametricSurface::initializeGeoms()
{
    mp_Geom = new Geometry();
    mp_Geom_Curvature = new Geometry();
    mp_Geom_PU = new Geometry();
    mp_Geom_PV = new Geometry();
    mp_Geom_C = new HfT_osg_Plugin01_Cons(m_Pointanz_C, m_type_C, m_cua, m_cue, m_cva, m_cve, Vec4(1.f, 0.f, 0.f, 1.f), 16);
    mp_Geom_B = new HfT_osg_Plugin01_Cons(m_Pointanz_B, CNATBOUND, m_cua, m_cue, m_cva, m_cve, Vec4(0.2f, 1.f, 0.f, 1.f), 16);

    this->addDrawable(mp_Geom);
    this->addDrawable(mp_Geom_Curvature);
    this->addDrawable(mp_Geom_PU);
    this->addDrawable(mp_Geom_PV);
    this->addDrawable(mp_Geom_C);
    this->addDrawable(mp_Geom_B);
}
void HfT_osg_Plugin01_ParametricSurface::initializeGeomsArraysandDrawables()
{
    initializeVertexArrays();
    initializeDrawElements();
}
void HfT_osg_Plugin01_ParametricSurface::initializeVertexArrays()
{
    // Anzahl der Flächenpunkte: (0 -> m_su) x (0 -> m_sv) je Patch
    mp_Points_Geom = new Vec3Array(m_Pointanz);
    mp_Normals_Geom = new Vec3Array(m_Pointanz);

    mp_Geom->setVertexArray(mp_Points_Geom);
    mp_Geom->setNormalArray(mp_Normals_Geom);
    mp_Geom->setNormalBinding(Geometry::BIND_PER_VERTEX);

    mp_Geom_Curvature->setVertexArray(mp_Points_Geom);
    mp_Geom_Curvature->setNormalArray(mp_Normals_Geom);
    mp_Geom_Curvature->setNormalBinding(Geometry::BIND_PER_VERTEX);
    mp_CurvatureColorArray = new Vec4Array(m_Pointanz);
    mp_Geom_Curvature->setColorArray(mp_CurvatureColorArray);
    mp_Geom_Curvature->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    mp_Geom_PU->setVertexArray(mp_Points_Geom);
    mp_Geom_PU->setNormalArray(mp_Normals_Geom);
    mp_Geom_PU->setNormalBinding(Geometry::BIND_PER_VERTEX);
    mp_Geom_PV->setVertexArray(mp_Points_Geom);
    mp_Geom_PV->setNormalArray(mp_Normals_Geom);
    mp_Geom_PV->setNormalBinding(Geometry::BIND_PER_VERTEX);

    mp_Coords_Geom = new Vec2Array(mp_Points_Geom->size());
    mp_TexCoords_Geom = new Vec2Array(mp_Points_Geom->size());
    mp_Geom->setTexCoordArray(0, mp_TexCoords_Geom);

    mp_GC_Geom = new DoubleArray(mp_Coords_Geom->size());
    mp_MC_Geom = new DoubleArray(mp_Coords_Geom->size());
}
void HfT_osg_Plugin01_ParametricSurface::initializeDrawElements()
{
    mp_PointEdges_Geom = new DrawElementsUInt(PrimitiveSet::POINTS, 0);
    mp_TriangleEdges_Geom = new DrawElementsUInt(PrimitiveSet::TRIANGLES, 0);
    mp_QuadEdges_Geom = new DrawElementsUInt(PrimitiveSet::QUADS, 0);
    mp_LineEdges_Geom_PU = new DrawElementsUInt(PrimitiveSet::LINES, 0);
    mp_LineEdges_Geom_PV = new DrawElementsUInt(PrimitiveSet::LINES, 0);
    // Leider kann man die DrawElements erst dann der Geometry zuweisen
    // wenn sie gefüllt sind, warum auch immer ???
    // Daher Zuweisung in den CreateMode Routinen
}
void HfT_osg_Plugin01_ParametricSurface::initializeStateSets()
{
    mp_StateSet_Geom = new HfT_osg_StateSet(m_Surfmode, mp_Image_Texture, m_Pointanz);
    mp_Geom->setStateSet(mp_StateSet_Geom);

    mp_StateSet_Geom_Curvature = new HfT_osg_StateSet(SGAUSS);
    mp_Geom_Curvature->setStateSet(mp_StateSet_Geom_Curvature);

    mp_StateSet_Geom_PU = new HfT_osg_StateSet(SLINES);
    mp_Geom_PU->setStateSet(mp_StateSet_Geom_PU);
    mp_StateSet_Geom_PV = new HfT_osg_StateSet(SLINES);
    mp_Geom_PV->setStateSet(mp_StateSet_Geom_PV);
}
void HfT_osg_Plugin01_ParametricSurface::computeParameterValues()
{
    int k = 0;
    // Flächenparameter (u_i,v_j) erstmal äquidistant
    for (int j = 1; j <= m_m; j++)
    {
        double pva = m_cva + ((j - 1) * (m_cve - m_cva)) / m_m;
        double pve = m_cva + (j * (m_cve - m_cva)) / m_m;
        for (int i = 1; i <= m_n; i++)
        {
            double pua = m_cua + ((i - 1) * (m_cue - m_cua)) / m_n;
            double pue = m_cua + (i * (m_cue - m_cua)) / m_n;

            for (int jp = 0; jp <= m_sv; jp++)
            {
                double v = pva + (pve - pva) * jp / m_sv;
                double vt = 1.0 * jp / m_sv;
                for (int ip = 0; ip <= m_su; ip++)
                {
                    double u = pua + (pue - pua) * ip / m_su;
                    double ut = 1.0 * ip / m_su;

                    // Feld muss in initializeMembers() allokiert sein
                    k = computeIndex(ip, jp, i, j);
                    (*mp_Coords_Geom)[k].set(u, v);
                    (*mp_TexCoords_Geom)[k].set(ut, vt);
                }
            }
        }
    }
}

void HfT_osg_Plugin01_ParametricSurface::createGeometryandMode()
{ // Punkt, Normalen und Darstellungsmodus werden erzeugt
    this->computeParameterValues();
    this->createGeometry();
    this->createMode();
}
void HfT_osg_Plugin01_ParametricSurface::quicksort(Vec4Array *s, int left, int right)
{
    if (left < right)
    {
        // pi Index des Pivotelements
        int pi = partition(s, left, right);
        quicksort(s, left, pi - 1);
        quicksort(s, pi + 1, right);
    }
}
int HfT_osg_Plugin01_ParametricSurface::partition(Vec4Array *s, int left, int right)
{
    int x = (*s)[right].x(); // wähle Pivot-Element aus
    int l = left;
    int r = right - 1;
    while (true)
    {
        while (l <= r && (*s)[l].x() < x)
            l++;
        while (l <= r && (*s)[r].x() > x)
            r--;
        if (l < r)
        {
            int t1 = (*s)[l].x();
            (*s)[l].x() = (*s)[r].x();
            (*s)[r].x() = t1;
            int t2 = (*s)[l].y();
            (*s)[l].y() = (*s)[r].y();
            (*s)[r].y() = t2;
            int t3 = (*s)[l].z();
            (*s)[l].z() = (*s)[r].z();
            (*s)[r].z() = t3;
            int t4 = (*s)[l].w();
            (*s)[l].w() = (*s)[r].w();
            (*s)[r].w() = t4;
        }
        else
        {
            int t1 = (*s)[l].x();
            (*s)[l].x() = (*s)[right].x();
            (*s)[right].x() = t1;
            int t2 = (*s)[l].y();
            (*s)[l].y() = (*s)[right].y();
            (*s)[right].y() = t2;
            int t3 = (*s)[l].z();
            (*s)[l].z() = (*s)[right].z();
            (*s)[right].z() = t3;
            int t4 = (*s)[l].w();
            (*s)[l].w() = (*s)[right].w();
            (*s)[right].w() = t4;
            return l; // l ist die Position des Pivot-Elements
        }
    }
}

void HfT_osg_Plugin01_ParametricSurface::createGeometry()
{
    double u, v;

    /*mp_Points_Geom4 = new Vec4Array(m_Pointanz);*/

    // Hier jetzt Parametrisierung setzen
    setXpara(m_xstr);
    setYpara(m_ystr);
    setZpara(m_zstr);
    mp_ParserSurface->SetA(m_a);
    mp_ParserSurface->SetB(m_b);
    mp_ParserSurface->SetC(m_c);

    // Fläche digitalisieren
    for (unsigned int i = 0; i < mp_Coords_Geom->size(); i++)
    {
        u = mp_Coords_Geom->at(i)[0];
        v = mp_Coords_Geom->at(i)[1];
        digitalize(u, v);
        (*mp_Points_Geom)[i].set(m_Point);
        (*mp_Normals_Geom)[i].set(m_Normal);
        std::string s = HfT_double_to_string(m_GC);
        if (s == "-1.#IND")
        {
            //fprintf(stderr,"Singularitaet in %f %f \n",u,v);
            m_GC = 0.;
        }
        if (abs(m_GC) > 10000)
            m_GC = 0;
        (*mp_GC_Geom)[i] = m_GC;
        //fprintf(stderr,"GaussKruemmung in %f %f ist %f \n",u,v,m_GC);
        s = HfT_double_to_string(m_MC);
        if (s == "-1.#IND")
        {
            //fprintf(stderr,"Singularitaet in %f %f \n",u,v);
            m_MC = 0.;
        }
        if (abs(m_MC) > 10000)
            m_MC = 0;
        // Achtung hier -m_MC zuordnen bis hanspeter Fehler findet
        (*mp_MC_Geom)[i] = -m_MC;
    }
    // Äußeren Rand extra generieren
    if (mp_Geom_B && (!mp_Geom_B->m_isSet3D))
    {
        mp_Geom_B->setParameterValues(m_cua, m_cue, m_cva, m_cve);
        computeCons(mp_Geom_B);
    }

    // Flächenkurven, wenn nötig, generieren
    if (mp_Geom_C && (!mp_Geom_C->m_isSet3D))
        computeCons(mp_Geom_C);

    ////Punkte in Vec4array umspeichern für einzelnen Zugriff spaeter in step 9
    //for(int i = 0; i<m_Pointanz; i++){
    //	(*mp_Points_Geom4)[i].x() = (*mp_Points_Geom)[i].x();
    //	(*mp_Points_Geom4)[i].y() = (*mp_Points_Geom)[i].y();
    //	(*mp_Points_Geom4)[i].z() = (*mp_Points_Geom)[i].z();
    //	(*mp_Points_Geom4)[i].w() = (*mp_Points_Geom)[i].z()/*(float) i*/;
    //}
    //int rechts = (*mp_Points_Geom4).size() -1;
    //quicksort(mp_Points_Geom4, 0, rechts);
}
//Darstellungsweise der Flaeche
void HfT_osg_Plugin01_ParametricSurface::createMode()
{
    switch (this->m_Surfmode)
    {
    //Pointcloud
    case SPOINTS:
    {
        this->computePointEdges();
        mp_Geom->addPrimitiveSet(mp_PointEdges_Geom);
        mp_StateSet_Geom->setSurfMode(m_Surfmode);
        mp_StateSet_Geom->createPointMode();
    }
    break;
    //Paralines
    case SLINES:
    {
        this->computeParaLineEdges();
        mp_Geom_PU->addPrimitiveSet(mp_LineEdges_Geom_PU);
        mp_StateSet_Geom_PU->setSurfMode(m_Surfmode);
        mp_StateSet_Geom_PU->createLineMode(Vec4(0.8f, 0.0f, 0.0f, 1.0f));
        mp_Geom_PV->addPrimitiveSet(mp_LineEdges_Geom_PV);
        mp_StateSet_Geom_PV->setSurfMode(m_Surfmode);
        mp_StateSet_Geom_PV->createLineMode(Vec4(0., 0., 1., 1.));
    }
    break;
    //Triangles
    case STRIANGLES:
    {
        this->computeTriangleEdges();
        mp_Geom->addPrimitiveSet(mp_TriangleEdges_Geom);
        mp_StateSet_Geom->setSurfMode(m_Surfmode);
        mp_StateSet_Geom->createTriangleMode();
    }
    break;
    //Quads
    case SQUADS:
    {
        this->computeQuadEdges();
        mp_Geom->addPrimitiveSet(mp_QuadEdges_Geom);
        mp_StateSet_Geom->setSurfMode(m_Surfmode);
        mp_StateSet_Geom->createQuadMode();
    }
    break;
    //Shade mode
    case SSHADE:
    {
        this->computeParaLineEdges();
        mp_Geom_PU->addPrimitiveSet(mp_LineEdges_Geom_PU);
        mp_StateSet_Geom_PU->setSurfMode(m_Surfmode);
        mp_StateSet_Geom_PU->createLineMode(Vec4(0.8f, 0.0f, 0.0f, 1.0f));
        mp_Geom_PV->addPrimitiveSet(mp_LineEdges_Geom_PV);
        mp_StateSet_Geom_PV->setSurfMode(m_Surfmode);
        mp_StateSet_Geom_PV->createLineMode(Vec4(0., 0., 1., 1.));
        this->computeQuadEdges();
        mp_Geom->addPrimitiveSet(mp_QuadEdges_Geom);
        mp_StateSet_Geom->setSurfMode(m_Surfmode);
        mp_StateSet_Geom->createShadeMode(Vec4(0.8f, 0.8f, 0.3f, 1.0f));
    }
    break;
    //Texture mode
    case STEXTURE:
    {
        this->computeQuadEdges();
        mp_Geom->addPrimitiveSet(mp_QuadEdges_Geom);
        mp_StateSet_Geom->setSurfMode(m_Surfmode);
        mp_StateSet_Geom->createTextureMode();
    }
    break;
    //Transparent mode
    case STRANSPARENT:
    {
        this->computeQuadEdges();
        mp_Geom->addPrimitiveSet(mp_QuadEdges_Geom);
        mp_StateSet_Geom->setSurfMode(m_Surfmode);
        mp_StateSet_Geom->createTransparentMode();
    }
    break;
    //Gauss curvature mode
    case SGAUSS:
    {
        this->computeTriangleEdges();
        mp_Geom_Curvature->addPrimitiveSet(mp_TriangleEdges_Geom);
        updateCurvatureColorArray(1);
        mp_StateSet_Geom_Curvature->setSurfMode(m_Surfmode);
        mp_StateSet_Geom_Curvature->createGaussCurvatureMode();
    }
    break;
    case SMEAN:
    {
        this->computeTriangleEdges();
        mp_Geom_Curvature->addPrimitiveSet(mp_TriangleEdges_Geom);
        updateCurvatureColorArray(0);
        mp_StateSet_Geom_Curvature->setSurfMode(m_Surfmode);
        mp_StateSet_Geom_Curvature->createMeanCurvatureMode();
    }
    break;
    default:
        break;
    }
}
void HfT_osg_Plugin01_ParametricSurface::createMode(Vec4 color)
{
    switch (this->m_Surfmode)
    {
    //Pointcloud
    case SPOINTS:
    {
        this->computePointEdges();
        mp_Geom->addPrimitiveSet(mp_PointEdges_Geom);
        mp_StateSet_Geom->setSurfMode(m_Surfmode);
        mp_StateSet_Geom->createPointMode(color);
    }
    break;
    //Paralines
    case SLINES:
    {
        this->computeParaLineEdges();
        mp_Geom_PU->addPrimitiveSet(mp_LineEdges_Geom_PU);
        mp_StateSet_Geom_PU->setSurfMode(m_Surfmode);
        mp_StateSet_Geom_PU->createLineMode(color);
        mp_Geom_PV->addPrimitiveSet(mp_LineEdges_Geom_PV);
        mp_StateSet_Geom_PV->setSurfMode(m_Surfmode);
        mp_StateSet_Geom_PV->createLineMode(color);
    }
    break;
    //Triangles
    case STRIANGLES:
    {
        this->computeTriangleEdges();
        mp_Geom->addPrimitiveSet(mp_TriangleEdges_Geom);
        mp_StateSet_Geom->setSurfMode(m_Surfmode);
        mp_StateSet_Geom->createTriangleMode(color);
    }
    break;
    //Quads
    case SQUADS:
    {
        this->computeQuadEdges();
        mp_Geom->addPrimitiveSet(mp_QuadEdges_Geom);
        mp_StateSet_Geom->setSurfMode(m_Surfmode);
        mp_StateSet_Geom->createQuadMode(color);
    }
    break;
    //Shade mode
    case SSHADE:
    {
        this->computeParaLineEdges();
        mp_Geom_PU->addPrimitiveSet(mp_LineEdges_Geom_PU);
        mp_StateSet_Geom_PU->setSurfMode(m_Surfmode);
        mp_StateSet_Geom_PU->createLineMode(color);
        mp_Geom_PV->addPrimitiveSet(mp_LineEdges_Geom_PV);
        mp_StateSet_Geom_PV->setSurfMode(m_Surfmode);
        mp_StateSet_Geom_PV->createLineMode(color);
        this->computeQuadEdges();
        mp_Geom->addPrimitiveSet(mp_QuadEdges_Geom);
        mp_StateSet_Geom->setSurfMode(m_Surfmode);
        mp_StateSet_Geom->createShadeMode(color);
    }
    break;
    //Texture mode
    case STEXTURE:
    {
        this->computeQuadEdges();
        mp_Geom->addPrimitiveSet(mp_QuadEdges_Geom);
        mp_StateSet_Geom->setSurfMode(m_Surfmode);
        mp_StateSet_Geom->createTextureMode(color);
    }
    break;
    //Transparent mode
    case STRANSPARENT:
    {
        this->computeQuadEdges();
        mp_Geom->addPrimitiveSet(mp_QuadEdges_Geom);
        mp_StateSet_Geom->setSurfMode(m_Surfmode);
        mp_StateSet_Geom->createTransparentMode(color);
    }
    break;
    //Gauss curvature mode
    case SGAUSS:
    {
        this->computeTriangleEdges();
        mp_Geom_Curvature->addPrimitiveSet(mp_TriangleEdges_Geom);
        updateCurvatureColorArray(1);
        mp_StateSet_Geom_Curvature->setSurfMode(m_Surfmode);
        mp_StateSet_Geom_Curvature->createGaussCurvatureMode();
    }
    break;
    case SMEAN:
    {
        this->computeTriangleEdges();
        mp_Geom_Curvature->addPrimitiveSet(mp_TriangleEdges_Geom);
        updateCurvatureColorArray(0);
        mp_StateSet_Geom_Curvature->setSurfMode(m_Surfmode);
        mp_StateSet_Geom_Curvature->createMeanCurvatureMode();
    }
    break;
    default:
        break;
    }
}
void HfT_osg_Plugin01_ParametricSurface::computeTriangleEdges()
{
    for (int j = 1; j <= m_m; j++)
    {
        for (int i = 1; i <= m_n; i++)
        {
            computeTriangleEdges(i, j);
        }
    }
}
void HfT_osg_Plugin01_ParametricSurface::computeTriangleEdges(int pU, int pV)
{
    int k = 0;

    // Indices of Triangles in patch(pU,pV)
    for (int j = 0; j < m_sv; j++)
    {
        for (int i = 0; i < m_su; i++)
        {
            k = computeIndex(i, j, pU, pV);
            mp_TriangleEdges_Geom->push_back(k);
            k = computeIndex(i + 1, j, pU, pV);
            mp_TriangleEdges_Geom->push_back(k);
            k = computeIndex(i, j + 1, pU, pV);
            mp_TriangleEdges_Geom->push_back(k);

            k = computeIndex(i + 1, j, pU, pV);
            mp_TriangleEdges_Geom->push_back(k);
            k = computeIndex(i + 1, j + 1, pU, pV);
            mp_TriangleEdges_Geom->push_back(k);
            k = computeIndex(i, j + 1, pU, pV);
            mp_TriangleEdges_Geom->push_back(k);
        }
    }
}

void HfT_osg_Plugin01_ParametricSurface::computeQuadEdges()
{
    for (int j = 1; j <= m_m; j++)
    {
        for (int i = 1; i <= m_n; i++)
        {
            computeQuadEdges(i, j);
        }
    }
}
void HfT_osg_Plugin01_ParametricSurface::computeQuadEdges(int pU, int pV)
{
    int k = 0;
    // Indices of Quads in patch(pU,pV)
    for (int j = 0; j < m_sv; j++)
    {
        for (int i = 0; i < m_su; i++)
        {
            k = computeIndex(i, j, pU, pV);
            mp_QuadEdges_Geom->push_back(k);
            k = computeIndex(i + 1, j, pU, pV);
            mp_QuadEdges_Geom->push_back(k);
            k = computeIndex(i + 1, j + 1, pU, pV);
            mp_QuadEdges_Geom->push_back(k);
            k = computeIndex(i, j + 1, pU, pV);
            mp_QuadEdges_Geom->push_back(k);
        }
    }
}
void HfT_osg_Plugin01_ParametricSurface::computePointEdges()
{
    for (int j = 1; j <= m_m; j++)
    {
        for (int i = 1; i <= m_n; i++)
        {
            computePointEdges(i, j);
        }
    }
}
void HfT_osg_Plugin01_ParametricSurface::computePointEdges(int pU, int pV)
{
    int k = 0;
    // Indices of SPOINTS in patch(pU,pV)
    for (int j = 0; j <= m_sv; j++)
    {
        for (int i = 0; i <= m_su; i++)
        {
            k = computeIndex(i, j, pU, pV);
            mp_PointEdges_Geom->push_back(k);
        }
    }
}

void HfT_osg_Plugin01_ParametricSurface::computeParaLineEdges()
{
    for (int j = 1; j <= m_m; j++)
    {
        for (int i = 1; i <= m_n; i++)
        {
            computeParaLineEdges(i, j);
        }
    }
}
void HfT_osg_Plugin01_ParametricSurface::computeParaLineEdges(int pU, int pV)
{
    int k = 0;
    // Indices of u-Lines in patch(pU,pV)
    for (int j = 0; j <= m_sv; j++)
    {
        for (int i = 0; i < m_su; i++)
        {
            k = computeIndex(i, j, pU, pV);
            mp_LineEdges_Geom_PU->push_back(k);
            k = computeIndex(i + 1, j, pU, pV);
            mp_LineEdges_Geom_PU->push_back(k);
        }
    }
    // Indices of v-Lines in patch(pU,pV)
    for (int j = 0; j < m_sv; j++)
    {
        for (int i = 0; i <= m_su; i++)
        {
            k = computeIndex(i, j, pU, pV);
            mp_LineEdges_Geom_PV->push_back(k);
            k = computeIndex(i, j + 1, pU, pV);
            mp_LineEdges_Geom_PV->push_back(k);
        }
    }
}

int HfT_osg_Plugin01_ParametricSurface::computeIndex(int i, int j, int pU, int pV)
{
    // Routine berechnet Index k für Punkt P(u_i,v_j) in Patch (pU,pV)

    // Indizes (i,j) der Flächenpunkte P(u_i,v_j) kommen der Reihenfolge
    // hintereinander in Index-Vektor
    // Reihenfolge des Abdigitalisierens:
    // In Patch:  Punkte werden immer zuerst in u-Richtung erfasst
    // Auf der Fläche: In "Richtung" pU-Patches --> dann pV Patches
    // Bsp. Für Patch(1,1) gilt:
    // (0,0) -> 0,  (1,0) -> 1, ..., (m_su,0) -> m_su
    // (0,1) -> m_su+1, ...

    int k = 0;
    // Anzahl der Punkte je Patch:
    int anzp = (m_sv + 1) * (m_su + 1);
    // Startindex in Patch (pU,pV) des linken unteren Punktes ianf:
    int ianf = anzp * ((pU - 1) + m_n * (pV - 1));

    k = (ianf + i) + j * (m_su + 1);

    return k;
}
//wird nicht benutzt
// Recompute Methoden
//void HfT_osg_Plugin01_ParametricSurface ::recomputeGeometryandMode()
//{
//	// Falls auch neuer SurfMode, dann muss er vorher gesetzt werden
//	this -> computeParameterValues();
//	this -> recomputeGeometry();
//	this -> recomputeMode(m_Surfmode);
//}
void HfT_osg_Plugin01_ParametricSurface::recomputeGeometry()
{ // Neue Flächengeometrie inkl. Rand und Flächenkurve
    this->clearGeomsArraysandDrawables();
    this->initializeGeomsArraysandDrawables();
    this->computeParameterValues();

    HfT_osg_Plugin01_Cons *cons = this->getCons();
    if (cons)
    {
        HfT_osg_Plugin01_Cons *newcons = new HfT_osg_Plugin01_Cons(*cons);
        this->replaceCons(newcons);
    }
    HfT_osg_Plugin01_Cons *boundary = this->getBoundary();
    if (boundary)
    {
        HfT_osg_Plugin01_Cons *newboundary = new HfT_osg_Plugin01_Cons(*boundary);
        this->replaceBoundary(newboundary);
    }

    this->createGeometry();
    this->createMode();
}
float HfT_osg_Plugin01_ParametricSurface::abstand(osg::Vec3 startPickPos, int i)
{
    float x = (*mp_Points_Geom)[i].x() - startPickPos.x();
    float y = (*mp_Points_Geom)[i].y() - startPickPos.y();
    float z = (*mp_Points_Geom)[i].z() - startPickPos.z();

    float abst = (float)sqrt(x * x + y * y + z * z);
    return abst;
}
void HfT_osg_Plugin01_ParametricSurface::insertArray2(osg::Vec3 startPickPos)
{
    mp_Points_Geom2 = new Vec2Array(m_Pointanz);
    //Punkte in Vec4array umspeichern für einzelnen Zugriff spaeter in step 9
    for (int i = 0; i < m_Pointanz; i++)
    {
        (*mp_Points_Geom2)[i].x() = abstand(startPickPos, i);
        (*mp_Points_Geom2)[i].y() = (float)i;
    }
}
void HfT_osg_Plugin01_ParametricSurface::insertArray4()
{
    mp_Points_Geom4 = new Vec4Array(m_Pointanz);

    //Punkte in Vec4array umspeichern für einzelnen Zugriff spaeter in step 9
    for (int i = 0; i < m_Pointanz; i++)
    {
        (*mp_Points_Geom4)[i].x() = (*mp_Points_Geom)[i].x();
        (*mp_Points_Geom4)[i].y() = (*mp_Points_Geom)[i].y();
        (*mp_Points_Geom4)[i].z() = (*mp_Points_Geom)[i].z();
        (*mp_Points_Geom4)[i].w() = (float)i;
    }
}
void HfT_osg_Plugin01_ParametricSurface::recomputeGeometry(int iPatU, int iPatV, int iSegmU, int iSegmV)
{
    // Neue Flächengeometrie ohne Rand und Flächenkurve
    m_Pointanz = iPatU * (iSegmU + 1) * iPatV * (iSegmV + 1); //Anzahl Flächenpunkte
    m_n = iPatU;
    m_m = iPatV;
    m_su = iSegmU;
    m_sv = iSegmV;

    this->clearGeomsArraysandDrawables();
    this->initializeGeomsArraysandDrawables();

    this->computeParameterValues();
    this->createGeometry();
    this->createMode();
}

void HfT_osg_Plugin01_ParametricSurface::recomputeMode(SurfMode iSurfMode)
{
    this->clearMode();
    m_Surfmode = iSurfMode;
    this->createMode();
}
void HfT_osg_Plugin01_ParametricSurface::recomputeMode(SurfMode iSurfMode, Image *image)
{
    this->clearMode();
    m_Surfmode = iSurfMode;
    // Image wird auch an Stateset durchgereicht
    this->setImage(image);
    this->createMode();
}
void HfT_osg_Plugin01_ParametricSurface::recomputeMode(SurfMode iSurfMode, Vec4 color, Image *image)
{
    this->clearMode();
    m_Surfmode = iSurfMode;
    // Image wird auch an Stateset durchgereicht
    this->setImage(image);
    this->createMode(color);
}
void HfT_osg_Plugin01_ParametricSurface::computeCons(HfT_osg_Plugin01_Cons *cons)
{
    double u, v;
    setXpara(m_xstr);
    setYpara(m_ystr);
    setZpara(m_zstr);
    mp_ParserSurface->SetA(m_a);
    mp_ParserSurface->SetB(m_b);
    mp_ParserSurface->SetC(m_c);
    // Hier wird Cons  in der Parameterebene bereechnet
    if (cons && (cons->m_type != CNOTHING))
    {
        for (unsigned int i = 0; i < cons->mp_Coords_Geom->size(); i++)
        {
            u = cons->mp_Coords_Geom->at(i)[0];
            v = cons->mp_Coords_Geom->at(i)[1];
            digitalize(u, v);
            (*cons->mp_Points_Geom)[i].set(m_Point);
            (*cons->mp_Normals_Geom)[i].set(m_Normal);
        }
    }
}
void HfT_osg_Plugin01_ParametricSurface::clearGeoms()
{
    removeDrawables(getNumDrawables());
}
void HfT_osg_Plugin01_ParametricSurface::clearGeomsArraysandDrawables()
{
    clearVertexArrays();
    clearDrawElements();
}
void HfT_osg_Plugin01_ParametricSurface::clearVertexArrays()
{
    mp_Points_Geom->clear();
    //mp_Points_Geom4 ->clear();
    mp_Normals_Geom->clear();
    mp_CurvatureColorArray->clear();
    if (mp_TexCoords_Geom)
        mp_TexCoords_Geom->clear();
}
void HfT_osg_Plugin01_ParametricSurface::clearDrawElements()
{

    if (m_Surfmode == SSHADE)
    {
        mp_Geom->removePrimitiveSet(0);
        mp_Geom_PU->removePrimitiveSet(0);
        mp_Geom_PV->removePrimitiveSet(0);
    }
    else if (m_Surfmode == SLINES)
    {
        mp_Geom_PU->removePrimitiveSet(0);
        mp_Geom_PV->removePrimitiveSet(0);
    }
    else if (m_Surfmode == SGAUSS)
    {
        mp_Geom_Curvature->removePrimitiveSet(0);
    }
    else if (m_Surfmode == SMEAN)
    {
        mp_Geom_Curvature->removePrimitiveSet(0);
    }
    else
    {
        mp_Geom->removePrimitiveSet(0);
    }
    // Alle Member-Felder auch bereinigen
    mp_QuadEdges_Geom->clear();
    mp_PointEdges_Geom->clear();
    mp_TriangleEdges_Geom->clear();
    mp_LineEdges_Geom_PU->clear();
    mp_LineEdges_Geom_PV->clear();
}

void HfT_osg_Plugin01_ParametricSurface::clearMode()
{
    this->clearDrawElements();
    mp_StateSet_Geom->clearMode();
    mp_StateSet_Geom_PU->clearMode();
    mp_StateSet_Geom_PV->clearMode();
    mp_StateSet_Geom_Curvature->clearMode();
}

Vec4d HfT_osg_Plugin01_ParametricSurface::Curvature2Color(double curvature, double curvaturemin, double curvaturemax)
{
    Vec4d color;
    curvature *= 1000;
    curvaturemin *= 1000;
    curvaturemax *= 1000;

    double rotf = rot(curvature, curvaturemax);
    double gruenf = gruen(curvature, curvaturemin);
    double blauf = blau();

    color.set(rotf, gruenf, blauf, 1.);

    //fprintf(stderr,"r,g,b = %f %f %f \n",color[0],color[1],color[2]);
    return (color);
}
void HfT_osg_Plugin01_ParametricSurface::updateCurvatureColorArray(int flag)
{
    // Routine noch nach StateSet Klasse verschieben
    double curvmin = 100000, curvmax = -100000;
    Vec4d curvaturecolor;

    // Gauss oder mittlerer Krümmung Farbe zuweisen
    if (flag) // Gausskrümmung
    {
        for (unsigned int i = 0; i < mp_GC_Geom->size(); i++)
        {
            double curvature = mp_GC_Geom->at(i);
            if (curvature < curvmin)
                curvmin = curvature;
            if (curvature > curvmax)
                curvmax = curvature;
            //fprintf(stderr,"Kruemmung ist: %f \n",curvature);
        }
        // Farben aus Krümmungswerten generieren
        fprintf(stderr, "minKruemmung, maxKruemmung sind: %f %f \n", curvmin, curvmax);
        for (unsigned int i = 0; i < mp_GC_Geom->size(); i++)
        {
            double curvature = mp_GC_Geom->at(i);
            curvaturecolor = Curvature2Color(curvature, curvmin, curvmax);
            (*mp_CurvatureColorArray)[i].set(curvaturecolor[0], curvaturecolor[1], curvaturecolor[2], curvaturecolor[3]);
        }
    }
    else //Mittlere Krümmung
    {
        for (unsigned int i = 0; i < mp_MC_Geom->size(); i++)
        {
            double curvature = mp_MC_Geom->at(i);
            if (curvature < curvmin)
                curvmin = curvature;
            if (curvature > curvmax)
                curvmax = curvature;
            //fprintf(stderr,"Kruemmung ist: %f \n",curvature);
        }
        // Farben aus Krümmungswerten generieren
        fprintf(stderr, "minKruemmung, maxKruemmung sind: %f %f \n", curvmin, curvmax);
        for (unsigned int i = 0; i < mp_MC_Geom->size(); i++)
        {
            double curvature = mp_MC_Geom->at(i);
            curvaturecolor = Curvature2Color(curvature, curvmin, curvmax);
            (*mp_CurvatureColorArray)[i].set(curvaturecolor[0], curvaturecolor[1], curvaturecolor[2], curvaturecolor[3]);
        }
    }
}
void HfT_osg_Plugin01_ParametricSurface::digitalize(double u, double v)
{
    HlVector F = mp_ParserSurface->f(u, v);
    m_Point = Vec3(F.mX, F.mY, F.mZ);
    HlVector N = mp_ParserSurface->nvek(u, v);
    N = N.normiert();
    m_Normal = Vec3(N.mX, N.mY, N.mZ);
    m_GC = mp_ParserSurface->K(u, v);
    m_MC = mp_ParserSurface->H(u, v);

    /* Mal für später
	HlVector Fu = mp_ParserSurface->dfu(u,v);
	HlVector Fv = mp_ParserSurface->dfv(u,v);
	HlVector Fuv = mp_ParserSurface->dfuv(u,v);
	HlVector Fuu = mp_ParserSurface->dfuu(u,v);
	HlVector Fvv = mp_ParserSurface->dfvv(u,v);
	HlVector N = Fu.kreuzprodukt(Fv);
	
	m_PointU = Vec3(Fu.mX, Fu.mY, Fu.mZ);
	m_PointV = Vec3(Fv.mX, Fv.mY, Fv.mZ);
    m_PointUU = Vec3(Fuu.mX, Fuu.mY, Fuu.mZ);
	m_PointUV = Vec3(Fuv.mX, Fuv.mY, Fuv.mZ);
	m_PointVV = Vec3(Fvv.mX, Fvv.mY, Fvv.mZ);
	*/
}

// Fr. Uebele
Vec3dArray *HfT_osg_Plugin01_ParametricSurface::computeSchnittkurve(double u, double v, int i)
{
    setXpara(m_xstr);
    setYpara(m_ystr);
    setZpara(m_zstr);
    mp_ParserSurface->SetA(m_a);
    mp_ParserSurface->SetB(m_b);
    mp_ParserSurface->SetC(m_c);

    return digitalize(u, v, i);
}
// Fr. Uebele

// Fr. Uebele
Vec3dArray *HfT_osg_Plugin01_ParametricSurface::digitalize(double u, double v, int i)
{
    Vec3dArray *feld = new Vec3dArray();
    if (i == 1)
    {
        HlVector F = mp_ParserSurface->f(u, v);
        Vec3 pkt = Vec3(F.mX, F.mY, F.mZ);
        HlVector Fu = mp_ParserSurface->dfu(u, v);
        Vec3 pointU = Vec3(Fu.mX, Fu.mY, Fu.mZ);
        HlVector Fv = mp_ParserSurface->dfv(u, v);
        Vec3 pointV = Vec3(Fv.mX, Fv.mY, Fv.mZ);
        HlVector Fuv = mp_ParserSurface->dfuv(u, v);
        Vec3 pointUV = Vec3(Fuv.mX, Fuv.mY, Fuv.mZ);
        HlVector Fuu = mp_ParserSurface->dfuu(u, v);
        Vec3 pointUU = Vec3(Fuu.mX, Fuu.mY, Fuu.mZ);
        HlVector Fvv = mp_ParserSurface->dfvv(u, v);
        Vec3 pointVV = Vec3(Fvv.mX, Fvv.mY, Fvv.mZ);
        HlVector N = Fu.kreuzprodukt(Fv);
        Vec3 pointN = Vec3(N.mX, N.mY, N.mZ);

        feld->push_back(pkt);
        feld->push_back(pointU);
        feld->push_back(pointV);
        feld->push_back(pointUV);
        feld->push_back(pointUU);
        feld->push_back(pointVV);
        feld->push_back(pointN);
    }
    else
    {
        double K = mp_ParserSurface->K(u, v);
        double H = mp_ParserSurface->H(u, v);
        double E = mp_ParserSurface->E(u, v);
        double F = mp_ParserSurface->F(u, v);
        double G = mp_ParserSurface->G(u, v);
        double L = mp_ParserSurface->L(u, v);
        double M = mp_ParserSurface->M(u, v);
        double N = mp_ParserSurface->N(u, v);
        double EG_FF = mp_ParserSurface->EG_FF(u, v);
        feld->push_back(Vec3d(K, H, E));
        feld->push_back(Vec3d(F, G, L));
        feld->push_back(Vec3d(M, N, EG_FF));
    }
    return feld;
}
// Fr. Uebele

double HfT_osg_Plugin01_ParametricSurface::rot(double g, double max)
{
    double rot;
    if (g <= 0)
    {
        rot = 1.0f;
    }
    else
    {
        rot = 0.5f - (g * (0.5f / max));
    }
    return rot;
}

double HfT_osg_Plugin01_ParametricSurface::gruen(double g, double min)
{
    double gruen;
    if (g <= 0)
    {
        gruen = 0.5f - (g * (0.5f / min));
    }
    else
    {
        gruen = 1.0f;
    }
    return gruen;
}

double HfT_osg_Plugin01_ParametricSurface::blau()
{
    double blau = 0.0f;
    /*
	if(g<0){
		blau = 1.0f - (g*(1.0f/min));
		//blau = g*(1.0f/min);
	}
	else {
		//blau = 1.0f-(g*(1.0f/max));
		blau = g*(1.0f/max);

	}
	*/
    return blau;
}
