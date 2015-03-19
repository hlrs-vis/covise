/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * HfT_osg_Plugin01_ParametricSurface .h
 *
 *  Created on: 10.12.2010
 *      Author: F-JS
 */

#ifndef HFT_OSG_PARAMETRIC_SURFACE_H_
#define HFT_OSG_PARAMETRIC_SURFACE_H_

#include <string>

#include <osg/Matrix>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Array>
#include <osg/ref_ptr>
#include <osg/Geode>
#include <osg/Group>
#include <osg/Geometry>
#include <osg/Drawable>
#include <osgDB/ReadFile>
#include <osg/Image>

#include "HfT_osg_Plugin01_Cons.h"
#include "hlparametricsurface3d.h"
#include "HfT_osg_StateSet.h"

using namespace osg;

//Abstrakte Basisklasse
class HfT_osg_Plugin01_ParametricSurface : public Geode
{
    friend class HfT_osg_Plugin01_Cons;

public:
    //constructor
    HfT_osg_Plugin01_ParametricSurface();

    HfT_osg_Plugin01_ParametricSurface(double a, double b, double c,
                                       int iPatU, int iPatV,
                                       int iSegmU, int iSegmV,
                                       double iLowU, double iUpU,
                                       double iLowV, double iUpV,
                                       SurfMode iSurfMode, ConsType iconsType, int iconsPoints);

    HfT_osg_Plugin01_ParametricSurface(double a, int iPatU, int iPatV,
                                       int iSegmU, int iSegmV,
                                       double iLowU, double iUpU,
                                       double iLowV, double iUpV,
                                       SurfMode iSurfMode, ConsType iconsType, int iconsPoints, Image *imgage);

    HfT_osg_Plugin01_ParametricSurface(double a, double b,
                                       int iPatU, int iPatV,
                                       int iSegmU, int iSegmV,
                                       double iLowU, double iUpU,
                                       double iLowV, double iUpV,
                                       SurfMode iSurfMode, ConsType iconsType, int iconsPoints, Image *imgage);

    HfT_osg_Plugin01_ParametricSurface(double a, double b, double c,
                                       int iPatU, int iPatV, int iSegmU, int iSegmV,
                                       double iLowU, double iUpU, double iLowV, double iUpV,
                                       SurfMode iSurfMode, ConsType iconsType, int iconsPoints,
                                       Image *imgage);

    //destructor
    virtual ~HfT_osg_Plugin01_ParametricSurface();

    //getter and setter
    int getSurfType();
    void setSurfType(int iSurfType);

    int getPatchesU();
    void setPatchesU(int iNumPatches);

    int getSegmentsU();
    void setSegmentsU(int iSemU);

    int getPatchesV();
    void setPatchesV(int iNumPatches);

    int getSegmentsV();
    void setSegmentsV(int iSemV);

    double getLowerBoundU();
    void setLowerBoundU(double iLowU);

    double getUpperBoundU();
    void setUpperBoundU(double iUpU);

    double getLowerBoundV();
    void setLowerBoundV(double iLowV);

    double getUpperBoundV();
    void setUpperBoundV(double iUpV);

    double getRadian();
    void setRadian(const double &iRadian);
    double getLength();
    void setLength(const double &iLength);
    double getHeight();
    void setHeight(const double &height);

    void setColor(SurfMode surfmode, Vec4d color);
    Vec4 getColor();

    Vec3Array *getPointArray();
    Vec2Array *getPointArray2();
    Vec4Array *getPointArray4();

    Vec3Array *getNormalArray();

    string getXpara();
    void setXpara(string xstr);
    string getYpara();
    void setYpara(string ystr);
    string getZpara();
    void setZpara(string zstr);
    void setParametrization(double A, double B, double C,
                            double ua, double ue, double va, double ve,
                            string xpstr, string ypstr, string zpstr);

    SurfMode getSurfMode();
    void setSurfMode(SurfMode iSurfMode);

    Vec2Array *getParameterValues();
    void setParameterValues(Vec2Array *iPaV);

    Image *getAllParam(int &Surftype, SurfMode &smode, int &pU, int &pV, int &Su, int &Sv,
                       ConsType &ctype, int &canz);

    Image *getImage();
    void setImage(Image *image);

    HlParametricSurface3d *getParserSurface();
    void setParserSurface(HlParametricSurface3d *parsersurface);

    // DrawElementsUint Indexlisten in Abhängigkeit des Modes berechnen
    void computeTriangleEdges();
    void computeTriangleEdges(int pU, int pV);
    void computeQuadEdges();
    void computeQuadEdges(int pU, int pV);
    void computeParaLineEdges();
    void computeParaLineEdges(int pU, int pV);
    void computePointEdges();
    void computePointEdges(int pU, int pV);
    void computeParameterValues();

    // Darstellung
    void createGeometryandMode();
    virtual void createGeometry();
    void createMode();
    void createMode(Vec4 color);

    virtual void digitalize(double U, double V);
    // Fr. Uebele
    Vec3dArray *digitalize(double U, double V, int i);
    // Fr. Uebele
    void updateCurvatureColorArray(int flag);
    Vec4d Curvature2Color(double curvature, double curvaturemin, double curvaturemax);

    // recompute
    void recomputeGeometryandMode();
    void recomputeGeometry();
    void recomputeGeometry(int iPatU, int iPatV, int iSegmU, int iSegmV);
    void recomputeMode(SurfMode iSurfMode);
    void recomputeMode(SurfMode iSurfMode, Image *image);
    void recomputeMode(SurfMode iSurfMode, Vec4 color, Image *image);

    // Zugriffsfunktionen etc. für den Flächenrand
    int getBoundaryPointanz();
    void setBoundaryPointanz(int cp);
    Vec2Array *getBoundaryParameterValues();
    void setBoundaryParameterValues(Vec2Array *iPaV);
    HfT_osg_Plugin01_Cons *getBoundary();
    void setBoundary(HfT_osg_Plugin01_Cons *boundary);
    void replaceBoundary(HfT_osg_Plugin01_Cons *boundary);
    void setColorBoundary(Vec4d color);

    // Zugriffsfunktionen etc. für die innere Flächenkurve
    int getConsPointanz();
    void setConsPointanz(int cp);
    Vec2Array *getConsParameterValues();
    void setConsParameterValues(Vec2Array *iPaV);
    ConsType getConsType();
    void setConsType(ConsType cm);
    HfT_osg_Plugin01_Cons *getCons();
    void setCons(HfT_osg_Plugin01_Cons *Cons);
    void replaceCons(HfT_osg_Plugin01_Cons *Cons);
    void computeCons(HfT_osg_Plugin01_Cons *cons);
    // Fr. Uebele
    Vec3dArray *computeSchnittkurve(double u, double v, int i);
    // Fr. Uebele
    //Array4 anlegen fuer step9 Punkt auswaehlen
    void insertArray4();
    void insertArray2(osg::Vec3 startPickPos);

protected:
    // Typ der Fläche
    int m_Surftype;
    // Patchanzahl und Segmentanzahl
    int m_n;
    int m_m;
    int m_su;
    int m_sv;
    int m_Pointanz;
    // Darstellungsmodus der Fläche
    SurfMode m_Surfmode;
    // Intervallgrenzen der Parameter
    double m_cua;
    double m_cue;
    double m_cva;
    double m_cve;
    // Parametrisierung

    // Felder für die zugehörigen (u,v) Koordinaten
    Vec2Array *mp_Coords_Geom;
    Vec2Array *mp_TexCoords_Geom;
    ref_ptr<Image> mp_Image_Texture;

    // Werte für den FlächenRand und innere Flächenkurve:
    int m_Pointanz_B;
    ConsType m_type_C;
    int m_Pointanz_C;

    //Parameter der Fläche als double
    double m_a;
    double m_b;
    double m_c;

    // Für die Krümmungen
    double m_GC;
    double m_MC;
    DoubleArray *mp_GC_Geom;
    DoubleArray *mp_MC_Geom;

    // Parametrisierung der Fläche als string
    std::string m_xstr;
    std::string m_ystr;
    std::string m_zstr;

    // Pointer auf Surfaceklasse von HPH
    HlParametricSurface3d *mp_ParserSurface;

    // Punkt und Normale (auch für Flächenkurve)
    Vec3d m_Point;
    Vec3d m_Normal;
    Vec3d m_PointU;
    Vec3d m_PointV;
    Vec3d m_PointUU;
    Vec3d m_PointUV;
    Vec3d m_PointVV;

    // Farbwerte für Flächenpunkte, Parameterlinien, Flächenränder
    Vec4Array *mp_CurvatureColorArray;
    // Geometrie Objekte für Fläche, Parameternetze, Flächenkurve, Flächenrand
    ref_ptr<Geometry> mp_Geom;
    ref_ptr<Geometry> mp_Geom_Curvature;
    ref_ptr<Geometry> mp_Geom_PU;
    ref_ptr<Geometry> mp_Geom_PV;
    ref_ptr<HfT_osg_Plugin01_Cons> mp_Geom_C;
    ref_ptr<HfT_osg_Plugin01_Cons> mp_Geom_B;

    // Felder für die digitalisierten Punkte und Normalen
    Vec3Array *mp_Points_Geom;
    Vec4Array *mp_Points_Geom4;
    Vec2Array *mp_Points_Geom2;
    Vec3Array *mp_Normals_Geom;

    // Array mit Eckenindices für die Vec3Array der digitalisierten Punkte
    ref_ptr<DrawElementsUInt> mp_PointEdges_Geom;
    ref_ptr<DrawElementsUInt> mp_TriangleEdges_Geom;
    ref_ptr<DrawElementsUInt> mp_QuadEdges_Geom;
    ref_ptr<DrawElementsUInt> mp_LineEdges_Geom_PU;
    ref_ptr<DrawElementsUInt> mp_LineEdges_Geom_PV;

    // Darstellungsmodi für die Geometrie Objekte
    HfT_osg_StateSet *mp_StateSet_Geom;
    HfT_osg_StateSet *mp_StateSet_Geom_Curvature;
    HfT_osg_StateSet *mp_StateSet_Geom_PU;
    HfT_osg_StateSet *mp_StateSet_Geom_PV;

    //interne Komfort Methoden
    float abstand(osg::Vec3 startPickPos, int i);
    void quicksort(Vec4Array *s, int left, int right);
    int partition(Vec4Array *s, int left, int right);
    void initializeMembers();
    void initializeGeoms();
    void initializeGeomsArraysandDrawables();
    void initializeVertexArrays();
    void initializeDrawElements();
    void initializeStateSets();
    void initializeParser();

    void clearMode();
    void clearGeoms();
    void clearGeomsArraysandDrawables();
    void clearVertexArrays();
    void clearDrawElements();

    double rot(double g, double max);
    double gruen(double g, double min);
    double blau();

    int computeIndex(int i, int j, int pU, int pV);
};

#endif /* HFT_OSG_PARAMETRIC_SURFACE_H_ */
