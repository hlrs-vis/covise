/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef HfT_osg_Plugin01_Cons_H_
#define HfT_osg_Plugin01_Cons_H_

#include <osg/Node>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/StateSet>
#include <osg/PolygonMode>
#include <osg/Drawable>
#include <osg/Material>
#include <osg/LineWidth>

class HfT_osg_Plugin01_ParametricSurface;

enum ConsType
{
    CNOTHING,
    CUCENTER,
    CVCENTER,
    CDIAGONAL,
    CTRIANGLE,
    CELLIPSE,
    CSQUARE,
    CNATBOUND
}; //ParamLinien
using namespace osg;

class HfT_osg_Plugin01_Cons : public Geometry
{

    friend class HfT_osg_Plugin01_ParametricSurface;

public:
    //constructor
    HfT_osg_Plugin01_Cons();
    HfT_osg_Plugin01_Cons(HfT_osg_Plugin01_Cons &iCons);
    HfT_osg_Plugin01_Cons(int anz);
    HfT_osg_Plugin01_Cons(Vec2Array *ParameterValues);
    HfT_osg_Plugin01_Cons(int anz, ConsType Type, HfT_osg_Plugin01_ParametricSurface *surface);
    HfT_osg_Plugin01_Cons(int anz, ConsType Type, double ua, double ue, double va, double ve);
    HfT_osg_Plugin01_Cons(int anz, ConsType Type, double ua, double ue, double va, double ve, Vec4d color, int formula_);
    HfT_osg_Plugin01_Cons(int anz, ConsType Type, Vec2d pos, double ua, double ue, double va, double ve);
    HfT_osg_Plugin01_Cons(int anz, ConsType Type, Vec2d pos, double ua, double ue, double va, double ve, Vec4d color);

    //destructor
    ~HfT_osg_Plugin01_Cons();

    //getter and setter
    int getPointanz();
    void setPointanz(int cp);
    Vec2Array *getParameterValues();
    void setParameterValues(Vec2Array *iPaV);
    void setParameterValues(ConsType type, double ua, double ue, double va, double ve);
    void setParameterValues(double ua, double ue, double va, double ve);
    void setParameterValues(ConsType type);
    Vec3Array *getPoints();
    Vec3Array *getNormals();
    ConsType getType();
    void setType(ConsType cm);
    int getLowerBound();
    void setUpperBound(double iUpT);
    int getUpperBound();
    void setLowerBound(double iLowT);
    void setPosition(Vec2d iPos);
    Vec2d getPosition();
    Vec4d getColor();
    void setColor(Vec4d color);

    DrawElementsUInt *getDrawElementsUInt();
    StateSet *getStateSet();
    PolygonMode *getPolygonMode();
    Material *getMaterial();

protected:
    // Member Varialen
    double m_ua, m_ue, m_va, m_ve;
    ConsType m_type;
    int m_Pointanz;
    int m_formula_;
    bool m_isSet3D;
    double m_ta, m_te;
    Vec4d m_Color;
    Vec2d m_Position;

    Vec3Array *mp_Points_Geom;
    Vec3Array *mp_Normals_Geom;
    Vec2Array *mp_Coords_Geom;
    DrawElementsUInt *mp_LineEdges_Geom;
    StateSet *mp_StateSet_Geom;
    PolygonMode *mp_PolygonMode;
    Material *mp_Material;
    LineWidth *mp_Lw;

    // Methoden
    void initializeMembers();
    void createParameterValuesfromType();
    void createNatBound();
    void createUCenter();
    void createVCenter();
    void createDiagonal();
    void createEllipse();
    void createTriangle();
    void createSquare();
    void createMode();
    void computeEdges();
    void setColorAndMaterial(Vec4 Color, Material *Mat);
};

#endif /* HfT_osg_Plugin01_Cons_H_ */
