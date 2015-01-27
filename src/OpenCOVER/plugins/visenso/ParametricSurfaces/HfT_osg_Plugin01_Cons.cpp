/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "HfT_osg_Plugin01_Cons.h"
#include "HfT_osg_Plugin01_ParametricSurface.h"

using namespace osg;

HfT_osg_Plugin01_Cons::HfT_osg_Plugin01_Cons()
    : Geometry()
{
    m_type = CNOTHING;
    m_ta = 0.;
    m_te = 1.;
    m_Pointanz = 100;
    m_Position = Vec2d(0., 0.);
    m_isSet3D = false;
    this->initializeMembers();
}
HfT_osg_Plugin01_Cons::HfT_osg_Plugin01_Cons(HfT_osg_Plugin01_Cons &iCons)
    : Geometry()
{
    m_type = iCons.m_type;
    m_ta = iCons.m_ta;
    m_te = iCons.m_te;
    m_Pointanz = iCons.m_Pointanz;
    m_Position = iCons.m_Position;
    m_Color = iCons.m_Color;
    m_isSet3D = false;

    this->initializeMembers();
    setParameterValues(iCons.getParameterValues());
    if (m_type == CNATBOUND)
        mp_Lw->setWidth(2.f);
    if (m_type != CNOTHING)
        this->createMode();
}
HfT_osg_Plugin01_Cons::HfT_osg_Plugin01_Cons(int anz)
    : Geometry()
{
    m_type = CNOTHING;
    m_ta = 0.;
    m_te = 1.;
    m_Pointanz = anz;
    m_Position = Vec2d(0., 0.);
    m_isSet3D = false;
    this->initializeMembers();
}
HfT_osg_Plugin01_Cons::HfT_osg_Plugin01_Cons(Vec2Array *ParameterValues)
    : Geometry()
{
    m_type = CNOTHING;
    m_ta = 0.;
    m_te = 1.;
    m_Pointanz = ParameterValues->size();
    m_Position = Vec2d(0., 0.);
    m_isSet3D = false;
    this->initializeMembers();
    this->setParameterValues(ParameterValues);
}
HfT_osg_Plugin01_Cons::HfT_osg_Plugin01_Cons(int anz, ConsType Type, class HfT_osg_Plugin01_ParametricSurface *surface)
    : Geometry()
{
    m_ta = 0.;
    m_te = 1.;
    m_Pointanz = anz;
    m_Position = Vec2d(0., 0.);
    m_isSet3D = false;
    this->initializeMembers();
    if (Type == CNATBOUND)
        mp_Lw->setWidth(2.f);
    this->setParameterValues(Type, surface->m_cua, surface->m_cue, surface->m_cva, surface->m_cve);
    if (Type != CNOTHING)
        this->createMode();
}
HfT_osg_Plugin01_Cons::HfT_osg_Plugin01_Cons(int anz, ConsType Type, double ua, double ue, double va, double ve)
    : Geometry()
{
    m_ta = 0.;
    m_te = 1.;
    m_Pointanz = anz;
    m_Position = Vec2d(0., 0.);
    m_isSet3D = false;
    this->initializeMembers();
    if (Type == CNATBOUND)
        mp_Lw->setWidth(2.f);
    this->setParameterValues(Type, ua, ue, va, ve);
    if (Type != CNOTHING)
        this->createMode();
}
HfT_osg_Plugin01_Cons::HfT_osg_Plugin01_Cons(int anz, ConsType Type, double ua, double ue, double va, double ve, Vec4d color, int formula_)
    : Geometry()
{
    m_ta = 0.;
    m_te = 1.;
    m_Pointanz = anz;
    m_Position = Vec2d(0., 0.);
    m_Color = color;
    m_formula_ = formula_;
    m_isSet3D = false;
    this->initializeMembers();
    this->setParameterValues(Type, ua, ue, va, ve);
    if (Type == CNATBOUND)
        mp_Lw->setWidth(2.f);
    if (Type != CNOTHING)
        this->createMode();
}
HfT_osg_Plugin01_Cons::HfT_osg_Plugin01_Cons(int anz, ConsType Type, Vec2d pos, double ua, double ue, double va, double ve)
    : Geometry()
{
    m_ta = 0.;
    m_te = 1.;
    m_Pointanz = anz;
    m_Position = pos;
    m_isSet3D = false;
    this->initializeMembers();
    this->setParameterValues(Type, ua, ue, va, ve);
    if (Type == CNATBOUND)
        mp_Lw->setWidth(2.f);
    if (Type != CNOTHING)
        this->createMode();
}
HfT_osg_Plugin01_Cons::HfT_osg_Plugin01_Cons(int anz, ConsType Type, Vec2d pos, double ua, double ue, double va, double ve, Vec4d color)
    : Geometry()
{
    m_ta = 0.;
    m_te = 1.;
    m_Pointanz = anz;
    m_Position = pos;
    m_Color = color;
    m_isSet3D = false;
    this->initializeMembers();
    this->setParameterValues(Type, ua, ue, va, ve);
    if (Type == CNATBOUND)
        mp_Lw->setWidth(2.f);
    if (Type != CNOTHING)
        this->createMode();
}
HfT_osg_Plugin01_Cons::~HfT_osg_Plugin01_Cons()
{
    /*	m_ua = 0.0; //muss alles ausgeklammert sein
	m_ue = 0.0;
	m_va = 0.0;
	m_ve = 0.0;
	m_type = CNOTHING;
    m_Pointanz  = 0;
	m_ta=0.;
	m_te=0.;

	Node *node = this -> getParent(0); // eigentlich Geode 
	Geode *geode = dynamic_cast<Geode *> (node);
	// Geode *geode = node ->asGeode();
	geode->removeDrawable(this);

	delete this; */
}
void HfT_osg_Plugin01_Cons::initializeMembers()
{
    mp_Coords_Geom = new Vec2Array(m_Pointanz + 1);
    mp_Points_Geom = new Vec3Array(m_Pointanz + 1);
    mp_Normals_Geom = new Vec3Array(m_Pointanz + 1);
    this->setVertexArray(mp_Points_Geom);
    this->setNormalArray(mp_Normals_Geom);
    this->setNormalBinding(Geometry::BIND_PER_VERTEX);
    mp_LineEdges_Geom = new DrawElementsUInt(PrimitiveSet::LINES, 0);
    mp_StateSet_Geom = new StateSet();
    this->setStateSet(mp_StateSet_Geom);

    mp_PolygonMode = new PolygonMode(PolygonMode::FRONT_AND_BACK, PolygonMode::LINE);
    mp_Material = new Material();
    mp_Lw = new LineWidth(3.f);
}

void HfT_osg_Plugin01_Cons::createParameterValuesfromType()
{
    switch (m_type)
    {
    // Für die Flächenkurve (u,v) Werte merken
    case CUCENTER: // u-Mitte-Linie
    {
        createUCenter();
    }
    break;

    case CVCENTER: // v-Mitte-Linie
    {
        createVCenter();
    }
    break;

    case CDIAGONAL: // Diagonale von links unten nach rechts oben
    {
        createDiagonal();
    }
    break;
    case CELLIPSE: // max. Ellipse
    {
        createEllipse();
    }
    break;
    case CTRIANGLE: // Dreieck (ua,va), (um,ve), (ue,va)
    {
        createTriangle();
    }
    break;
    case CSQUARE: // Viereck (ua + 1/10*du,va +1/10*dv),
        //(ue - 1/10*du ,va + 1/10*dv), ...
        {
            createSquare();
        }
        break;
    case CNATBOUND: // natürlicher Rand der Fläche
    {
        createNatBound();
    }
    break;
    default:
        break;
    }
}

// Nochmal exra den Flächenrand digitalisieren
void HfT_osg_Plugin01_Cons::createNatBound()
{
    int anz1 = 0;
    int anz2 = 0;
    int anz3 = 0;
    int anz4 = 0;
    int k = 0, i = 0;

    anz1 = m_Pointanz / 4;
    anz2 = m_Pointanz / 4;
    anz3 = m_Pointanz / 4;
    anz4 = m_Pointanz - anz1 - anz2 - anz3;

    double dt = m_te - m_ta, u, v, t;
    // Untere Randkurve
    for (i = 0; i <= anz1 - 1; i++)
    {
        t = m_ta + (m_te - m_ta) * i / (anz1 - 1);
        u = /*m_ua+(m_ue-m_ua)*t;*/ 1 / dt * ((m_te - t) * m_ua + (t - m_ta) * m_ue);
        v = m_va;

        (*mp_Coords_Geom)[k].set(u, v);
        k++;
    }
    // Rechte Randkurve
    for (i = 0; i <= anz2 - 1; i++)
    {
        t = m_ta + (m_te - m_ta) * i / (anz2 - 1);
        u = m_ue;
        v = /*m_va+(m_ve-m_va)*t;*/ 1 / dt * ((m_te - t) * m_va + (t - m_ta) * m_ve);

        (*mp_Coords_Geom)[k].set(u, v);
        k++;
    }
    // Obere Randkurve von hinten nach vorne
    for (i = 0; i <= anz3 - 1; i++)
    {
        t = m_ta + (m_te - m_ta) * i / (anz3 - 1);
        u = /*m_ue-(m_ue-m_ua)*t;*/ 1 / dt * ((m_te - t) * m_ue + (t - m_ta) * m_ua);
        v = m_ve;

        (*mp_Coords_Geom)[k].set(u, v);
        k++;
    }
    // Linke Randkurve von oben nach unten
    for (i = 0; i <= anz4 - 1; i++)
    {
        t = m_ta + (m_te - m_ta) * i / (anz4 - 1);
        u = m_ua;
        v = /*m_ve-(m_ve-m_va)*t;*/ 1 / dt * ((m_te - t) * m_ve + (t - m_ta) * m_va);

        (*mp_Coords_Geom)[k].set(u, v);
        k++;
    }
}
void HfT_osg_Plugin01_Cons::createUCenter()
{
    double dt = m_te - m_ta;
    double u = 0.5 * (m_ua + m_ue);
    u += m_Position[0];
    if (u > m_ue)
        u = m_ue;
    else if (u < m_ua)
        u = m_ua;
    int k = 0;
    for (int i = 0; i <= m_Pointanz; i++)
    {
        double t = m_ta + (m_te - m_ta) * i / m_Pointanz;
        double v = m_va + ((t - m_ta) / dt) * (m_ve - m_va);

        v += m_Position[1];
        if (v > m_ve)
            v = m_ve;
        else if (v < m_va)
            v = m_va;

        (*mp_Coords_Geom)[k].set(u, v);
        k++;
    }
}
void HfT_osg_Plugin01_Cons::createVCenter()
{
    double dt = m_te - m_ta;
    double v = 0.5 * (m_va + m_ve);
    v += m_Position[1];
    if (v > m_ve)
        v = m_ve;
    else if (v < m_va)
        v = m_va;
    int k = 0;
    for (int i = 0; i <= m_Pointanz; i++)
    {
        double t = m_ta + (m_te - m_ta) * i / m_Pointanz;
        double u = m_ua + ((t - m_ta) / dt) * (m_ue - m_ua);

        u += m_Position[0];
        if (u > m_ue)
            u = m_ue;
        else if (u < m_ua)
            u = m_ua;

        (*mp_Coords_Geom)[k].set(u, v);
        k++;
    }
}
void HfT_osg_Plugin01_Cons::createDiagonal()
{
    double dt = m_te - m_ta;
    int k = 0;
    for (int i = 0; i <= m_Pointanz; i++)
    {
        double t = m_ta + (m_te - m_ta) * i / m_Pointanz;
        double u = m_ua + ((t - m_ta) / dt) * (m_ue - m_ua);
        double v = m_va + ((t - m_ta) / dt) * (m_ve - m_va);

        u += m_Position[0];
        if (u > m_ue)
            u = m_ue;
        else if (u < m_ua)
            u = m_ua;
        v += m_Position[1];
        if (v > m_ve)
            v = m_ve;
        else if (v < m_va)
            v = m_va;

        (*mp_Coords_Geom)[k].set(u, v);
        k++;
    }
}
void HfT_osg_Plugin01_Cons::createEllipse()
{
    double Pi = 4.0 * atan(1.0);
    double dt = m_te - m_ta;
    int k = 0;
    for (int i = 0; i <= m_Pointanz; i++)
    {
        double t = m_ta + (m_te - m_ta) * i / m_Pointanz;
        double u = 0.5 * (m_ua + m_ue) + cos(2 * Pi * (t - m_ta) / dt) * (m_ue - m_ua) / 2.;
        double v = 0.5 * (m_va + m_ve) + sin(2 * Pi * (t - m_ta) / dt) * (m_ve - m_va) / 2.;

        u += m_Position[0];
        if (u > m_ue)
            u = m_ue;
        else if (u < m_ua)
            u = m_ua;
        v += m_Position[1];
        if (v > m_ve)
            v = m_ve;
        else if (v < m_va)
            v = m_va;

        (*mp_Coords_Geom)[k].set(u, v);
        k++;
    }
}
void HfT_osg_Plugin01_Cons::createTriangle()
{
    // Insgesamt wieder m_Pointanz +1 Punkte erfassen
    int anz1 = m_Pointanz / 3;
    int anz2 = m_Pointanz / 3;
    int anz3 = m_Pointanz - anz1 - anz2;
    int k = 0, i = 0;
    double dt = m_te - m_ta, u, v, t;
    for (i = 0; i <= anz1 - 1; i++)
    {
        t = m_ta + (m_te - m_ta) * i / (anz1 - 1);
        u = 1 / dt * ((m_te - t) * m_ua + (t - m_ta) * 0.5 * (m_ua + m_ue));
        v = 1 / dt * ((m_te - t) * m_va + (t - m_ta) * m_ve);

        u += m_Position[0];
        if (u > m_ue)
            u = m_ue;
        else if (u < m_ua)
            u = m_ua;
        v += m_Position[1];
        if (v > m_ve)
            v = m_ve;
        else if (v < m_va)
            v = m_va;

        (*mp_Coords_Geom)[k].set(u, v);
        k++;
    }
    for (i = 0; i <= anz2 - 1; i++)
    {
        t = m_ta + (m_te - m_ta) * i / (anz2 - 1);
        u = 1 / dt * ((m_te - t) * 0.5 * (m_ua + m_ue) + (t - m_ta) * m_ue);
        v = 1 / dt * ((m_te - t) * m_ve + (t - m_ta) * m_va);

        u += m_Position[0];
        if (u > m_ue)
            u = m_ue;
        else if (u < m_ua)
            u = m_ua;
        v += m_Position[1];
        if (v > m_ve)
            v = m_ve;
        else if (v < m_va)
            v = m_va;

        (*mp_Coords_Geom)[k].set(u, v);
        k++;
    }
    for (i = 0; i <= anz3; i++)
    {
        t = m_ta + (m_te - m_ta) * i / anz3;
        u = 1 / dt * ((m_te - t) * m_ue + (t - m_ta) * m_ua);
        v = m_va;

        u += m_Position[0];
        if (u > m_ue)
            u = m_ue;
        else if (u < m_ua)
            u = m_ua;
        v += m_Position[1];
        if (v > m_ve)
            v = m_ve;
        else if (v < m_va)
            v = m_va;

        (*mp_Coords_Geom)[k].set(u, v);
        k++;
    }
}
void HfT_osg_Plugin01_Cons::createSquare()
{
    // Insgesamt wieder m_Pointanz +1 Punkte erfassen
    int anz1 = m_Pointanz / 4;
    int anz2 = m_Pointanz / 4;
    int anz3 = m_Pointanz / 4;
    int anz4 = m_Pointanz - anz1 - anz2 - anz3;
    int k = 0, i = 0;
    double du = m_ue - m_ua;
    double dv = m_ve - m_va;
    double dt = m_te - m_ta, u, v, t;
    for (i = 0; i <= anz1 - 1; i++)
    {
        t = m_ta + (m_te - m_ta) * i / (anz1 - 1);
        u = 1 / dt * ((m_te - t) * (m_ua + 0.1 * du) + (t - m_ta) * (m_ue - 0.1 * du));
        v = m_va + 0.1 * dv;

        u += m_Position[0];
        if (u > m_ue)
            u = m_ue;
        else if (u < m_ua)
            u = m_ua;
        v += m_Position[1];
        if (v > m_ve)
            v = m_ve;
        else if (v < m_va)
            v = m_va;

        (*mp_Coords_Geom)[k].set(u, v);
        k++;
    }
    for (i = 0; i <= anz2 - 1; i++)
    {
        t = m_ta + (m_te - m_ta) * i / (anz2 - 1);
        u = m_ue - 0.1 * du;
        v = 1 / dt * ((m_te - t) * (m_va + 0.1 * dv) + (t - m_ta) * (m_ve - 0.1 * dv));

        u += m_Position[0];
        if (u > m_ue)
            u = m_ue;
        else if (u < m_ua)
            u = m_ua;
        v += m_Position[1];
        if (v > m_ve)
            v = m_ve;
        else if (v < m_va)
            v = m_va;

        (*mp_Coords_Geom)[k].set(u, v);
        k++;
    }
    for (i = 0; i <= anz3 - 1; i++)
    {
        t = m_ta + (m_te - m_ta) * i / (anz3 - 1);
        u = 1 / dt * ((m_te - t) * (m_ue - 0.1 * du) + (t - m_ta) * (m_ua + 0.1 * du));
        v = m_ve - 0.1 * dv;

        u += m_Position[0];
        if (u > m_ue)
            u = m_ue;
        else if (u < m_ua)
            u = m_ua;
        v += m_Position[1];
        if (v > m_ve)
            v = m_ve;
        else if (v < m_va)
            v = m_va;

        (*mp_Coords_Geom)[k].set(u, v);
        k++;
    }
    for (i = 0; i <= anz4; i++)
    {
        t = m_ta + (m_te - m_ta) * i / anz4;
        u = m_ua + 0.1 * du;
        v = 1 / dt * ((m_te - t) * (m_ve - 0.1 * dv) + (t - m_ta) * (m_va + 0.1 * dv));

        u += m_Position[0];
        if (u > m_ue)
            u = m_ue;
        else if (u < m_ua)
            u = m_ua;
        v += m_Position[1];
        if (v > m_ve)
            v = m_ve;
        else if (v < m_va)
            v = m_va;

        (*mp_Coords_Geom)[k].set(u, v);
        k++;
    }
}
void HfT_osg_Plugin01_Cons::createMode()
{
    computeEdges();
    this->addPrimitiveSet(mp_LineEdges_Geom);
    mp_StateSet_Geom->setAttribute(mp_Lw);
    setColorAndMaterial(m_Color, mp_Material);
    mp_StateSet_Geom->setAttribute(mp_Material, osg::StateAttribute::PROTECTED); //geaendert fuer Menu sichtbar
}
void HfT_osg_Plugin01_Cons::computeEdges()
{
    // Da Punkte von 0 --> m_Pointanz

    for (int k = 0; k < m_Pointanz - 1; k++) //kleiner -1
    {
        mp_LineEdges_Geom->push_back(k);
        mp_LineEdges_Geom->push_back(k + 1);
    }
}
// Getter
int HfT_osg_Plugin01_Cons::getPointanz()
{
    return m_Pointanz;
}

Vec2Array *HfT_osg_Plugin01_Cons::getParameterValues()
{
    return mp_Coords_Geom;
}

Vec3Array *HfT_osg_Plugin01_Cons::getPoints()
{
    return mp_Points_Geom;
}

Vec3Array *HfT_osg_Plugin01_Cons::getNormals()
{
    return mp_Normals_Geom;
}

ConsType HfT_osg_Plugin01_Cons::getType()
{
    return m_type;
}
int HfT_osg_Plugin01_Cons::getLowerBound()
{
    return m_ta;
}
int HfT_osg_Plugin01_Cons::getUpperBound()
{
    return m_te;
}
Vec2d HfT_osg_Plugin01_Cons::getPosition()
{
    return m_Position;
}
Vec4d HfT_osg_Plugin01_Cons::getColor()
{
    return m_Color;
}
DrawElementsUInt *HfT_osg_Plugin01_Cons::getDrawElementsUInt()
{
    return mp_LineEdges_Geom;
}
StateSet *HfT_osg_Plugin01_Cons::getStateSet()
{
    return mp_StateSet_Geom;
}
PolygonMode *HfT_osg_Plugin01_Cons::getPolygonMode()
{
    return mp_PolygonMode;
}
Material *HfT_osg_Plugin01_Cons::getMaterial()
{
    return mp_Material;
}

// Setter
void HfT_osg_Plugin01_Cons::setPointanz(int iCp)
{
    m_Pointanz = iCp;
}

void HfT_osg_Plugin01_Cons::setParameterValues(Vec2Array *iPaV)
{
    for (unsigned int i = 0; i < iPaV->size(); i++)
    {
        Vec2d p = (*iPaV)[i];
        (*mp_Coords_Geom)[i].set(p[0], p[1]);
    }
}
void HfT_osg_Plugin01_Cons::setParameterValues(ConsType Type, double ua, double ue, double va, double ve)
{
    m_type = Type;
    m_ua = ua;
    m_ue = ue;
    m_va = va, m_ve = ve;
    createParameterValuesfromType();
}
void HfT_osg_Plugin01_Cons::setParameterValues(ConsType Type)
{
    m_type = Type;
    createParameterValuesfromType();
}
void HfT_osg_Plugin01_Cons::setParameterValues(double ua, double ue, double va, double ve)
{
    m_ua = ua;
    m_ue = ue;
    m_va = va, m_ve = ve;
    createParameterValuesfromType();
}

void HfT_osg_Plugin01_Cons::setType(ConsType iCm)
{
    m_type = iCm;
}

void HfT_osg_Plugin01_Cons::setLowerBound(double iLowT)
{
    m_ta = iLowT;
}
void HfT_osg_Plugin01_Cons::setUpperBound(double iUpT)
{
    m_te = iUpT;
}
void HfT_osg_Plugin01_Cons::setPosition(Vec2d ipos)
{
    m_Position = ipos;
}
void HfT_osg_Plugin01_Cons::setColor(Vec4d color)
{
    m_Color = color;
}
void HfT_osg_Plugin01_Cons::setColorAndMaterial(Vec4 Color, Material *Mat)
{
    Mat->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    Mat->setDiffuse(Material::FRONT, Color);
    Mat->setSpecular(Material::FRONT, Color);
}
