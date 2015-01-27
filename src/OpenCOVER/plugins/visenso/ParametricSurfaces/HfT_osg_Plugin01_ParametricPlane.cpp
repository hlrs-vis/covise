/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * HfT_osg_Plugin01_ParametricPlane.cpp
 *
 *  Created on: 13.12.2010
 *      Author: F-JS
 */

#include "HfT_osg_Plugin01_ParametricPlane.h"

//default constructor
HfT_osg_Plugin01_ParametricPlane::
    HfT_osg_Plugin01_ParametricPlane()
    : HfT_osg_Plugin01_ParametricSurface()
{
    m_a = 1;
    m_b = 1;
    m_c = 1;
    m_xstr = "A*u";
    m_ystr = "B*v";
    m_zstr = "0";
    mp_ParserSurface->SetA(m_a);
    mp_ParserSurface->SetB(m_b);
    mp_ParserSurface->SetC(m_c);

    this->setXpara(m_xstr);
    this->setYpara(m_ystr);
    this->setZpara(m_zstr);
}

//copy constructor
HfT_osg_Plugin01_ParametricPlane::
    HfT_osg_Plugin01_ParametricPlane(const HfT_osg_Plugin01_ParametricPlane &iParametricPlane)
    : HfT_osg_Plugin01_ParametricSurface()
{
    this->createGeometryandMode();
}

HfT_osg_Plugin01_ParametricPlane::
    HfT_osg_Plugin01_ParametricPlane(double a, double b, int iPatchesU, int iPatchesV,
                                     int isegmU, int isegmV,
                                     double iLowU, double iUpU, double iLowV,
                                     double iUpV, SurfMode iMode, ConsType iconsTyp,
                                     int iconsPoints, Image *image)
    : HfT_osg_Plugin01_ParametricSurface(a, b, iPatchesU, iPatchesV, isegmU, isegmV,
                                         iLowU, iUpU, iLowV, iUpV, iMode,
                                         iconsTyp, iconsPoints, image)
{
    m_ua = iLowU;
    m_ue = iUpU;
    m_va = iLowV;
    m_ve = iUpV;
    m_a = a;
    m_b = b;
    m_c = 1;
    m_xstr = "A*u";
    m_ystr = "B*v";
    m_zstr = "0";
    mp_ParserSurface->SetA(m_a);
    mp_ParserSurface->SetB(m_b);
    mp_ParserSurface->SetC(m_c);

    this->setXpara(m_xstr);
    this->setYpara(m_ystr);
    this->setZpara(m_zstr);
    this->createGeometryandMode();
}
HfT_osg_Plugin01_ParametricPlane::~HfT_osg_Plugin01_ParametricPlane()
{
}
double HfT_osg_Plugin01_ParametricPlane::getLowerBoundUorg()
{
    return m_ua;
}

double HfT_osg_Plugin01_ParametricPlane::getUpperBoundUorg()
{
    return m_ue;
}

double HfT_osg_Plugin01_ParametricPlane::getLowerBoundVorg()
{
    return m_va;
}

double HfT_osg_Plugin01_ParametricPlane::getUpperBoundVorg()
{
    return m_ve;
}
void HfT_osg_Plugin01_ParametricPlane::setLowerBoundUorg(double iLowU)
{
    m_ua = iLowU;
}
void HfT_osg_Plugin01_ParametricPlane::setUpperBoundUorg(double iUpU)
{
    m_ue = iUpU;
}

void HfT_osg_Plugin01_ParametricPlane::setLowerBoundVorg(double iLowV)
{
    m_va = iLowV;
}
void HfT_osg_Plugin01_ParametricPlane::setUpperBoundVorg(double iUpV)
{
    m_ve = iUpV;
}

void HfT_osg_Plugin01_ParametricPlane::setBoundaryorg(HfT_osg_Plugin01_Cons *boundary)
{
    // Neue Cons wird hinzugefügt und mp_Geom_C = Cons gesetzt
    this->computeCons(boundary);
    this->addDrawable(boundary);
    mp_Geom_Borg = boundary;
}
HfT_osg_Plugin01_Cons *HfT_osg_Plugin01_ParametricPlane::getBoundaryorg()
{
    return mp_Geom_Borg;
}
void HfT_osg_Plugin01_ParametricPlane::replaceBoundaryorg(HfT_osg_Plugin01_Cons *boundaryorg)
{
    this->computeCons(boundaryorg);
    if (mp_Geom_Borg)
        this->replaceDrawable(mp_Geom_Borg, boundaryorg);
    else
        this->addDrawable(boundaryorg);
    mp_Geom_Borg = boundaryorg;
}
