/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <osgDB/ReadFile>
#include <osgDB/fstream>

#include "HfT_osg_Plugin01_ReadSurfDescription.h"

//liest Flaechen aus Datei

//default constructor
HfT_osg_Plugin01_ReadSurfDescription::HfT_osg_Plugin01_ReadSurfDescription()
{
    m_surfname = "Unbekanntes Description File";
    // Standardmäßig Kugel
    m_a = 12.;
    m_b = 0.;
    m_c = 0.;
    m_ua = 0.;
    m_ue = 6.28;
    m_va = -3.14;
    m_ve = 0.;
    m_xstr = "A*cos(u)*sin(v)";
    m_ystr = "A*sin(u)*sin(v)";
    m_zstr = "A*cos(v)";
}
HfT_osg_Plugin01_ReadSurfDescription::HfT_osg_Plugin01_ReadSurfDescription(std::string filename)
{
    m_surfname = filename;
    this->readData();
}
HfT_osg_Plugin01_ReadSurfDescription::~HfT_osg_Plugin01_ReadSurfDescription()
{
}
std::string HfT_osg_Plugin01_ReadSurfDescription::getXstr()
{
    return m_xstr;
}
std::string HfT_osg_Plugin01_ReadSurfDescription::getYstr()
{
    return m_ystr;
}
std::string HfT_osg_Plugin01_ReadSurfDescription::getZstr()
{
    return m_zstr;
}
double HfT_osg_Plugin01_ReadSurfDescription::getA()
{
    return m_a;
}
double HfT_osg_Plugin01_ReadSurfDescription::getB()
{
    return m_b;
}
double HfT_osg_Plugin01_ReadSurfDescription::getC()
{
    return m_c;
}
double HfT_osg_Plugin01_ReadSurfDescription::getUa()
{
    return m_ua;
}
double HfT_osg_Plugin01_ReadSurfDescription::getUe()
{
    return m_ue;
}
double HfT_osg_Plugin01_ReadSurfDescription::getVe()
{
    return m_ve;
}
double HfT_osg_Plugin01_ReadSurfDescription::getVa()
{
    return m_va;
}

void HfT_osg_Plugin01_ReadSurfDescription::get_Param(std::string &xstr, std::string &ystr, std::string &zstr,
                                                     double &a, double &b, double &c,
                                                     double &ua, double &ue, double &va, double &ve)
{
    xstr = getXstr();
    ystr = getYstr();
    zstr = getZstr();
    a = getA();
    b = getB();
    c = getC();
    ua = getUa();
    ue = getUe();
    va = getVa();
    ve = getVe();
}

bool HfT_osg_Plugin01_ReadSurfDescription::readData()
{
    std::fstream ein;
    char c;

    ein.open(m_surfname.c_str(), std::fstream::in);
    if (ein.is_open())
    {
        ein >> m_a >> c >> m_b >> c >> m_c;
        ein >> m_ua >> c >> m_ue;
        ein >> m_va >> c >> m_ve;
        ein >> m_xstr;
        ein >> m_ystr;
        ein >> m_zstr;

        ein.close();
        return true;
    }
    else
        return false;
}
