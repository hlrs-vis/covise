/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef HfT_osg_Plugin01_ReadSurfDescription_H_
#define HfT_osg_Plugin01_ReadSurfDescription_H_

// Klasse zum Einlesen der Flächenbeschreibung
#include <string>

class HfT_osg_Plugin01_ReadSurfDescription
{
public:
    HfT_osg_Plugin01_ReadSurfDescription();
    HfT_osg_Plugin01_ReadSurfDescription(std::string pathconfig);
    virtual ~HfT_osg_Plugin01_ReadSurfDescription();

    std::string getXstr();
    std::string getYstr();
    std::string getZstr();
    double getUa();
    double getUe();
    double getVa();
    double getVe();
    double getA();
    double getB();
    double getC();
    void get_Param(std::string &xstr, std::string &ystr, std::string &zstr,
                   double &a, double &b, double &c,
                   double &ua, double &ue, double &va, double &ve);
    bool readData();

private:
    std::string m_surfname;
    std::string m_xstr, m_ystr, m_zstr;
    double m_a, m_b, m_c;
    double m_ua, m_ue, m_va, m_ve;
};
#endif /* HfT_osg_Plugin01_ReadSurfDescription_H_ */
