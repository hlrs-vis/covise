/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef HfT_osg_Plugin01_ParametricPlane_H_
#define HfT_osg_Plugin01_ParametricPlane_H_

#include "HfT_osg_Plugin01_ParametricSurface.h"

class HfT_osg_Plugin01_ParametricPlane : public HfT_osg_Plugin01_ParametricSurface
{
public:
    HfT_osg_Plugin01_ParametricPlane();
    HfT_osg_Plugin01_ParametricPlane(const HfT_osg_Plugin01_ParametricPlane &iParametricPlane);
    HfT_osg_Plugin01_ParametricPlane(double a, double b, int iPatchesU,
                                     int iPatchesV, int iSegmU, int iSegmV,
                                     double iLowU, double iUpU, double iLowV,
                                     double iUpV, SurfMode iMode, ConsType iconstyp, int iconsPoints, Image *image);

    virtual ~HfT_osg_Plugin01_ParametricPlane();

    double getLowerBoundUorg();
    void setLowerBoundUorg(double iLowU);

    double getUpperBoundUorg();
    void setUpperBoundUorg(double iUpU);

    double getLowerBoundVorg();
    void setLowerBoundVorg(double iLowV);

    double getUpperBoundVorg();
    void setUpperBoundVorg(double iUpV);

    HfT_osg_Plugin01_Cons *getBoundaryorg();
    void setBoundaryorg(HfT_osg_Plugin01_Cons *boundary);

    void replaceBoundaryorg(HfT_osg_Plugin01_Cons *boundaryorg);

private:
    double m_ua, m_ue, m_va, m_ve;
    ref_ptr<HfT_osg_Plugin01_Cons> mp_Geom_Borg;
};

#endif /* HfT_osg_Plugin01_ParametricPlane_H_ */
