/** @file coordconv.hxx
 * conversion between coordinate systems.
 * FAMU Copyright (C) 1998-2003 Institute for Theory of Electrical Engineering
 * @author K. Frenner
 * @author W. Hafla
 */

// #include "coordconv.hxx" // conversion between coordinate systems.

#ifndef __coordconv_hxx__
#define __coordconv_hxx__


/**
 * conversion between coordinate systems.
 * @author K. Frenner
 */
namespace coordConv
{
    void kart2zy(const double& x, const double& y, const double& z, 
                 double &rho2, double &phi2, double &z2);

    void zy2kart(const double& rho, const double& phi, const double& z, 
                 double &x2,   double &y2,   double &z2);

}


#endif // __coordconv_hxx__

