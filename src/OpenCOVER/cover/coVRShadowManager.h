/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVR_SHADOW_MANAGER_H
#define COVR_SHADOW_MANAGER_H

/*! \file
 \brief  manage Shadows

 \author (C)
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

#include <util/coExport.h>
#include <osgShadow/ShadowedScene>
#include <osgShadow/ShadowMap>

namespace opencover
{
class COVEREXPORT coVRShadowManager
{
public:
    coVRShadowManager();
    ~coVRShadowManager();
    static coVRShadowManager *instance();

    void setTechnique(const std::string &tech);
    std::string getTechnique(){return technique;};

private:
    
    std::string technique;
    static coVRShadowManager* inst;
};

}
#endif
