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
#include <osgShadow/ShadowVolume>
#include <osgShadow/ShadowTexture>
#include <osgShadow/SoftShadowMap>
#include <osgShadow/StandardShadowMap>
#include <osgShadow/MinimalShadowMap>
#include <osgShadow/LightSpacePerspectiveShadowMap>

namespace opencover
{
class COVEREXPORT coVRShadowManager
{
public:
    ~coVRShadowManager();
    static coVRShadowManager *instance();
    osgShadow::ShadowedScene *newScene();

    void setLight(osg::LightSource *ls);

    void setTechnique(const std::string &tech);
    std::string getTechnique();
    void setSoftnessWidth(float w);
    void setJitteringScale(float s);
    void setTextureSize(osg::Vec2s ts);

private:
    
    coVRShadowManager();
    std::string technique;
    static coVRShadowManager* inst;
    osg::ref_ptr<osgShadow::ShadowVolume> sv;
    osg::ref_ptr<osgShadow::ShadowTexture> st;
    osg::ref_ptr<osgShadow::SoftShadowMap> softSM;
    osg::ref_ptr<osgShadow::StandardShadowMap> standardSM;
    osg::ref_ptr<osgShadow::MinimalShadowMap> lspsm;
    osg::ref_ptr<osgShadow::MinimalShadowMap> lspsmcb;
    osg::ref_ptr<osgShadow::MinimalShadowMap> lspsmdb;
    osg::ref_ptr<osgShadow::ShadowMap> shadowMap;
    osg::ref_ptr<osg::LightSource> lightSource;
};

}
#endif
