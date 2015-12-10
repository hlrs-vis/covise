#include "cover/coVRShadowManager.h"
#include "cover/coVRPluginSupport.h"
#include "cover/VRSceneGraph.h"
#include <config/CoviseConfig.h>
#include <osgShadow/ShadowVolume>
#include <osgShadow/ShadowTexture>
#include <osgShadow/SoftShadowMap>
#include <osgShadow/StandardShadowMap>
#include <osgShadow/MinimalShadowMap>
#include <osgShadow/LightSpacePerspectiveShadowMap>
using namespace opencover;
coVRShadowManager* coVRShadowManager::inst=NULL;
coVRShadowManager::coVRShadowManager()
{
    std::string tech = covise::coCoviseConfig::getEntry("value","COVER.ShadowTechnique","ShadowVolume");
    
    osgShadow::ShadowedScene *shadowedScene = opencover::cover->getScene();
    
    shadowedScene->setReceivesShadowTraversalMask(Isect::ReceiveShadow);
    shadowedScene->setCastsShadowTraversalMask(Isect::CastShadow);
    setTechnique(tech);
}
coVRShadowManager* coVRShadowManager::instance()
    {
        if(inst == NULL)
        {
            inst = new coVRShadowManager();
        }
        return inst;
}
void coVRShadowManager::setTechnique(const std::string &tech)
{
    if(technique == tech)
        return;
    technique = tech;
    osgShadow::ShadowedScene *shadowedScene = opencover::cover->getScene();
    if (technique=="ShadowVolume")
    {
        osg::ref_ptr<osgShadow::ShadowVolume> sv = new osgShadow::ShadowVolume;
        sv->setDynamicShadowVolumes(false);
        sv->setDrawMode(osgShadow::ShadowVolumeGeometry::STENCIL_TWO_SIDED);
        //sv->setDrawMode(osgShadow::ShadowVolumeGeometry::STENCIL_TWO_PASS);
        shadowedScene->setShadowTechnique(sv.get());
    }
    else if (technique=="ShadowTexture")
    {
        osg::ref_ptr<osgShadow::ShadowTexture> st = new osgShadow::ShadowTexture;
        shadowedScene->setShadowTechnique(st.get());
    }
    else if (technique=="SoftShadowMap")
    {
        osg::ref_ptr<osgShadow::SoftShadowMap> sm = new osgShadow::SoftShadowMap;
        shadowedScene->setShadowTechnique(sm.get());
    }
    else if (technique=="StandardShadowMap")
    {
        osg::ref_ptr<osgShadow::StandardShadowMap> st = new osgShadow::StandardShadowMap;
        shadowedScene->setShadowTechnique(st.get());
    }
    //else if (technique.compare("ParallelSplitShadowMap")==0)
    //{
    //}
    else if (technique=="LightSpacePerspectiveShadowMapVB")
    {
        osg::ref_ptr<osgShadow::MinimalShadowMap> sm = new osgShadow::LightSpacePerspectiveShadowMapVB;
        /*unsigned int texSize = 2048;
        //float minLightMargin = 100000.f;
        //float maxFarPlane = 5;
        //unsigned int baseTexUnit = 0;
        unsigned int shadowTexUnit = 1;
        //sm->setMinLightMargin( minLightMargin );
        //sm->setMaxFarPlane( maxFarPlane );
        sm->setTextureSize( osg::Vec2s( texSize, texSize ) );
        sm->setShadowTextureCoordIndex( shadowTexUnit );
        sm->setShadowTextureUnit( shadowTexUnit );*/
        shadowedScene->setShadowTechnique(sm.get());
    }
    else if (technique=="LightSpacePerspectiveShadowMapCB")
    {
        osg::ref_ptr<osgShadow::MinimalShadowMap> sm = new osgShadow::LightSpacePerspectiveShadowMapCB;
        shadowedScene->setShadowTechnique(sm.get());
    }
    else if (technique=="LightSpacePerspectiveShadowMapDB")
    {
        osg::ref_ptr<osgShadow::MinimalShadowMap> sm = new osgShadow::LightSpacePerspectiveShadowMapDB;
        shadowedScene->setShadowTechnique(sm.get());
    }
    else if (technique=="ShadowMap")
    {
        osg::ref_ptr<osgShadow::ShadowMap> sm = new osgShadow::ShadowMap;
        shadowedScene->setShadowTechnique(sm.get());

        int mapres = 1024;
        sm->setTextureSize(osg::Vec2s(mapres,mapres));
    }
    else
    {
        shadowedScene->setShadowTechnique(NULL); // no shadow
    }
}