#include "cover/coVRShadowManager.h"
#include "cover/coVRPluginSupport.h"
#include "cover/VRSceneGraph.h"
#include <config/CoviseConfig.h>
using namespace opencover;
coVRShadowManager* coVRShadowManager::inst=NULL;
coVRShadowManager::coVRShadowManager()
{
    std::string tech = covise::coCoviseConfig::getEntry("value","COVER.ShadowTechnique","none");
    
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

void coVRShadowManager::setLight(osg::LightSource *ls)
{
    lightSource = ls;
    if (technique=="SoftShadowMap" && softSM.get()!=NULL)
    {
        softSM->setLight(lightSource->getLight());
    }
    else if (technique=="StandardShadowMap" && standardSM.get()!=NULL)
    {
        standardSM->setLight(lightSource->getLight());
    }
    else if (technique=="LightSpacePerspectiveShadowMapVB" && lspsm.get()!=NULL)
    {
        lspsm->setLight(lightSource->getLight());
    }
    else if (technique=="LightSpacePerspectiveShadowMapCB" && lspsmcb.get()!=NULL)
    {
        lspsmcb->setLight(lightSource->getLight());
    }
    else if (technique=="LightSpacePerspectiveShadowMapDB" && lspsmdb.get()!=NULL)
    {
        lspsmdb->setLight(lightSource->getLight());
    }
    else if (technique=="ShadowMap" && shadowMap.get()!=NULL)
    {
        shadowMap->setLight(lightSource.get());
    }
}
void coVRShadowManager::setTechnique(const std::string &tech)
{
    if(technique == tech)
        return;
    technique = tech;
    osgShadow::ShadowedScene *shadowedScene = opencover::cover->getScene();
    if (technique=="ShadowVolume")
    {
        sv = new osgShadow::ShadowVolume;
        sv->setDynamicShadowVolumes(false);
        sv->setDrawMode(osgShadow::ShadowVolumeGeometry::STENCIL_TWO_SIDED);
        //sv->setDrawMode(osgShadow::ShadowVolumeGeometry::STENCIL_TWO_PASS);
        shadowedScene->setShadowTechnique(sv.get());
    }
    else if (technique=="ShadowTexture")
    {
        st = new osgShadow::ShadowTexture;
        shadowedScene->setShadowTechnique(st.get());
    }
    else if (technique=="SoftShadowMap")
    {
        softSM = new osgShadow::SoftShadowMap;
        shadowedScene->setShadowTechnique(softSM.get());
        if(lightSource.get())
        {
            softSM->setLight(lightSource->getLight());
        }
    }
    else if (technique=="StandardShadowMap")
    {
        standardSM = new osgShadow::StandardShadowMap;
        shadowedScene->setShadowTechnique(standardSM.get());
        if(lightSource.get())
        {
            standardSM->setLight(lightSource->getLight());
        }
    }
    //else if (technique.compare("ParallelSplitShadowMap")==0)
    //{
    //}
    else if (technique=="LightSpacePerspectiveShadowMapVB")
    {
        osg::ref_ptr<osgShadow::MinimalShadowMap> lspsm = new osgShadow::LightSpacePerspectiveShadowMapVB;
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
        shadowedScene->setShadowTechnique(lspsm.get());
        if(lightSource.get())
        {
            lspsm->setLight(lightSource->getLight());
        }
    }
    else if (technique=="LightSpacePerspectiveShadowMapCB")
    {
        lspsmcb = new osgShadow::LightSpacePerspectiveShadowMapCB;
        shadowedScene->setShadowTechnique(lspsmcb.get());
        if(lightSource.get())
        {
            lspsmcb->setLight(lightSource->getLight());
        }
    }
    else if (technique=="LightSpacePerspectiveShadowMapDB")
    {
        lspsmdb = new osgShadow::LightSpacePerspectiveShadowMapDB;
        shadowedScene->setShadowTechnique(lspsmdb.get());
        if(lightSource.get())
        {
            lspsmdb->setLight(lightSource->getLight());
        }
    }
    else if (technique=="ShadowMap")
    {
        shadowMap = new osgShadow::ShadowMap;
        shadowedScene->setShadowTechnique(shadowMap.get());

        int mapres = 4096;
        shadowMap->setTextureSize(osg::Vec2s(mapres,mapres));
        if(lightSource.get())
        {
            shadowMap->setLight(lightSource.get());
        }
    }
    else
    {
        shadowedScene->setShadowTechnique(NULL); // no shadow
    }
}