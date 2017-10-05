#include "coVRShadowManager.h"
#include "VRSceneGraph.h"
#include "coVRPluginSupport.h"

using namespace opencover;

coVRShadowManager* coVRShadowManager::inst=NULL;

class coShadowedScene: public osgShadow::ShadowedScene
{
public:
    using osgShadow::ShadowedScene::setNumChildrenRequiringUpdateTraversal;
};

coVRShadowManager::coVRShadowManager()
{
    assert(!inst);

    technique = "undefined";

    shadowMap = new osgShadow::ShadowMap;
    lspsmdb = new osgShadow::LightSpacePerspectiveShadowMapDB;
    lspsmcb = new osgShadow::LightSpacePerspectiveShadowMapCB;
    lspsm = new osgShadow::LightSpacePerspectiveShadowMapVB;
    standardSM = new osgShadow::StandardShadowMap;
    softSM = new osgShadow::SoftShadowMap;
    st = new osgShadow::ShadowTexture;
    sv = new osgShadow::ShadowVolume;
}

coVRShadowManager::~coVRShadowManager()
{
    setTechnique("none");
    inst = NULL;
}


coVRShadowManager* coVRShadowManager::instance()
{
    if(inst == NULL)
    {
        inst = new coVRShadowManager();
    }
    return inst;
}

osgShadow::ShadowedScene *coVRShadowManager::newScene()
{
    auto shadowedScene = new coShadowedScene;

    shadowedScene->setReceivesShadowTraversalMask(Isect::ReceiveShadow);
    shadowedScene->setCastsShadowTraversalMask(Isect::CastShadow);
    shadowedScene->setShadowTechnique(nullptr);

    if (shadowedScene->getShadowTechnique() == nullptr)
    {
        shadowedScene->setNumChildrenRequiringUpdateTraversal(shadowedScene->getNumChildrenRequiringUpdateTraversal()-1);
    }

    return shadowedScene;
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

    osgShadow::ShadowedScene *shadowedScene = dynamic_cast<osgShadow::ShadowedScene *>(opencover::cover->getScene());
    if (!shadowedScene) {
        std::cerr << "coVRShadowManager: scene is not a ShadowedScene" << std::endl;
        return;
    }

    bool haveTechnique = shadowedScene->getShadowTechnique() != nullptr;

    technique = tech;
    if (technique=="ShadowVolume")
    {
        if(sv.get()==NULL)
        {
            sv = new osgShadow::ShadowVolume;
        }
        sv->setDynamicShadowVolumes(false);
        sv->setDrawMode(osgShadow::ShadowVolumeGeometry::STENCIL_TWO_SIDED);
        //sv->setDrawMode(osgShadow::ShadowVolumeGeometry::STENCIL_TWO_PASS);
        shadowedScene->setShadowTechnique(sv.get());
    }
    else if (technique=="ShadowTexture")
    {
        if(st.get()==NULL)
        {
            st = new osgShadow::ShadowTexture;
        }
        shadowedScene->setShadowTechnique(st.get());
    }
    else if (technique=="SoftShadowMap")
    {
        if(softSM.get()==NULL)
        {
            softSM = new osgShadow::SoftShadowMap;
        }
        shadowedScene->setShadowTechnique(softSM.get());
        if(lightSource.get())
        {
            softSM->setLight(lightSource->getLight());
        }
    }
    else if (technique=="StandardShadowMap")
    {
        if(standardSM.get()==NULL)
        {
            standardSM = new osgShadow::StandardShadowMap;
        }
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
        if(lspsm.get()==NULL)
        {
            lspsm = new osgShadow::LightSpacePerspectiveShadowMapVB;
        }
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
        if(lspsmcb.get()==NULL)
        {
            lspsmcb = new osgShadow::LightSpacePerspectiveShadowMapCB;
        }
        shadowedScene->setShadowTechnique(lspsmcb.get());
        if(lightSource.get())
        {
            lspsmcb->setLight(lightSource->getLight());
        }
    }
    else if (technique=="LightSpacePerspectiveShadowMapDB")
    {
        if(lspsmdb.get()==NULL)
        {
            lspsmdb = new osgShadow::LightSpacePerspectiveShadowMapDB;
        }
        shadowedScene->setShadowTechnique(lspsmdb.get());
        if(lightSource.get())
        {
            lspsmdb->setLight(lightSource->getLight());
        }
    }
    else if (technique=="ShadowMap")
    {
        if(shadowMap.get()==NULL)
        {
            shadowMap = new osgShadow::ShadowMap;
        }
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
        shadowedScene->cleanSceneGraph();
    }

    coShadowedScene *scene = dynamic_cast<coShadowedScene *>(shadowedScene);
    if (!scene)
        return;

    if (scene->getShadowTechnique() == nullptr)
    {
        if (haveTechnique)
        {
            scene->setNumChildrenRequiringUpdateTraversal(shadowedScene->getNumChildrenRequiringUpdateTraversal()-1);
        }
    }
    else
    {
        if (!haveTechnique)
        {
            scene->setNumChildrenRequiringUpdateTraversal(shadowedScene->getNumChildrenRequiringUpdateTraversal()+1);
        }
    }
}

std::string coVRShadowManager::getTechnique()
{
    return technique;
}

void coVRShadowManager::setSoftnessWidth(float w)
{
    if(softSM) softSM->setSoftnessWidth(w);
}

void coVRShadowManager::setJitteringScale(float s)
{
    if(softSM) softSM->setJitteringScale(s);
}

void coVRShadowManager::setTextureSize(osg::Vec2s ts)
{
    if(softSM) softSM->setTextureSize(ts);
    if(shadowMap) shadowMap->setTextureSize(ts);
    if(standardSM) standardSM->setTextureSize(ts);
    if(lspsm) lspsm->setTextureSize(ts);
    if(lspsmcb) lspsmcb->setTextureSize(ts);
    if(lspsmdb) lspsmdb->setTextureSize(ts);
}
