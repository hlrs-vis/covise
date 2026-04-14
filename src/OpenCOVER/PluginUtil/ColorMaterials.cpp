#include "ColorMaterials.h"
#include <array>
using namespace opencover;
using namespace opencover::material;
std::array<osg::ref_ptr<osg::Material>, (int)material::LAST> materials; 
std::array<osg::Vec4, (int)material::LAST> materialsAmbient{osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f),
                                                            osg::Vec4(0.2f, 0.0f, 0.0f, 1.0f),
                                                            osg::Vec4(0.0f, 0.2f, 0.0f, 1.0f),
                                                            osg::Vec4(0.0f, 0.0f, 0.2f, 1.0f)
}; 

std::array<osg::Vec4, (int)material::LAST> materialsDiffuse{osg::Vec4(1, 1, 1, 1.0),
                                                            osg::Vec4(1, 0, 0, 1.0),
                                                            osg::Vec4(0, 1, 0, 1.0),
                                                            osg::Vec4(0, 0, 1, 1.0)
}; 

void initMaterial(osg::Material *m)
{
    m->setColorMode(osg::Material::OFF);
    m->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    m->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0, 0, 0, 1.0));
    m->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
}

osg::Material * initMaterial(Color c)
{
    int i = (int)c;
    auto &mat = materials[i];
    if(!mat)
    {
        mat = new osg::Material;
        mat->setDiffuse(osg::Material::FRONT_AND_BACK,materialsDiffuse[i]);
        mat->setAmbient(osg::Material::FRONT_AND_BACK,materialsAmbient[i]);
        initMaterial(mat.get());
    }
    return mat.get();
}

osg::Material *material::get(Color c)
{
    return initMaterial(c);
}

