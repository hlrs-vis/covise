#include "VrmlNodeTexture.h"

namespace vrml
{

bool VrmlNodeTexture::s_useTextureNPOT = true;
int VrmlNodeTexture::s_maxTextureSize = 65536;

void VrmlNodeTexture::enableTextureNPOT(bool flag)
{
    s_useTextureNPOT = flag;
}

bool VrmlNodeTexture::useTextureNPOT()
{
    return s_useTextureNPOT;
}

void VrmlNodeTexture::setMaxTextureSize(int size)
{
    s_maxTextureSize = size;
}

int VrmlNodeTexture::maxTextureSize()
{
    return s_maxTextureSize;
}



VrmlNodeType *VrmlNodeTexture::defineType(VrmlNodeType *t)
{
    return VrmlNode::defineType(t);
}

void VrmlNodeTexture::initFields(VrmlNodeTexture *node, VrmlNodeType *t)
{
    //open for future implementations
}

VrmlNodeTexture::VrmlNodeTexture(VrmlScene *s, const std::string &name)
    : VrmlNodeTemplate(s, name)
{
    d_blendModeOverwrite = -1;
}

VrmlNodeTexture *VrmlNodeTexture::toTexture() const
{
    return (VrmlNodeTexture *)this;
}

}
