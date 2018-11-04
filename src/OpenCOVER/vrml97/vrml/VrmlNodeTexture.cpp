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

}
