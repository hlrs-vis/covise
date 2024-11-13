#include "VrmlNodeChild.h"

using namespace vrml;

void VrmlNodeChild::initFields(VrmlNodeChild *node, VrmlNodeType *t)
{
    //space for future implementations
}

VrmlNodeChild::VrmlNodeChild(VrmlScene *scene, const std::string& name)
    : VrmlNode(scene, name)
{
}