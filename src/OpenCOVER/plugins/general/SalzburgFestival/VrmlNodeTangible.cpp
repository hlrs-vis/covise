#include <algorithm> 
#include <vector>

#include <vrml97/vrml/System.h>

#include "VrmlNodeTangible.h"

using namespace vrml;

std::vector<VrmlNodeTangible*> VrmlNodeTangible::allNodeTangibles;

VrmlNodeTangible::VrmlNodeTangible(VrmlScene *scene)
    : VrmlNodeChild(scene, typeName()), d_angle(0.0f)
{
    allNodeTangibles.push_back(this);
    setModified();
}

VrmlNodeTangible::VrmlNodeTangible(const VrmlNodeTangible &n)
    : VrmlNodeChild(n), d_angle(n.d_angle)
{
    allNodeTangibles.push_back(this);
    setModified();
}

VrmlNodeTangible::~VrmlNodeTangible()
{
    auto it = std::find(allNodeTangibles.begin(), allNodeTangibles.end(), this);
    if (it != allNodeTangibles.end())
        allNodeTangibles.erase(it);
}

void VrmlNodeTangible::initFields(VrmlNodeTangible *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    if (t)
    {
        t->addExposedField("angle", VrmlField::SFFLOAT);
        t->addEventOut("angle_changed", VrmlField::SFFLOAT);
    }
}

const char *VrmlNodeTangible::typeName()
{
    return "Tangible";
}

VrmlNodeTangible *VrmlNodeTangible::toTangible() const
{
    return (VrmlNodeTangible *)(this);
}

const std::vector<VrmlNodeTangible*> &VrmlNodeTangible::getAllNodeTangibles()
{
    return allNodeTangibles;
}

void VrmlNodeTangible::setField(const char *fieldName, const VrmlField &fieldValue)
{
    if (strcmp(fieldName, "angle") == 0)
    {
        const VrmlSFFloat *f = fieldValue.toSFFloat();
        if (f)
        {
            d_angle = f->get();
            setModified();
        }
    }
    else
    {
        VrmlNodeChild::setField(fieldName, fieldValue);
    }
}

const VrmlField *VrmlNodeTangible::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "angle") == 0)
        return &d_angle;
    return VrmlNodeChild::getField(fieldName);
}

void VrmlNodeTangible::setAngle(float angle)
{
    d_angle = angle;
    eventOut(System::the->time(), "angle_changed", d_angle);
    setModified();
}
