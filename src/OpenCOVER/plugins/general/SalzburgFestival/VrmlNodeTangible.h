#ifndef VRML_NODE_TANGIBLE_H
#define VRML_NODE_TANGIBLE_H

#include <util/coExport.h>

#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlSFFloat.h>

class VRML97COVEREXPORT VrmlNodeTangible : public vrml::VrmlNodeChild
{
public:
    VrmlNodeTangible(vrml::VrmlScene *scene = 0);
    VrmlNodeTangible(const VrmlNodeTangible &n);
    virtual ~VrmlNodeTangible();

    static void initFields(VrmlNodeTangible *node, vrml::VrmlNodeType *t);
    static const char *typeName();

    virtual VrmlNodeTangible *toTangible() const;

    static const std::vector<VrmlNodeTangible *> &getAllNodeTangibles();

    void setField(const char *fieldName, const vrml::VrmlField &fieldValue);
    const vrml::VrmlField *getField(const char *fieldName) const;

    void setAngle(float a);

private:
    vrml::VrmlSFFloat d_angle;

    static std::vector<VrmlNodeTangible *> allNodeTangibles;
};

#endif // VRML_NODE_TANGIBLE_H
