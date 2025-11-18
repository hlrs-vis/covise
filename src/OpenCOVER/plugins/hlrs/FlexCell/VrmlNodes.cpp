#include "VrmlNodes.h"

#include <cover/coVRPluginSupport.h>

std::set<FlexCellNode *> flexCellNodes;

void FlexCellNode::initFields(FlexCellNode *node, vrml::VrmlNodeType *t) {
    vrml::VrmlNodeChild::initFields(node, t);
    if(!t)
        return;
    constexpr int numAxes = 7;
    for(size_t i = 0; i < numAxes; ++i)
        t->addEventOut(("Achse" + std::to_string(i+1)).c_str(), vrml::VrmlField::SFFLOAT);
    t->addEventOut("Variant", vrml::VrmlField::SFINT32);
    t->addEventOut("pinch", vrml::VrmlField::SFBOOL);
    t->addEventOut("bend", vrml::VrmlField::SFBOOL);
    t->addEventOut("loosen", vrml::VrmlField::SFBOOL);
}

FlexCellNode::FlexCellNode(vrml::VrmlScene *scene)
: VrmlNodeChild(scene, typeName())
{
    flexCellNodes.insert(this);
}

FlexCellNode::~FlexCellNode()
{
    flexCellNodes.erase(this);
}

void FlexCellNode::send(size_t axis, float value)
{
    if(axis >= 7)
        return;
    vrml::VrmlSFFloat vrmlValue(value);
    eventOut(opencover::cover->frameTime(), ("Achse" + std::to_string(axis+1)).c_str(), vrmlValue);
}

void FlexCellNode::bend()
{
    vrml::VrmlSFTime vrmlValue(opencover::cover->frameTime());
    eventOut(opencover::cover->frameTime(), "Bend", vrmlValue);
}

void FlexCellNode::switchWorkpiece(int variant)
{
    vrml::VrmlSFInt vrmlValue(variant);
    eventOut(opencover::cover->frameTime(), "Variant", vrmlValue);
}

void FlexCellNode::bendAnimation(int animation)
{
    switch (animation)
    {
    case 0: //nothing
        break;
    case 1: // pinch
        eventOut(opencover::cover->frameTime(), "pinch", vrml::VrmlSFBool(true));
        break;
    case 2: // bend
        eventOut(opencover::cover->frameTime(), "bend", vrml::VrmlSFBool(true));
        break;
    case 3: // loosen
        eventOut(opencover::cover->frameTime(), "loosen", vrml::VrmlSFBool(true));
        break;
    default:
        break;
    }
}