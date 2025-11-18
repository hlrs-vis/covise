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
    t->addEventOut("variant", vrml::VrmlField::SFINT32);
    t->addEventOut("variantInBender", vrml::VrmlField::SFINT32);
    t->addEventOut("pinch", vrml::VrmlField::SFTIME);
    t->addEventOut("bend", vrml::VrmlField::SFTIME);
    t->addEventOut("loosen", vrml::VrmlField::SFTIME);
    // t->addEventOut("attachPartToRobot", vrml::VrmlField::SFINT32);
    // t->addEventOut("detachPartToRobot", vrml::VrmlField::SFINT32);
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
    eventOut(opencover::cover->frameTime(), "variant", vrmlValue);
}

void FlexCellNode::switchWorkpieceInBender(int variant)
{
    vrml::VrmlSFInt vrmlValue(variant);
    eventOut(opencover::cover->frameTime(), "variantInBender", vrmlValue);
}

// void FlexCellNode::attachPartToRobot(int variant)
// {
//     vrml::VrmlSFInt vrmlValue(variant);
//     eventOut(opencover::cover->frameTime(), "attachPartToRobot", vrmlValue);
// }

// void FlexCellNode::detachPartToRobot(int variant)
// {
//     vrml::VrmlSFInt vrmlValue(variant);
//     eventOut(opencover::cover->frameTime(), "detachPartToRobot", vrmlValue);
// }

void FlexCellNode::bendAnimation(int animation)
{
    auto now = opencover::cover->frameTime();
    auto vrmlValue = vrml::VrmlSFTime(now);
    switch (animation)
    {
    case 0: //nothing
        break;
    case 1: // pinch
        eventOut(now, "pinch", vrmlValue);
        break;
    case 2: // bend
        eventOut(now, "bend", vrmlValue);
        break;
    case 3: // loosen
        eventOut(now, "loosen", vrmlValue);
        break;
    default:
        break;
    }
}