#ifndef COVISE_PLUGIN_FLEX_CELL_VRMLNODES_H
#define COVISE_PLUGIN_FLEX_CELL_VRMLNODES_H


#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlNodeType.h>

#include <array>
#include <set>

class FlexCellNode : public vrml::VrmlNodeChild {
public:
    static void initFields(FlexCellNode *node, vrml::VrmlNodeType *t);
    static const char *typeName() { return "FlexCell"; }
    FlexCellNode(vrml::VrmlScene *scene);
    ~FlexCellNode();
    void send(size_t axis, float value);
    void bend();
    void switchWorkpiece(int variant);
    void switchWorkpieceInBender(int variant);
    // void attachPartToRobot(int variant);
    // void detachPartToRobot(int variant);
    void bendAnimation(int animation);

    // std::array<vrml::VrmlSFFloat, 7> axisNames;
};

extern std::set<FlexCellNode *> flexCellNodes;

#endif // COVISE_PLUGIN_FLEX_CELL_VRMLNODES_H