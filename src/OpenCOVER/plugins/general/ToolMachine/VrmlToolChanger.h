#ifndef COVER_PLUGIN_TOOLCHANGER_VRMLNODE_H
#define COVER_PLUGIN_TOOLCHANGER_VRMLNODE_H

#include "LogicInterface.h"

#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlNodeType.h>

#include <osg/MatrixTransform>
#include <utils/pointer/NullCopyPtr.h>



class ToolChangerNode : public vrml::VrmlNodeChild {
public:
    ToolChangerNode(vrml::VrmlScene *scene);
    ~ToolChangerNode();
    static void initFields(ToolChangerNode *node, vrml::VrmlNodeType *t);
    static const char *name() { return "ToolChangerNode"; }

    vrml::VrmlSFString arm;
    vrml::VrmlSFString changer;
    vrml::VrmlSFString cover;
    vrml::VrmlSFNode toolHead;
    vrml::VrmlSFString toolMagazineName;

    opencover::utils::pointer::NullCopyPtr<LogicInterface> toolChanger;

};

extern std::set<ToolChangerNode *> toolChangers;


osg::MatrixTransform *toOsg(vrml::VrmlNode *node);

#endif // COVER_PLUGIN_TOOLCHANGER_VRMLNODE_H
