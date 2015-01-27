/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VARIANT_BEHAVIOR_H
#define VARIANT_BEHAVIOR_H

#include "Behavior.h"

#include <QRegExp>

#include <osg/Node>

struct VariantNode
{
    osg::ref_ptr<osg::Node> node;
    std::vector<osg::ref_ptr<osg::Group> > parents; // set when node is beeing removed, cleared when node is beeing added again
};

struct Variant
{
    QRegExp regexp;
    std::string geoNameSpace;
    std::vector<VariantNode> nodeList; // stores the nodes on attach (valid as long as the Behavior is attached, empty otherwise)
};

struct VariantGroup
{
    std::map<std::string, Variant> variantList;
};

class VariantBehavior : public Behavior
{
public:
    VariantBehavior();
    virtual ~VariantBehavior();

    virtual int attach(SceneObject *);
    virtual int detach();

    virtual EventErrors::Type receiveEvent(Event *e);

    virtual bool buildFromXML(QDomElement *behaviorElement);

private:
    void findNodes(Variant *variant, osg::Node *node);

    void switchVariant(std::string groupName, std::string variantName);
    void switchVariant(VariantGroup *group, std::string variantName);
    void setVariantVisible(Variant *variant, bool visible);

    std::map<std::string, VariantGroup> _groupList;
};

#endif
