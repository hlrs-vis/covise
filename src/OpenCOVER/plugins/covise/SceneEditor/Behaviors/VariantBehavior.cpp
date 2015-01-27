/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VariantBehavior.h"
#include "../SceneObject.h"

#include "../Events/SwitchVariantEvent.h"

VariantBehavior::VariantBehavior()
{
    _type = BehaviorTypes::VARIANT_BEHAVIOR;
}

VariantBehavior::~VariantBehavior()
{
}

int VariantBehavior::attach(SceneObject *so)
{
    // connects this behavior to its scene object
    Behavior::attach(so);

    std::map<std::string, VariantGroup>::iterator groupIt;
    for (groupIt = _groupList.begin(); groupIt != _groupList.end(); groupIt++)
    {
        VariantGroup *group = &groupIt->second;
        // loop over variants in group
        std::map<std::string, Variant>::iterator variantIt;
        for (variantIt = group->variantList.begin(); variantIt != group->variantList.end(); variantIt++)
        {
            Variant *variant = &variantIt->second;
            findNodes(variant, so->getGeometryNode((&variantIt->second)->geoNameSpace));
            // make all except first variant invisible
            if (variantIt != group->variantList.begin())
            {
                setVariantVisible(variant, false);
            }
        }
    }

    return 1;
}

int VariantBehavior::detach()
{
    std::map<std::string, VariantGroup>::iterator groupIt;
    for (groupIt = _groupList.begin(); groupIt != _groupList.end(); groupIt++)
    {
        VariantGroup *group = &groupIt->second;
        // loop over variants in group
        std::map<std::string, Variant>::iterator variantIt;
        for (variantIt = group->variantList.begin(); variantIt != group->variantList.end(); variantIt++)
        {
            Variant *variant = &variantIt->second;
            setVariantVisible(variant, true);
            variant->nodeList.clear();
        }
    }

    Behavior::detach();

    return 1;
}

EventErrors::Type VariantBehavior::receiveEvent(Event *e)
{
    if (e->getType() == EventTypes::SWITCH_VARIANT_EVENT)
    {
        SwitchVariantEvent *sve = dynamic_cast<SwitchVariantEvent *>(e);
        switchVariant(sve->getGroup(), sve->getVariant());
        return EventErrors::SUCCESS;
    }

    return EventErrors::UNHANDLED;
}

bool VariantBehavior::buildFromXML(QDomElement *behaviorElement)
{
    // read all groups
    QDomElement groupElem = behaviorElement->firstChildElement("group");
    while (!groupElem.isNull())
    {
        VariantGroup variantGroup;
        QDomElement variantElem = groupElem.firstChildElement("variant");
        while (!variantElem.isNull())
        {
            Variant variant;
            // regexp
            variant.regexp = QRegExp(variantElem.attribute("regexp", ""));
            variant.geoNameSpace = variantElem.attribute("namespace", "").toStdString();
            // add
            std::string name = variantElem.attribute("name", "").toStdString();
            variantGroup.variantList[name] = variant;
            // next
            variantElem = variantElem.nextSiblingElement("variant");
        }
        // add
        std::string name = groupElem.attribute("name", "").toStdString();
        _groupList[name] = variantGroup;
        // next
        groupElem = groupElem.nextSiblingElement("group");
    }
    return true;
}

void VariantBehavior::findNodes(Variant *variant, osg::Node *node)
{
    if (!node)
    {
        return;
    }
    if (variant->regexp.exactMatch(QString(node->getName().c_str())))
    {
        VariantNode vn;
        vn.node = node;
        variant->nodeList.push_back(vn);
    }
    else
    {
        osg::Group *g = node->asGroup();
        if (g != NULL)
        {
            for (int i = 0; i < g->getNumChildren(); i++)
            {
                findNodes(variant, g->getChild(i));
            }
        }
    }
}

void VariantBehavior::switchVariant(std::string groupName, std::string variantName)
{
    std::map<std::string, VariantGroup>::iterator it;
    it = _groupList.find(groupName);
    if (it != _groupList.end())
    {
        switchVariant(&it->second, variantName);
    }
}

void VariantBehavior::switchVariant(VariantGroup *group, std::string variantName)
{
    std::map<std::string, Variant>::iterator it;
    for (it = group->variantList.begin(); it != group->variantList.end(); it++)
    {
        setVariantVisible(&it->second, (variantName == it->first));
    }
}

void VariantBehavior::setVariantVisible(Variant *variant, bool visible)
{
    for (std::vector<VariantNode>::iterator nodeIt = variant->nodeList.begin(); nodeIt != variant->nodeList.end(); nodeIt++)
    {
        osg::Node *node = (*nodeIt).node;
        if (visible)
        {
            node->setNodeMask(node->getNodeMask() | opencover::Isect::Visible);
        }
        else
        {
            node->setNodeMask(node->getNodeMask() & ~opencover::Isect::Visible);
        }
        //      if (((*nodeIt).parents.size() == 0) && !visible)
        //      {
        //         // remove
        //         osg::Node::ParentList parents = node->getParents();
        //         for (osg::Node::ParentList::iterator parentIt = parents.begin(); parentIt != parents.end(); parentIt++)
        //         {
        //            (*parentIt)->removeChild(node);
        //            (*nodeIt).parents.push_back(*parentIt);
        //         }
        //      }
        //      else if (((*nodeIt).parents.size() > 0) && visible)
        //      {
        //         // add
        //         std::vector< osg::ref_ptr<osg::Group> >::iterator parentIt;
        //         for (parentIt = (*nodeIt).parents.begin(); parentIt != (*nodeIt).parents.end(); ++parentIt)
        //         {
        //            (*parentIt)->addChild(node);
        //         }
        //         (*nodeIt).parents.clear();
        //      }
    }
}
