/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SceneObjectCreator.h"

#include "Behaviors/TransformBehavior.h"
#include "Behaviors/SinusScalingBehavior.h"
#include "Behaviors/HighlightBehavior.h"
#include "Behaviors/AppearanceBehavior.h"
#include "Behaviors/MountBehavior.h"
#include "Behaviors/CameraBehavior.h"
#include "Behaviors/KinematicsBehavior.h"
#include "Behaviors/VariantBehavior.h"

#include <iostream>

SceneObjectCreator::SceneObjectCreator()
{
}

SceneObjectCreator::~SceneObjectCreator()
{
}

SceneObject *SceneObjectCreator::createFromXML(QDomElement *root)
{
    SceneObject *so = new SceneObject();
    if (!buildFromXML(so, root))
    {
        delete so;
        return NULL;
    }
    return so;
}

bool SceneObjectCreator::buildFromXML(SceneObject *so, QDomElement *root)
{
    // read name
    QDomElement classElem = root->firstChildElement("classification");
    if (!classElem.isNull())
    {
        QDomElement nameElem = classElem.firstChildElement("name");
        if (!nameElem.isNull())
            so->setName(nameElem.attribute("value").toStdString().c_str());
    }
    // read behaviors
    _createBehaviorsFromXML(so, root);
    return true;
}

bool SceneObjectCreator::_createBehaviorsFromXML(SceneObject *so, QDomElement *root)
{
    QDomElement behaveRoot = root->firstChildElement("behavior");
    if (!behaveRoot.isNull())
    {
        QDomElement b = behaveRoot.firstChildElement();
        while (!b.isNull())
        {
            _createBehaviorFromXML(so, &b); // ignore faulty behaviors and continue
            b = b.nextSiblingElement();
        }
    }
    return true;
}

bool SceneObjectCreator::_createBehaviorFromXML(SceneObject *so, QDomElement *behaviorElement)
{
    Behavior *b = NULL;
    std::string bString = behaviorElement->tagName().toStdString();

    if (bString == "TransformBehavior")
    {
        b = new TransformBehavior();
    }
    else if (bString == "SinusScalingBehavior")
    {
        b = new SinusScalingBehavior();
    }
    else if (bString == "HighlightBehavior")
    {
        b = new HighlightBehavior();
    }
    else if (bString == "AppearanceBehavior")
    {
        b = new AppearanceBehavior();
    }
    else if (bString == "MountBehavior")
    {
        b = new MountBehavior();
    }
    else if (bString == "CameraBehavior")
    {
        b = new CameraBehavior();
    }
    else if (bString == "KinematicsBehavior")
    {
        b = new KinematicsBehavior();
    }
    else if (bString == "VariantBehavior")
    {
        b = new VariantBehavior();
    }

    if (b == NULL)
    {
        std::cerr << "Error: Ignoring unknown behavior: " << bString << std::endl;
        return false;
    }

    if (!b->buildFromXML(behaviorElement))
    {
        delete b;
        return false;
    }

    if (so->addBehavior(b) != 1)
    {
        delete b;
        return false;
    }

    return true;
}
