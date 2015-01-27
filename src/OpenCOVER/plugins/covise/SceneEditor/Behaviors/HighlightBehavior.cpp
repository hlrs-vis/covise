/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "HighlightBehavior.h"
#include "../Asset.h"
#include "../SceneUtils.h"

#include <osg/Material>

#include <iostream>
#include <math.h>

HighlightBehavior::HighlightBehavior()
{
    _type = BehaviorTypes::HIGHLIGHT_BEHAVIOR;

    _highlightWhenTouched = false;

    _outlineHL = false; // NOTE: Outline is much slower

    _overrideNode = NULL;
    _overrideNodeFX = NULL;
    _stateSet = NULL;

    _isTouched = false;
    _isSelected = false;
}

HighlightBehavior::~HighlightBehavior()
{
}

int HighlightBehavior::attach(SceneObject *so)
{
    // connects this behavior to its scene object
    Behavior::attach(so);

    return 1;
}

int HighlightBehavior::detach()
{
    if (_overrideNode)
    {
        SceneUtils::removeNode(_overrideNode.get());
        _overrideNode = NULL;
        _stateSet = NULL;
        _material = NULL;
    }
    if (_overrideNodeFX)
    {
        SceneUtils::removeNode(_overrideNodeFX.get());
        _overrideNodeFX = NULL;
    }

    Behavior::detach();

    return 1;
}

EventErrors::Type HighlightBehavior::receiveEvent(Event *e)
{
    if (e->getType() == EventTypes::MOUSE_ENTER_EVENT)
    {
        if (_highlightWhenTouched)
        {
            _isTouched = true;
            _updateHighlight();
        }
    }
    else if (e->getType() == EventTypes::MOUSE_EXIT_EVENT)
    {
        if (_highlightWhenTouched)
        {
            _isTouched = false;
            _updateHighlight();
        }
    }
    else if (e->getType() == EventTypes::SELECT_EVENT)
    {
        _isSelected = true;
        _updateHighlight();
    }
    else if (e->getType() == EventTypes::DESELECT_EVENT)
    {
        _isSelected = false;
        _updateHighlight();
    }

    return EventErrors::UNHANDLED;
}

bool HighlightBehavior::buildFromXML(QDomElement *behaviorElement)
{
    (void)behaviorElement;
    return true;
}

void HighlightBehavior::_updateHighlight()
{
    if (_isSelected)
    {
        _addHighlight(osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f), 2.0f);
    }
    else if (_isTouched)
    {
        _addHighlight(osg::Vec4(1.0f, 1.0f, 0.0f, 1.0f), 1.0f);
    }
    else
    {
        _removeHighlight();
    }
}

void HighlightBehavior::_addHighlight(osg::Vec4 color, float width)
{
    if (_outlineHL)
    {
        if (!_overrideNodeFX)
        {
            _overrideNodeFX = new osgFX::Outline();
            SceneUtils::insertNode(_overrideNodeFX.get(), _sceneObject);
        }
        _overrideNodeFX->setColor(color);
        _overrideNodeFX->setWidth(width);
        _overrideNodeFX->setEnabled(true);
    }
    else
    {
        if (!_overrideNode)
        {
            // node
            _overrideNode = new osg::Group();
            SceneUtils::insertNode(_overrideNode.get(), _sceneObject);
            // stateSet
            _stateSet = new osg::StateSet();
            _overrideNode->setStateSet(_stateSet);
            // material
            _material = new osg::Material();
            _material->setColorMode(osg::Material::OFF);
            _material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
            _material->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
            _material->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
            _material->setAlpha(osg::Material::FRONT_AND_BACK, 1.0f);
            // program
            _emptyProgram = new osg::Program();
        }
        _material->setAmbient(osg::Material::FRONT_AND_BACK, color * 0.5f);
        _material->setDiffuse(osg::Material::FRONT_AND_BACK, color);

        _stateSet->setAttributeAndModes(_emptyProgram, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::ON);

        _stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::ON);
        _stateSet->setMode(GL_NORMALIZE, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::ON);
        _stateSet->setMode(GL_BLEND, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::OFF);

        _stateSet->setAttributeAndModes(_material, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::ON);
        _stateSet->setMode(osg::StateAttribute::PROGRAM, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::OFF);
        _stateSet->setMode(osg::StateAttribute::VERTEXPROGRAM, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::OFF);
        _stateSet->setMode(osg::StateAttribute::FRAGMENTPROGRAM, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::OFF);
        _stateSet->setMode(osg::StateAttribute::TEXTURE, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::OFF);
    }
}

void HighlightBehavior::_removeHighlight()
{
    if (_overrideNode)
    {
        _stateSet->setGlobalDefaults();
        _stateSet->removeAttribute(_emptyProgram);
    }
    if (_overrideNodeFX)
    {
        _overrideNodeFX->setEnabled(false);
    }
}
