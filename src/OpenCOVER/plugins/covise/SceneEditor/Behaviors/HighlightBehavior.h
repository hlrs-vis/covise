/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef HIGHLIGHT_BEHAVIOR_H
#define HIGHLIGHT_BEHAVIOR_H

#include "Behavior.h"
#include "../Events/MouseEnterEvent.h"
#include "../Events/MouseExitEvent.h"
#include "../Events/SelectEvent.h"
#include "../Events/DeselectEvent.h"

#include <osg/StateSet>
#include <osg/Material>
#include <osg/Program>
#include <osgFX/Outline>

class HighlightBehavior : public Behavior
{
public:
    HighlightBehavior();
    virtual ~HighlightBehavior();

    virtual int attach(SceneObject *);
    virtual int detach();
    virtual EventErrors::Type receiveEvent(Event *e);

    virtual bool buildFromXML(QDomElement *behaviorElement);

private:
    void _updateHighlight();
    void _addHighlight(osg::Vec4 color, float width);
    void _removeHighlight();

    bool _outlineHL;
    osg::ref_ptr<osgFX::Outline> _overrideNodeFX;

    osg::ref_ptr<osg::Group> _overrideNode;
    osg::ref_ptr<osg::Program> _emptyProgram;
    osg::ref_ptr<osg::StateSet> _stateSet;
    osg::ref_ptr<osg::Material> _material;

    bool _highlightWhenTouched;

    bool _isTouched;
    bool _isSelected;
};

#endif
