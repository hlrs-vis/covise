/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _AURAL_REALITY_SELECTION_H
#define _AURAL_REALITY_SELECTION_H

#include <set>

#include <PluginUtil/coSensor.h>
#include <osg/Node>

class Selectable
{
public:
    void select();
    void deselect();
    bool isSelected() const;

protected:
    virtual void updateSelection() { };
    bool m_isSelected = false;
};

class Selection
{
public:
    void selectSingle(Selectable *selectable);
    void removeFromSelection(Selectable *selectable);
    void toggleSelection(Selectable *selectable);
    void addToSelection(Selectable *selectable);
    const std::set<Selectable *> getSelected() const;

protected:
    std::set<Selectable *> m_selected;
};

class SelectableSensor : public coPickSensor
{
private:
    Selection *selection;
    Selectable *selectable;

public:
    SelectableSensor(Selection *selection, Selectable *selectable, osg::Node *node);
    ~SelectableSensor();
    void activate() override;
    void disactivate() override;
};

#endif
