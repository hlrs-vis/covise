/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Selection.h"

void Selectable::select()
{
    if (m_isSelected)
        return;
    m_isSelected = true;
    updateSelection();
}

void Selectable::deselect()
{
    if (!m_isSelected)
        return;
    m_isSelected = false;
    updateSelection();
}

bool Selectable::isSelected() const { return m_isSelected; }

void Selection::selectSingle(Selectable *selectable)
{
    for (auto i : m_selected)
    {
        if (i != selectable)
        {
            i->deselect();
            m_selected.erase(i);
        }
    }

    addToSelection(selectable);
}

void Selection::removeFromSelection(Selectable *selectable)
{
    if (m_selected.find(selectable) != m_selected.end())
    {
        selectable->deselect();
        m_selected.erase(selectable);
    }
}

void Selection::toggleSelection(Selectable *selectable)
{
    if (selectable->isSelected())
    {
        removeFromSelection(selectable);
    }
    else
    {
        addToSelection(selectable);
    }
}

void Selection::addToSelection(Selectable *selectable)
{
    if (m_selected.find(selectable) == m_selected.end())
    {
        m_selected.insert(selectable);
        selectable->select();
    }
}

const std::set<Selectable *> Selection::getSelected() const
{
    return m_selected;
}

SelectableSensor::SelectableSensor(Selection *selection, Selectable *selectable, osg::Node *node)
    : coPickSensor(node, false, vrui::coInteraction::AllButtons, vrui::coInteraction::Medium)
    , selection(selection)
    , selectable(selectable)
{
}
SelectableSensor::~SelectableSensor()
{
    if (active)
        disactivate();
}
void SelectableSensor::activate()
{
    selection->addToSelection(selectable);
}

void SelectableSensor::disactivate()
{
    // selection->removeFromSelection(selectable);
}
