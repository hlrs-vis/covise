/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   23.03.2010
**
**************************************************************************/

#include "dataelementcommands.hpp"

#include "src/data/dataelement.hpp"
#include "src/data/projectdata.hpp"

//################//
// Select         //
//################//

SelectDataElementCommand::SelectDataElementCommand(DataElement *element, DataCommand *parent)
    : DataCommand(parent)
    , lastTime_(QTime::currentTime())
{
    if (!element)
    {
        setInvalid();
        setText(QObject::tr("Select: invalid! No element given."));
        return;
    }

    elements_.append(element);

    setValid();
    setText(QObject::tr("Select"));
}

SelectDataElementCommand::SelectDataElementCommand(const QList<DataElement *> &elements, DataCommand *parent)
    : DataCommand(parent)
    , elements_(elements)
    , lastTime_(QTime::currentTime())
{
    if (elements.isEmpty())
    {
        setInvalid();
        setText(QObject::tr("Select: invalid! No element given."));
        return;
    }

    setValid();
    setText(QObject::tr("Select"));
}

SelectDataElementCommand::~SelectDataElementCommand()
{
    // nothing to be done
}

void
SelectDataElementCommand::redo()
{
    // Set selected //
    //
    foreach (DataElement *element, elements_)
    {
        element->setElementSelected(true);
    }

    lastTime_ = QTime::currentTime();

    setRedone();
}

void
SelectDataElementCommand::undo()
{
    // Set deselected //
    //
    foreach (DataElement *element, elements_)
    {
        element->setElementSelected(false);
    }

    setUndone();
}

bool
SelectDataElementCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const SelectDataElementCommand *command = static_cast<const SelectDataElementCommand *>(other);

    // Check sections //
    //
    if (lastTime_.secsTo(command->lastTime_) >= 2)
    {
        return false; // do not merge if last selection is more than 2 seconds old
    }

    // Success //
    //
    elements_.append(command->elements_);
    lastTime_ = QTime::currentTime();

    return true;
}

//################//
// Deselect       //
//################//

DeselectDataElementCommand::DeselectDataElementCommand(DataElement *element, DataCommand *parent)
    : DataCommand(parent)
    , lastTime_(QTime::currentTime())
{
    if (!element)
    {
        setInvalid();
        setText(QObject::tr("Deselect: invalid! No element given."));
        return;
    }

    elements_.append(element);

    setValid();
    setText(QObject::tr("Deselect"));
}

DeselectDataElementCommand::DeselectDataElementCommand(const QList<DataElement *> &elements, DataCommand *parent)
    : DataCommand(parent)
    , elements_(elements)
    , lastTime_(QTime::currentTime())
{
    if (elements.isEmpty())
    {
        setInvalid();
        setText(QObject::tr("Deselect: invalid! No element given."));
        return;
    }

    setValid();
    setText(QObject::tr("Deselect"));
}

DeselectDataElementCommand::~DeselectDataElementCommand()
{
    // nothing to be done
}

void
DeselectDataElementCommand::redo()
{
    // Set deselected //
    //
    foreach (DataElement *element, elements_)
    {
        element->setElementSelected(false);
    }

    lastTime_ = QTime::currentTime();

    setRedone();
}

void
DeselectDataElementCommand::undo()
{
    // Set selected //
    //
    foreach (DataElement *element, elements_)
    {
        element->setElementSelected(true);
    }

    setUndone();
}

bool
DeselectDataElementCommand::mergeWith(const QUndoCommand *other)
{
    // Check Ids //
    //
    if (other->id() != id())
    {
        return false;
    }

    const DeselectDataElementCommand *command = static_cast<const DeselectDataElementCommand *>(other);

    // Check sections //
    //
    if (lastTime_.secsTo(command->lastTime_) >= 2)
    {
        return false; // do not merge if last deselection is more than 2 seconds old
    }

    // Success //
    //
    elements_.append(command->elements_);
    lastTime_ = QTime::currentTime();

    return true;
}

//################//
// Hiding         //
//################//

HideDataElementCommand::HideDataElementCommand(DataElement *element, DataCommand *parent)
    : DataCommand(parent)
{
    if (!element)
    {
        setInvalid();
        setText(QObject::tr("Hide: invalid! No element given."));
        return;
    }

    if (element->isElementSelected())
    {
        formerlySelectedElements_.append(element);
    }
    else
    {
        formerlyDeselectedElements_.append(element);
    }

    if (element->isChildElementSelected())
    {
        foreach (DataElement *child, element->getSelectedChildElements())
        {
            formerlySelectedChildElements_.append(child);
        }
    }

    setText(QObject::tr("Hide Element"));
    setValid();
}

HideDataElementCommand::HideDataElementCommand(const QList<DataElement *> &elements, DataCommand *parent)
    : DataCommand(parent)
{
    if (elements.isEmpty())
    {
        setInvalid();
        setText(QObject::tr("Hide: invalid! No element given."));
        return;
    }

    // Split up //
    //
    foreach (DataElement *element, elements)
    {
        if (element->isElementSelected())
        {
            formerlySelectedElements_.append(element);
        }
        else
        {
            formerlyDeselectedElements_.append(element);
        }

        if (element->isChildElementSelected())
        {
            foreach (DataElement *child, element->getSelectedChildElements())
            {
                formerlySelectedChildElements_.append(child);
            }
        }
    }

    // Text //
    //
    if (elements.size() == 1)
    {
        setText(QObject::tr("Hide Element"));
    }
    else
    {
        setText(QObject::tr("Hide Elements"));
    }

    setValid();
}

HideDataElementCommand::~HideDataElementCommand()
{
    // nothing to be done
}

/*!
*
* If a DataElement gets hidden, it will also be deselected.
* If a DataElement gets unhidden, it will also be selected.
*/
void
HideDataElementCommand::redo()
{
    // Deselect Children //
    //
    foreach (DataElement *element, formerlySelectedChildElements_)
    {
        element->setElementSelected(false);
    }

    // Hide and deselect //
    //
    foreach (DataElement *element, formerlySelectedElements_)
    {
        element->setElementSelected(false);
        element->setElementHidden(true);
    }

    // Hide //
    //
    foreach (DataElement *element, formerlyDeselectedElements_)
    {
        element->setElementHidden(true);
    }

    setRedone();
}

/*!
*
* If a DataElement gets hidden, it will also be deselected.
* If a DataElement gets unhidden, it will also be selected.
*/
void
HideDataElementCommand::undo()
{
    // Show //
    //
    foreach (DataElement *element, formerlyDeselectedElements_)
    {
        element->setElementHidden(false);
    }

    // Show and select //
    //
    foreach (DataElement *element, formerlySelectedElements_)
    {
        element->setElementSelected(true);
        element->setElementHidden(false);
    }

    // Deselect Children //
    //
    foreach (DataElement *element, formerlySelectedChildElements_)
    {
        element->setElementSelected(true); // do this after the parents (so the parents will be noticed first)
    }

    setUndone();
}

//################//
// Unhiding         //
//################//

UnhideDataElementCommand::UnhideDataElementCommand(DataElement *element, DataCommand *parent)
    : DataCommand(parent)
{
    if (!element)
    {
        setInvalid();
        setText(QObject::tr("Unhide: invalid! Nothing to unhide."));
        return;
    }

    elements_.append(element);

    setText(QObject::tr("Unhide Element"));
    setValid();
}

UnhideDataElementCommand::UnhideDataElementCommand(const QList<DataElement *> &elements, DataCommand *parent)
    : DataCommand(parent)
    , elements_(elements)
{
    if (elements.isEmpty())
    {
        setInvalid();
        setText(QObject::tr("Unhide: invalid! Nothing to unhide."));
        return;
    }

    // Text //
    //
    if (elements.size() == 1)
    {
        setText(QObject::tr("Unhide Element"));
    }
    else
    {
        setText(QObject::tr("Unhide Elements"));
    }

    setValid();
}

UnhideDataElementCommand::~UnhideDataElementCommand()
{
    // nothing to be done
}

void
UnhideDataElementCommand::redo()
{
    // Unhide //
    //
    foreach (DataElement *element, elements_)
    {
        element->setElementHidden(false);
    }

    setRedone();
}

void
UnhideDataElementCommand::undo()
{
    // Hide again //
    //
    foreach (DataElement *element, elements_)
    {
        element->setElementHidden(true);
    }

    setUndone();
}
