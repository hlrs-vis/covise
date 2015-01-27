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

#ifndef DATAELEMENTCOMMANDS_HPP
#define DATAELEMENTCOMMANDS_HPP

// 1400

#include "datacommand.hpp"

class DataElement;

#include <QList>
#include <QTime>

//################//
// Select         //
//################//

class SelectDataElementCommand : public DataCommand
{

public:
    explicit SelectDataElementCommand(DataElement *element, DataCommand *parent = NULL);
    explicit SelectDataElementCommand(const QList<DataElement *> &elements, DataCommand *parent = NULL);
    virtual ~SelectDataElementCommand();

    virtual int id() const
    {
        return 0x1401;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    SelectDataElementCommand(); /* not allowed */
    SelectDataElementCommand(const SelectDataElementCommand &); /* not allowed */
    SelectDataElementCommand &operator=(const SelectDataElementCommand &); /* not allowed */

private:
    QList<DataElement *> elements_;

    QTime lastTime_;
};

//################//
// Deselect       //
//################//

class DeselectDataElementCommand : public DataCommand
{

public:
    explicit DeselectDataElementCommand(DataElement *element, DataCommand *parent = NULL);
    explicit DeselectDataElementCommand(const QList<DataElement *> &elements, DataCommand *parent = NULL);
    virtual ~DeselectDataElementCommand();

    virtual int id() const
    {
        return 0x1402;
    }

    virtual void undo();
    virtual void redo();

    virtual bool mergeWith(const QUndoCommand *other);

private:
    DeselectDataElementCommand(); /* not allowed */
    DeselectDataElementCommand(const DeselectDataElementCommand &); /* not allowed */
    DeselectDataElementCommand &operator=(const DeselectDataElementCommand &); /* not allowed */

private:
    QList<DataElement *> elements_;

    QTime lastTime_;
};

//################//
// Hiding         //
//################//

class HideDataElementCommand : public DataCommand
{

public:
    explicit HideDataElementCommand(DataElement *element, DataCommand *parent = NULL);
    explicit HideDataElementCommand(const QList<DataElement *> &elements, DataCommand *parent = NULL);
    virtual ~HideDataElementCommand();

    virtual int id() const
    {
        return 0x1404;
    }

    virtual void undo();
    virtual void redo();

private:
    HideDataElementCommand(); /* not allowed */
    HideDataElementCommand(const HideDataElementCommand &); /* not allowed */
    HideDataElementCommand &operator=(const HideDataElementCommand &); /* not allowed */

private:
    QList<DataElement *> formerlySelectedElements_;
    QList<DataElement *> formerlyDeselectedElements_;

    QList<DataElement *> formerlySelectedChildElements_;
};

//################//
// Unhiding         //
//################//

class UnhideDataElementCommand : public DataCommand
{

public:
    explicit UnhideDataElementCommand(DataElement *element, DataCommand *parent = NULL);
    explicit UnhideDataElementCommand(const QList<DataElement *> &elements, DataCommand *parent = NULL);
    virtual ~UnhideDataElementCommand();

    virtual int id() const
    {
        return 0x1404;
    }

    virtual void undo();
    virtual void redo();

private:
    UnhideDataElementCommand(); /* not allowed */
    UnhideDataElementCommand(const UnhideDataElementCommand &); /* not allowed */
    UnhideDataElementCommand &operator=(const UnhideDataElementCommand &); /* not allowed */

private:
    QList<DataElement *> elements_;
};

#endif // DATAELEMENTCOMMANDS_HPP
