/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Nico Eichhorn (c) 2014
**   <eichhorn@hlrs.de>
**   15.09.2014
**
**************************************************************************/

// not in use at the moment
// can be used later in development

#include "tile.hpp"

#include "src/data/projectdata.hpp"
#include "tilesystem.hpp"

Tile::Tile(const QString &name, const QString &id)
    : DataElement()
    , name_(name)
    , id_(id)
    , tileChanges_(0x0)
    , tileSystem_(NULL)
{
}

Tile::
    ~Tile()
{
}

void
Tile::setTileSystem(TileSystem *tileSystem)
{
    setParentElement(tileSystem);
    tileSystem_ = tileSystem;
    addTileChanges(Tile::CT_TileSystemChange);
}

// properties
//
QString
Tile::getIdName() const
{
    QString text = id_;
    if (!name_.isEmpty())
    {
        text.append(" (");
        text.append(name_);
        text.append(")");
    }
    return text;
}

void
Tile::setName(const QString &name)
{
    name_ = name;
    addTileChanges(Tile::CT_IdChange);
}

void
Tile::setID(const QString &id)
{
    id_ = id;
    addTileChanges(Tile::CT_IdChange);
}

// observer
//
void
Tile::notificationDone()
{
    tileChanges_ = 0x0;
    DataElement::notificationDone();
}

void
Tile::addTileChanges(int changes)
{
    if (changes)
    {
        tileChanges_ |= changes;
        notifyObservers();
    }
}

// visitor
//
void
Tile::accept(Visitor *visitor)
{
    visitor->visit(this);
}
