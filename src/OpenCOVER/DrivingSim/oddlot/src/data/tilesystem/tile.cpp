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

Tile::Tile(const odrID &id)
    : DataElement()
    , name_(id.getName())
    , id_(id)
    , tileChanges_(0x0)
    , tileSystem_(NULL)
{
}

Tile::Tile(int tid)
  : DataElement()
, tileChanges_(0x0)
, tileSystem_(NULL)
{
	name_ = "Tile";
	odrID ID;
	ID.setID(tid);
	ID.setName(name_);
	id_ = ID;
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
	return id_.getName();
}

void
Tile::setName(const QString &name)
{
    name_ = name;
    addTileChanges(Tile::CT_IdChange);
}

void
Tile::setID(const odrID &id)
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

void Tile::removeOSCID(const QString &ID)
{
	oscIDs.remove(ID);
}

int32_t Tile::uniqueID(odrID::IDType t)
{
	int32_t id = odrIDs[t].size();
	if(odrIDs[t].contains(id))
	{
		// ID numbers are higher than number of IDs thus there must be empty spaces, search from scratch
		id = 0;
		while (odrIDs[t].contains(id))
		{
			id++;
		}
	}
	odrIDs[t].insert(id);
	return id;
}

const QString
Tile::getUniqueOSCID(const QString &suggestion, const QString &name)
{
	
	// Try suggestion //
	//
	if (!suggestion.isNull() && !suggestion.isEmpty() && !name.isEmpty())
	{
		bool number = false;
		QStringList parts = suggestion.split("_");

		if (parts.size() > 2)
		{
			int tn = parts.at(0).toInt(&number);
			if (getID().getID() == tn)
			{
				if(oscIDs.contains(suggestion))
				{
					oscIDs.insert(suggestion);
					return suggestion;
				}
			}
		}
	}

	// Create new one //
	//
	QString myName = name;
	if (name.isEmpty())
	{
		myName = "unnamed";
	}
	

	QString id;
		id = QString("%1_%2_%3").arg(uniqueID(odrID::ID_OSC)).arg(getID().getID()).arg(myName);

	oscIDs.insert(id);
	return id;
}

// visitor
//
void
Tile::accept(Visitor *visitor)
{
    visitor->visit(this);
}
