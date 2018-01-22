/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Nico Eichhorn (c) 2014
**   <eichhorn@hlrs.de>
**   12.09.2014
**
**************************************************************************/

#ifndef TILE_HPP
#define TILE_HPP

#include "src/data/tilesystem/tilesystem.hpp"
#include "src/data/roadsystem/odrID.hpp"

//
class DataElement;

class Tile : public QObject, public DataElement
{
    Q_OBJECT

    //################//
    // STATIC         //
    //################//

public:
    enum TileChange
    {
        CT_ProjectDataChanged = 0x1,
        CT_TileSystemChange = 0x2,
        CT_NameChange = 0x5,
        CT_IdChange = 0x6
    };

public:
    explicit Tile(const odrID &tileID);
	explicit Tile(int tid);
    virtual ~Tile();

public:
    TileSystem *getTileSystem() const
    {
        return tileSystem_;
    }
    void setTileSystem(TileSystem *newTileSystem);

    // Element Properties //
    //
    const QString &getName() const
    {
        return name_;
    }
    const odrID &getID() const
    {
        return id_;
    }
    QString getIdName() const;

    void setName(const QString &name);
    void setID(const odrID &id);

    //Observer Pattern //

    virtual void notificationDone();

    int getTileChanges() const
    {
        return tileChanges_;
    }
    void addTileChanges(int changes);

	const QString getUniqueOSCID(const QString &suggestion,const QString &name);
	void removeOSCID(const QString &ID);
	int32_t uniqueID(odrID::IDType t);

    // Visitor Pattern //
    //

    virtual void accept(Visitor *visitor);

private:
    int tileChanges_;

    // TileSystem
    //

    TileSystem *tileSystem_;

    // Element properties //
    //
    QString name_; // name of the element
    odrID id_; // unique ID within database
	QSet<QString> oscIDs;
	QSet<int32_t> odrIDs[odrID::NUM_IDs];
};

#endif // TILE_HPP
