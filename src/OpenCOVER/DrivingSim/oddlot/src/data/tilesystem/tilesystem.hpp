/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   02.02.2010
**
**************************************************************************/

#ifndef TILESYSTEM_HPP
#define TILESYSTEM_HPP

#include "src/data/dataelement.hpp"
#include "src/data/roadsystem/odrID.hpp"

// Qt //
//
#include <QMap>
#include <QString>
#include <QStringList>

#include "tile.hpp"
class QDomElement;
class QDomDocument;

class TileSystem : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum TileSystemChange
    {
        CTS_ProjectDataChanged = 0x1,
        CTS_TileChange = 0x2,
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TileSystem();
    virtual ~TileSystem();

    // Tiles //
    //
	Tile *getTile(const odrID &id) const;
	Tile *getTile(int tid) const;
    QMap<odrID, Tile *> getTiles() const
    {
        return tiles_;
    }
    void addTile(Tile *tile);
    bool delTile(Tile *tile);
	void addTileIfNecessary(const odrID &elementID);

	void write(QDomDocument *doc_, QDomElement &root);

    Tile *getCurrentTile()
    {
        return currentTile_;
    };
    void setCurrentTile(Tile *tile);

    // ProjectData //
    //
    ProjectData *getParentProjectData() const
    {
        return parentProjectData_;
    }
    void setParentProjectData(ProjectData *projectData);

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getTileSystemChanges() const
    {
        return tileSystemChanges_;
    }
    void addTileSystemChanges(int changes);

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

    virtual void acceptForChildNodes(Visitor *visitor);

    virtual void acceptForTiles(Visitor *visitor);

private:
    //	TileSystem(); /* not allowed */
    TileSystem(const TileSystem &); /* not allowed */
    TileSystem &operator=(const TileSystem &); /* not allowed */


    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int tileSystemChanges_;

    // ProjectData //
    //
    ProjectData *parentProjectData_;

    // Tiles
    //

    QMap<odrID, Tile *> tiles_;
    Tile *currentTile_;
};

#endif // TILESYSTEM_HPP
