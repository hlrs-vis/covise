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

#ifndef ROADSYSTEM_HPP
#define ROADSYSTEM_HPP

#include "../dataelement.hpp"

// Qt //
//
#include <QMap>
#include <QString>
#include <QList>

#include "rsystemelement.hpp"
#include "odrID.hpp"

class RSystemElementRoad;
class RSystemElementController;
class RSystemElementJunction;
class RSystemElementFiddleyard;
class RSystemElementPedFiddleyard;
class RSystemElementJunctionGroup;
class Tile;


class RoadSystem : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum RoadSystemChange
    {
        CRS_ProjectDataChanged = 0x1,
        CRS_RoadChange = 0x2,
        CRS_ControllerChange = 0x4,
        CRS_JunctionChange = 0x8,
        CRS_FiddleyardChange = 0x10,
        CRS_PedFiddleyardChange = 0x20,
		CRS_JunctionGroupChange = 0x40
    };

	

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadSystem();
    virtual ~RoadSystem();

    // RSystemElements //
    //
    RSystemElementRoad *getRoad(const odrID &id) const;
    QMap<uint32_t, RSystemElementRoad *> getRoads() const
    {
        return roads_;
    }
    QList<RSystemElementRoad *> getRoads(const odrID &junction) const;

    QList<RSystemElementRoad *> getTileRoads(const odrID &tileId) const;

    void addRoad(RSystemElementRoad *road);
    bool delRoad(RSystemElementRoad *road);

    RSystemElementController *getController(const odrID &id) const;
    QMap<uint32_t, RSystemElementController *> getControllers() const
    {
        return controllers_;
    }
    QList<RSystemElementController *> getTileControllers(const odrID &tileId) const;
    void addController(RSystemElementController *controller);
    bool delController(RSystemElementController *controller);

    RSystemElementJunction *getJunction(const odrID &id) const;
    QMap<uint32_t, RSystemElementJunction *> getJunctions() const
    {
        return junctions_;
    }
    QList<RSystemElementJunction *> getTileJunctions(const odrID &tileId) const;
    void addJunction(RSystemElementJunction *junction);
    bool delJunction(RSystemElementJunction *junction);

    RSystemElementFiddleyard *getFiddleyard(const odrID &id) const;
    QMap<uint32_t, RSystemElementFiddleyard *> getFiddleyards() const
    {
        return fiddleyards_;
    }
    QList<RSystemElementFiddleyard *> getTileFiddleyards(const odrID &tileId) const;
    void addFiddleyard(RSystemElementFiddleyard *fiddleyard);
    bool delFiddleyard(RSystemElementFiddleyard *fiddleyard);

    RSystemElementPedFiddleyard *getPedFiddleyard(const odrID &id) const;
    QMap<uint32_t, RSystemElementPedFiddleyard *> getPedFiddleyards() const
    {
        return pedFiddleyards_;
    }
    QList<RSystemElementPedFiddleyard *> getTilePedFiddleyards(const odrID &tileId) const;
    void addPedFiddleyard(RSystemElementPedFiddleyard *fiddleyard);
    bool delPedFiddleyard(RSystemElementPedFiddleyard *fiddleyard);

	QMap<uint32_t, RSystemElementJunctionGroup *> getJunctionGroups() const
	{
		return junctionGroups_;
	}
	QList<RSystemElementJunctionGroup *> getTileJunctionGroups(const odrID &tileId) const;
	void addJunctionGroup(RSystemElementJunctionGroup *junctionGroup);
	bool delJunctionGroup(RSystemElementJunctionGroup *junctionGroup);

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
    int getRoadSystemChanges() const
    {
        return roadSystemChanges_;
    }
    void addRoadSystemChanges(int changes);

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

    virtual void acceptForChildNodes(Visitor *visitor);

    virtual void acceptForRoads(Visitor *visitor);
    virtual void acceptForControllers(Visitor *visitor);
    virtual void acceptForJunctions(Visitor *visitor);
    virtual void acceptForFiddleyards(Visitor *visitor);
    virtual void acceptForPedFiddleyards(Visitor *visitor);
	virtual void acceptForJunctionGroups(Visitor *visitor);

    // IDs //
    //

	odrID RoadSystem::getID(const QString &name, odrID::IDType t);
	odrID RoadSystem::getID(int32_t tileID, odrID::IDType t);
	odrID RoadSystem::getID(odrID::IDType t);// creates a unique ID with name unknown in current Tile
	odrID RoadSystem::getID(int32_t ID, int32_t tileID, QString &name, odrID::IDType t);

	void StringToNumericalIDs(const QMap<odrID, odrID> &idMap);
    void changeUniqueId(RSystemElement *element, const odrID &newId);
	int32_t uniqueID();
    void updateControllers();

	// Find closest road to global point //
	//
	RSystemElementRoad * findClosestRoad(const QPointF &to, double &s, double &t, QVector2D &vec);

    // OpenDRIVE Data //
    //
    void verify();


private:
    //	RoadSystem(); /* not allowed */
    RoadSystem(const RoadSystem &); /* not allowed */
    RoadSystem &operator=(const RoadSystem &); /* not allowed */


    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int roadSystemChanges_;
	QSet<int> allIDs;

    // ProjectData //
    //
    ProjectData *parentProjectData_;

    QMultiMap<QString, int> elementIds_;

    // RSystemElements //
    //
    // road //
    QMap<uint32_t, RSystemElementRoad *> roads_; // owned

    // controller //
    QMap<uint32_t, RSystemElementController *> controllers_; // owned

    // junction //
    QMap<uint32_t, RSystemElementJunction *> junctions_; // owned

    // fiddleyard //
    QMap<uint32_t, RSystemElementFiddleyard *> fiddleyards_; // owned

    // fiddleyard //
    QMap<uint32_t, RSystemElementPedFiddleyard *> pedFiddleyards_; // owned

	// junctionGroup //
	QMap<uint32_t, RSystemElementJunctionGroup *> junctionGroups_;
};

#endif // ROADSYSTEM_HPP
