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

#include "src/data/dataelement.hpp"

// Qt //
//
#include <QMap>
#include <QString>
#include <QList>

#include "rsystemelement.hpp"

class RSystemElementRoad;
class RSystemElementController;
class RSystemElementJunction;
class RSystemElementFiddleyard;
class RSystemElementPedFiddleyard;
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
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit RoadSystem();
    virtual ~RoadSystem();

    // RSystemElements //
    //
    RSystemElementRoad *getRoad(const QString &id) const;
    QMap<QString, RSystemElementRoad *> getRoads() const
    {
        return roads_;
    }
    QList<RSystemElementRoad *> getTileRoads(const QString &tileId) const;

    void addRoad(RSystemElementRoad *road);
    bool delRoad(RSystemElementRoad *road);

    RSystemElementController *getController(const QString &id) const;
    QMap<QString, RSystemElementController *> getControllers() const
    {
        return controllers_;
    }
    QList<RSystemElementController *> getTileControllers(const QString &tileId) const;
    void addController(RSystemElementController *controller);
    bool delController(RSystemElementController *controller);

    RSystemElementJunction *getJunction(const QString &id) const;
    QMap<QString, RSystemElementJunction *> getJunctions() const
    {
        return junctions_;
    }
    QList<RSystemElementJunction *> getTileJunctions(const QString &tileId) const;
    void addJunction(RSystemElementJunction *junction);
    bool delJunction(RSystemElementJunction *junction);

    RSystemElementFiddleyard *getFiddleyard(const QString &id) const;
    QMap<QString, RSystemElementFiddleyard *> getFiddleyards() const
    {
        return fiddleyards_;
    }
    QList<RSystemElementFiddleyard *> getTileFiddleyards(const QString &tileId) const;
    void addFiddleyard(RSystemElementFiddleyard *fiddleyard);
    bool delFiddleyard(RSystemElementFiddleyard *fiddleyard);

    RSystemElementPedFiddleyard *getPedFiddleyard(const QString &id) const;
    QMap<QString, RSystemElementPedFiddleyard *> getPedFiddleyards() const
    {
        return pedFiddleyards_;
    }
    QList<RSystemElementPedFiddleyard *> getTilePedFiddleyards(const QString &tileId) const;
    void addPedFiddleyard(RSystemElementPedFiddleyard *fiddleyard);
    bool delPedFiddleyard(RSystemElementPedFiddleyard *fiddleyard);

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

    // IDs //
    //
    void checkIDs(const QMap<QString, QString> &roadIds);
    void changeUniqueId(RSystemElement *element, QString newId);
    const QString getUniqueId(const QString &suggestion, QString &name);
    void updateControllers();


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

    // ProjectData //
    //
    ProjectData *parentProjectData_;

    QMultiMap<QString, int> elementIds_;

    // RSystemElements //
    //
    // road //
    QMap<QString, RSystemElementRoad *> roads_; // owned

    // controller //
    QMap<QString, RSystemElementController *> controllers_; // owned

    // junction //
    QMap<QString, RSystemElementJunction *> junctions_; // owned

    // fiddleyard //
    QMap<QString, RSystemElementFiddleyard *> fiddleyards_; // owned

    // fiddleyard //
    QMap<QString, RSystemElementPedFiddleyard *> pedFiddleyards_; // owned
};

#endif // ROADSYSTEM_HPP
