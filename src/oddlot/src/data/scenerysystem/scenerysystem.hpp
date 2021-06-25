/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   6/11/2010
**
**************************************************************************/

#ifndef SCENERYSYSTEM_HPP
#define SCENERYSYSTEM_HPP

#include "src/data/dataelement.hpp"

// Qt //
//
#include <QString>
#include <QMap>
#include <QStringList>

class SceneryMap;
class Heightmap;
class SceneryTesselation;

class ScenerySystem : public DataElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum ScenerySystemChange
    {
        CSC_ProjectDataChanged = 0x1,
        CSC_FileChanged = 0x2,
        CSC_MapChanged = 0x4,
        CSC_HeightmapChanged = 0x8,
        CSC_TesselationChanged = 0x10,
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ScenerySystem();
    virtual ~ScenerySystem();

    // IDs //
    //
    const QString getUniqueId(const QString &suggestion);

    // Geometry Files //
    //
    void addSceneryFile(const QString &filename);
    bool delSceneryFile(const QString &filename);
    QList<QString> getSceneryFiles() const
    {
        return sceneryFiles_;
    }

    // Background Images //
    //
    void addSceneryMap(SceneryMap *map);
    bool delSceneryMap(SceneryMap *map);
    QMap<QString, SceneryMap *> getSceneryMaps() const
    {
        return sceneryMaps_;
    }

    // Heightmaps //
    //
    void addHeightmap(Heightmap *map);
    bool delHeightmap(Heightmap *map);
    QMap<QString, Heightmap *> getHeightmaps() const
    {
        return heightmaps_;
    }

    // Tesselation //
    //
    SceneryTesselation *getSceneryTesselation() const
    {
        return sceneryTesselation_;
    }
    void setSceneryTesselation(SceneryTesselation *sceneryTesselation);

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
    int getScenerySystemChanges() const
    {
        return scenerySystemChanges_;
    }
    void addScenerySystemChanges(int changes);

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);
    virtual void acceptForChildNodes(Visitor *visitor);
    virtual void acceptForMaps(Visitor *visitor);
    virtual void acceptForHeightmaps(Visitor *visitor);
    virtual void acceptForTesselation(Visitor *visitor);

protected:
private:
    //	ScenerySystem(); /* not allowed */
    ScenerySystem(const ScenerySystem &); /* not allowed */
    ScenerySystem &operator=(const ScenerySystem &); /* not allowed */

    //################//
    // EVENTS         //
    //################//

public:
    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    // Change flags //
    //
    int scenerySystemChanges_;

    // ProjectData //
    //
    ProjectData *parentProjectData_;

    // Background Images //
    //
    QMap<QString, SceneryMap *> sceneryMaps_; // ownend

    // Heightmaps //
    //
    QMap<QString, Heightmap *> heightmaps_; // ownend

    // scenery //
    //
    QList<QString> sceneryFiles_;

    // Tesselation //
    //
    SceneryTesselation *sceneryTesselation_;

    // IDs //
    //
    QStringList ids_;
    int idCount_;
};

#endif // SCENERYSYSTEM_HPP
