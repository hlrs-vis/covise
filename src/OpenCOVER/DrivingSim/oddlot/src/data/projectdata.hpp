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

#ifndef PROJECTDATA_HPP
#define PROJECTDATA_HPP

#include <QObject>
#include <QMap>
#include <QString>
#include <QStringList>

#include "dataelement.hpp"

class ProjectWidget;
class QUndoGroup;
class QUndoStack;
class ChangeManager;

class DataElement;

class GeoReference;

class OSCBase;

class ProjectData : public QObject, public DataElement
{
    Q_OBJECT

    //################//
    // STATIC         //
    //################//

public:
    enum ProjectDataChange
    {
        CPD_RoadSystemChanged = 0x1,
        CPD_VehicleSystemChanged = 0x2,
        CPD_ScenerySystemChanged = 0x4,
        CPD_PedestrianSystemChanged = 0x8,
        CPD_TileSystemChanged = 0x800,
        CPD_RevChange = 0x10,
        CPD_NameChange = 0x20,
        CPD_VersionChange = 0x40,
        CPD_DateChange = 0x80,
        CPD_SizeChange = 0x100,
        CPD_SelectedElementsChanged = 0x200,
        CPD_HiddenElementsChanged = 0x400
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ProjectData(ProjectWidget *projectWidget, QUndoStack *undoStack, ChangeManager *changeManager, int revMajor = 1, int revMinor = 2, QString name = "", float version = 1.0f, QString date = "", double north = 10000.0, double south = -10000.0, double east = 10000.0, double west = -10000.0);
    virtual ~ProjectData();

    // OpenDRIVE:header //
    //
    int getRevMajor() const
    {
        return revMajor_;
    }
    int getRevMinor() const
    {
        return revMinor_;
    }

    QString getName() const
    {
        return name_;
    }
    float getVersion() const
    {
        return version_;
    }
    QString getDate() const
    {
        return date_;
    }

    double getNorth() const
    {
        return north_;
    } // OpenDrive y
    double getSouth() const
    {
        return south_;
    }
    double getEast() const
    {
        return east_;
    } // OpenDrive x
    double getWest() const
    {
        return west_;
    }

	GeoReference *getGeoReference()
	{
		return geoReferenceParams_;
	}

    void setRevMajor(int revMajor);
    void setRevMinor(int revMinor);

    void setName(const QString &name);
    void setVersion(float version);
    void setDate(const QString &date);

    void setNorth(double north);
    void setSouth(double south);
    void setEast(double east);
    void setWest(double west);

	void setGeoReference(GeoReference *geoParams);

    // RoadSystem //
    //
    RoadSystem *getRoadSystem() const
    {
        return roadSystem_;
    }
    void setRoadSystem(RoadSystem *newRoadSystem);

    // Tile //
    //

    TileSystem *getTileSystem() const
    {
        return tileSystem_;
    }
    void setTileSystem(TileSystem *newTileSystem);

    // VehicleSystem //
    //
    VehicleSystem *getVehicleSystem() const
    {
        return vehicleSystem_;
    }
    void setVehicleSystem(VehicleSystem *vehicleSystem);

    // PedestrianSystem //
    //
    PedestrianSystem *getPedestrianSystem() const
    {
        return pedestrianSystem_;
    }
    void setPedestrianSystem(PedestrianSystem *pedestrianSystem);

    // ScenerySystem //
    //
    ScenerySystem *getScenerySystem() const
    {
        return scenerySystem_;
    }
    void setScenerySystem(ScenerySystem *scenerySystem);

	// Oddlot OpenScenario elment base //
	//
	OSCBase * getOSCBase() const
	{
		return oscBase_;
	}
	void setOSCBase(OSCBase *base);

    // ProjectData //
    //
    virtual ProjectData *getProjectData()
    {
        return this;
    }

   // ProjectWidget //
   //
   virtual ProjectWidget *getProjectWidget()
   {
      return projectWidget_;
   }

    // Undo/Redo //
    //
    virtual QUndoStack *getUndoStack()
    {
        return undoStack_;
    }

    // ChangeManager //
    //
    virtual ChangeManager *getChangeManager()
    {
        return changeManager_;
    }

    // Active Element/Selected Elements //
    //
    DataElement *getActiveElement() const;
    QList<DataElement *> getSelectedElements() const
    {
        return selectedElements_;
    }
    void addSelectedElement(DataElement *dataElement);
    void removeSelectedElement(DataElement *dataElement);

    // Hidden Elements //
    //
    QList<DataElement *> getHiddenElements() const
    {
        return hiddenElements_;
    }
    void addHiddenElement(DataElement *dataElement);
    void removeHiddenElement(DataElement *dataElement);

    // Observer Pattern //
    //
    int getProjectDataChanges() const
    {
        return projectDataChanges_;
    }
    void addProjectDataChanges(int changes);
    virtual void notificationDone();

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

private:
    ProjectData(); /* not allowed */
    ProjectData(const ProjectData &); /* not allowed */
    ProjectData &operator=(const ProjectData &); /* not allowed */

//################//
// SIGNALS        //
//################//

signals:

    //################//
    // SLOTS          //
    //################//

public slots:

    void projectActivated(bool active);

	// OpenSCENARIO settings //
	//
	void changeOSCValidation(bool);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Change flags //
    //
    int projectDataChanges_;

    // Project //
    //
    ProjectWidget *projectWidget_; // linked

    // OpenDRIVE:header //
    //
    int revMajor_; // major revision number of OpenDRIVE format
    int revMinor_; // minor revision number of OpenDRIVE format
    QString name_; // database name
    float version_; // version number of this database (format: a.bb)
    QString date_; // time/date of database creation
    double north_; // maximum interial y value in [m]
    double south_; // minimum interial y value in [m]
    double east_; // maximum interial x value in [m]
    double west_; // minimum interial x value in [m]

    // RoadSystem //
    //
    RoadSystem *roadSystem_; // owned

    // TileSystem //
    //
    TileSystem *tileSystem_;

    // VehicleSystem //
    //
    VehicleSystem *vehicleSystem_; // owned

    // PedestrianSystem //
    //
    PedestrianSystem *pedestrianSystem_; // owned

    // ScenerySystem //
    //
    ScenerySystem *scenerySystem_; // owned

	// Oddlot OpenScenario base //
    //
	OSCBase *oscBase_; // owned

    // Undo/Redo //
    //
    QUndoStack *undoStack_; // linked

    // ChangeManager //
    //
    ChangeManager *changeManager_; // linked

    // Active Element/Selected Elements //
    //
    QList<DataElement *> selectedElements_; // linked

    // Hidden Elements //
    //
    QList<DataElement *> hiddenElements_; // linked

	// Georeference String //
	//
	GeoReference *geoReferenceParams_;
};

#endif // PROJECTDATA_HPP
