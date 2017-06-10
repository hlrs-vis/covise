/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Uwe Woessner (c) 2013
**   <woessner@hlrs.de.de>
**   03/2013
**
**************************************************************************/
#ifndef OSMIMPORT_HPP
#define OSMIMPORT_HPP

#include <QNetworkAccessManager>
#include <QUrl>
#include <QNetworkRequest>
#include <QNetworkReply>

#include <QDialog>
#include <QDomElement>

namespace Ui
{
class OsmImport;
}

class ProjectWidget;
class osmNode
{
public:
    osmNode();
    osmNode(QDomElement element);
    osmNode(const osmNode &n);
    void getCoordinates(double &x, double &y, double &z) const;
    double latitude;
    double longitude;
    uint64_t id;

private:
    //...
};

class osmWay
{
public:
    enum wayType
    {
        building = 0,
        highway,
        residential,
        footway,
        track,
        steps,
        primary,
        secondary,
        tertiary,
        motorway,
        living_street,
        cycleway,
        turning_circle,
        pedestrian,
        unclassified,
        path,
        service,
        unknown
    };
    static QString getTypeName(wayType t);
    osmWay();
    osmWay(const osmWay &w);
    osmWay(QDomElement element, QVector<osmNode *> &nodes);
    wayType type;
    QString name;
    int numLanes;
    int maxSpeed;
    bool bridge;
    std::vector<double> XVector;
    std::vector<double> YVector;
    std::vector<double> ZVector;

private:
};

class OsmImport : public QDialog
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit OsmImport();
    virtual ~OsmImport();
    bool importOSMFile(const QString &fileName);
    void setProject(ProjectWidget *pw)
    {
        project = pw;
    };

private:
    QVector<osmNode*> nodes;
    QVector<osmWay*> ways;
    QNetworkAccessManager *nam;
    //################//
    // SLOTS          //
    //################//

    ProjectWidget *project;

private slots:
    void okPressed();
    void on_preview_released();
    void finishedSlot(QNetworkReply *reply);
    bool parseDoc(QDomDocument &doc);
    

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::OsmImport *ui;

    bool init_;
};

#endif // OSMIMPORT_HPP
