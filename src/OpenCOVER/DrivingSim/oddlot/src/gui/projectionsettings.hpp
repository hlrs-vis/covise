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
#ifndef PROJECTIONSETTINGS_HPP
#define PROJECTIONSETTINGS_HPP

#include <QDialog>
#include <QMessageBox>
#include <QComboBox>
#include <proj_api.h>

class ProjectData;

namespace Ui
{
class ProjectionSettings;
}

class PresetInfo
{
public:
    PresetInfo(std::string f, std::string t, std::string n) :to(t), from(f), name(n) {};
    std::string to;
    std::string from;
    std::string name;
};
class ProjectionSettings : public QDialog
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:

    explicit ProjectionSettings();
    virtual ~ProjectionSettings();

    void transform(double &x, double &y, double &z);

    void setProjectData(ProjectData *pd);
    //ProjectData *projectData_;
    double XOffset;
    double YOffset;
    double ZOffset;
    /*
    static ProjectionSettings *instance()
    {
        return inst;
    };*/

private:
    std::vector<PresetInfo> presets;
    //static ProjectionSettings *inst;
    /*struct Presets
    {
        QList<Preset> presetsEnumList = {None,WGS84_to_Potsdam,WGS84_to_WGS84_Ellipsoid};
        QMap<Preset,QString> presetsToProj;
    };
    Presets presets;*/
    void update();
    void updateSettings();
    void updateUi();
    void checkProjForEPSG(const QString &proj);
    void checkProjForPreset(const QString &proj);
    QString prepareString(const QString &src);
    //################//
    // SLOTS          //
    //################//

private slots:
    void okPressed();
    void PresetIndexChanged(const QString &change);

    //################//
    // PROPERTIES     //
    //################//

private:
    Ui::ProjectionSettings *ui;
    ProjectData *projectData_;
};

#endif // OSMIMPORT_HPP
