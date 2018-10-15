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

#include "projectionsettings.hpp"
/*#include "projectwidget.hpp"
#include "src/mainwindow.hpp"*/
#include "ui_projectionsettings.h"
#include "src/data/projectdata.hpp"
#include "src/data/georeference.hpp"

// Data //

//ProjectionSettings *ProjectionSettings::inst = NULL;

void ProjectionSettings::okPressed()
{
    updateSettings();
}

ProjectionSettings::Preset ProjectionSettings::resolvePreset(const QString &input)
{
    if(input == "WGS84 to Potsdam") return WGS84_to_Potsdam;
    if(input == "WGS84 to WGS84 Ellipsoid") return WGS84_to_WGS84_Ellipsoid;

    return None;
}

void ProjectionSettings::PresetIndexChanged(const QString &change)
{
    Preset pre;
    if( (pre = resolvePreset(change)) != None)
    {
        ui->Proj4Edit->setText(presets.value(pre));
    }
    updateUi();
}

//################//
// CONSTRUCTOR    //
//################//

ProjectionSettings::ProjectionSettings()
    : ui(new Ui::ProjectionSettings)
    , projectData_(nullptr)
{
    //inst = this;

    ui->setupUi(this);

    //connect(this, SIGNAL(accepted()), this, SLOT(okPressed()));
    connect(ui->presetBox, QOverload<const QString&>::of(&QComboBox::currentIndexChanged),[=](const QString &change){PresetIndexChanged(change);});

    //Initialize presetBox
    /*QStringList set;
    Preset pre = None;
    for(int i = None; i < WGS84_to_WGS84_Ellipsoid; i++)
    {
        pre += i;
        set += QVariant::fromValue(i).value<QString>();
    }*/
    //QString s = QVariant::fromValue(ProjectionSettings::None).value<QString>();

    ui->presetBox->addItems(QStringList{"None","WGS84 to Potsdam","WGS84 to WGS84 Ellipsoid"});

    presets.insert(WGS84_to_Potsdam,"+proj=latlong +datum=WGS84 +to +proj=tmerc +lat_0=0 +lon_0=9 +k=1.000000 +x_0=3500000 +y_0=0 +ellps=bessel +datum=potsdam");
    presets.insert(WGS84_to_WGS84_Ellipsoid,"+proj=latlong +datum=WGS84 +to +proj=utm +zone=50 +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0");
    //ui->presetBox->addItems(presets);

    ui->XOffsetSpin->setDecimals(10);
    ui->XOffsetSpin->setMaximum(10000000);
    ui->XOffsetSpin->setMinimum(-10000000);
    //ui->XOffsetSpin->setValue(-5439122.807299255);
    ui->XOffsetSpin->setValue(-3506000);
    //ui->XOffsetSpin->setValue(-926151);
    ui->YOffsetSpin->setDecimals(10);
    ui->YOffsetSpin->setMaximum(10000000);
    ui->YOffsetSpin->setMinimum(-10000000);
    //ui->YOffsetSpin->setValue(-984970.1841083583);
    //ui->YOffsetSpin->setValue(-3463995);
    ui->YOffsetSpin->setValue(-5400147);
    ui->ZOffsetSpin->setDecimals(10);
    ui->ZOffsetSpin->setMaximum(10000000);
    ui->ZOffsetSpin->setMinimum(-10000000);
    //ui->ZOffsetSpin->setValue(-399.4944465);
    ui->ZOffsetSpin->setValue(0.0);
    //ui->FromDatumEdit->setText(QString("WGS84"));
    /*
#ifdef WIN32
	char *pValue;
	size_t len;
	errno_t err = _dupenv_s(&pValue, &len, "ODDLOTDIR");
	if (err || pValue == NULL || strlen(pValue) == 0)
		err = _dupenv_s(&pValue, &len, "COVISEDIR");
	if (err)
		pValue="";
	QString covisedir = pValue;
#else
	QString covisedir = getenv("ODDLOTDIR");
	if (covisedir == "")
		covisedir = getenv("COVISEDIR");
#endif
    QString dir = covisedir + "/share/covise/";*/
    //ui->ToProjectionEdit->setText(QString("tmerc +lat_0=0 +lon_0=9 +k=1.000000 +x_0=3500000 +y_0=0 +ellps=bessel +datum=potsdam"));
    //ui->ToProjectionEdit->setText(QString("tmerc +lat_0=0 +lon_0=9 +k=1.000000 +x_0=3500000 +y_0=0 +ellps=bessel +datum=potsdam +nadgrids="+dir+"BETA2007.gsb"));
    XOffset = ui->XOffsetSpin->value();
    YOffset = ui->YOffsetSpin->value();
    ZOffset = ui->ZOffsetSpin->value();
    //ui->ToProjectionEdit->setText(QString("utm +zone=50 +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0"));

    /*QString projFromString = "+proj=latlong +datum=" + ui->FromDatumEdit->text();
    //std::string projToString = "+proj=merc +x_0=-1008832.89 +y_0=-6179385.47";
    QString projToString = "+proj=" + ui->ToProjectionEdit->text();
    XOffset = ui->XOffsetSpin->value();
    YOffset = ui->YOffsetSpin->value();
    ZOffset = ui->ZOffsetSpin->value();
    
    //TODO: substitute with projectData implementation!
    //Converts string representation of coordinate system into projPJ Object (pj_init_plus(...))
    if (!(new_pj_from = pj_init_plus(projFromString.toUtf8().constData())))
    {
        QMessageBox msg;
        msg.setText("RoadSystem::parseIntermapRoad(): couldn't initialize projection source: " + projFromString);
        msg.exec();
        //std::cerr << "RoadSystem::parseIntermapRoad(): couldn't initalize projection source: " << projFromString << std::endl;
        //return false;
    }
    else
    connect(this, SIGNAL(accepted()), this, SLOT(okPressed()));
    {
        projectData_->setProj4ReferenceFrom(&new_pj_from);
        //pj_from = new_pj_from;
    }

    if (!(new_pj_to = pj_init_plus(projToString.toUtf8().constData())))
    {
        QMessageBox msg;
        msg.setText("RoadSystem::parseIntermapRoad(): couldn't initialize projection target: " + projToString);
        msg.exec();
        //std::cerr << "RoadSystem::parseIntermapRoad(): couldn't initalize projection target: " << projToString << std::endl;
        //return false;
    }
    else
    {
        projectData_->setProj4ReferenceTo(&new_pj_to);
        //pj_to = new_pj_to;
    }*/
    //inst = this;
}

ProjectionSettings::~ProjectionSettings()
{
    delete ui;
    delete projectData_;
}

void ProjectionSettings::transform(double &x, double &y, double &z)
{
    pj_transform(projectData_->getProj4ReferenceFrom(),projectData_->getProj4ReferenceTo(),1,1,&x,&y,&z);
    x += XOffset;
    y += YOffset;
    z += ZOffset;
}

void ProjectionSettings::update()
{
    if(projectData_->getGeoReference() == nullptr) {
        ui->presetBox->setCurrentIndex(1);
    }
    else
    {
        ui->Proj4Edit->setText(projectData_->getGeoReference()->getParams());
        updateUi();
    }
}

void ProjectionSettings::setProjectData(ProjectData *pd)
{
    projectData_ = pd;
    //connect(projectData_->getProjectWidget()->getMainWindow()->getFileSettings(), SIGNAL(accepted()), this, SLOT(okPressed()));
    update();
}

QString ProjectionSettings::prepareString(const QString &src)
{
    QString pre;
    if(src.indexOf("+init=epsg:") == -1)
    {
        pre = "+proj" + (src.section("+proj", 1,1));
    }
    else
    {
        pre = "+init=epsg:" + (src.section("+init=epsg:", 1,1)).section(' ',0,0);
    }
    return pre;
}

void ProjectionSettings::updateSettings()
{
    QMessageBox msg;
    QString proj = ui->Proj4Edit->text();
    QString projTo = prepareString(proj.section("+to ",1));
    QString projFrom = prepareString(proj);
    (projFrom).remove(" +to ");
    projPJ new_pj_from,new_pj_to;
    XOffset = ui->XOffsetSpin->value();
    YOffset = ui->YOffsetSpin->value();
    ZOffset = ui->ZOffsetSpin->value();

    new_pj_from = pj_init_plus((projFrom).toUtf8().constData());
    if(!(new_pj_from))
    {
        msg.setText("RoadSystem::parseIntermapRoad(): couldn't initialize projection source: " + projFrom);
        msg.exec();
    }
    else
    {
        new_pj_to = pj_init_plus((projTo).toUtf8().constData());
        projectData_->setProj4ReferenceFrom(new_pj_from);
        if(!(new_pj_to))
        {
            msg.setText("RoadSystem::parseIntermapRoad(): couldn't initialize projection target: " + projTo);
            msg.exec();
        }
        else
        {
            projectData_->setProj4ReferenceTo(new_pj_to);
            projectData_->setGeoReference(new GeoReference(proj));
            updateUi();
        }
    }
}

void ProjectionSettings::checkProjForEPSG(const QString &proj)
{
    QString update;
    if(proj.indexOf("+init=epsg:") != -1) {
        update = proj.section("+init=epsg:",1,1).section(" ",0,0);
    }
    else
    {
        update = "None";
    }
    ui->EPSGEdit->setText(update);
}

void ProjectionSettings::checkProjForPreset(const QString &proj)
{
    ui->presetBox->setCurrentIndex(presets.key(proj));
}

void ProjectionSettings::updateUi()
{
    checkProjForEPSG(ui->Proj4Edit->text());
    checkProjForPreset(ui->Proj4Edit->text());
}
