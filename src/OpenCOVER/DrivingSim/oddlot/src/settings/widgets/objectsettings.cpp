/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   11/2/2010
**
**************************************************************************/

#include "objectsettings.hpp"
#include "ui_objectsettings.h"

#include "src/mainwindow.hpp"

// Data //
//
#include "src/data/roadsystem/sections/objectobject.hpp"
#include "src/data/roadsystem/rsystemelementjunction.hpp"
#include "src/data/roadsystem/roadsystem.hpp"
#include "src/data/roadsystem/roadlink.hpp"
#include "src/data/roadsystem/rsystemelementroad.hpp"
#include "src/data/commands/roadcommands.hpp"
#include "src/data/commands/roadsystemcommands.hpp"
#include "src/data/commands/signalcommands.hpp"
#include "src/data/commands/roadsectioncommands.hpp"
#include "src/data/signalmanager.hpp"
#include "src/data/roadsystem/sections/lanesection.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"

// Qt //
//
#include <QInputDialog>
#include <QStringList>
#include <QVariant>

// Utils //
//
#include "math.h"

//################//
// CONSTRUCTOR    //
//################//

ObjectSettings::ObjectSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, Object *object)
    : SettingsElement(projectSettings, parentSettingsElement, object)
    , ui(new Ui::ObjectSettings)
    , object_(object)
    , init_(false)
    , valueChanged_(false)
{
    objectManager_ = getProjectSettings()->getProjectWidget()->getMainWindow()->getSignalManager();
    ui->setupUi(this);

	// TreeWidget for repeat parameters //
	//
/*	QList<QTreeWidgetItem *> itemList = ui->repeatTreeWidget->findItems("repeat", Qt::MatchExactly);
	QTreeWidgetItem *repeatItem = itemList.first();
	QTreeWidgetItem *layoutItem = new QTreeWidgetItem(repeatItem);
	ui->repeatTreeWidget->setItemWidget(layoutItem, 0, ui->repeatWidget);*/

    // Initial Values //
    //
    updateProperties();
	activateRepeatWidget(false);

    connect(ui->sSpinBox, SIGNAL(editingFinished()), this, SLOT(on_sSpinBox_editingFinished()));
    connect(ui->sSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->nameBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->nameBox, SIGNAL(textChanged(const QString&)), this, SLOT(onValueChanged()));
    connect(ui->tSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->tSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->zOffsetSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->zOffsetSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->typeBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->typeBox, SIGNAL(textChanged(const QString&)), this, SLOT(onValueChanged()));

    connect(ui->validLengthSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->validLengthSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->lengthSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->lengthSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->widthSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->widthSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->heightSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->heightSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->radiusSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->radiusSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->hdgSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->hdgSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->pitchSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->pitchSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->rollSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->rollSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->poleCheckBox, SIGNAL(stateChanged(int)), this, SLOT(onEditingFinished(int)));
    connect(ui->orientationComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onEditingFinished(int)));

    connect(ui->repeatSSpinBox, SIGNAL(editingFinished()), this, SLOT(on_repeatSSpinBox_editingFinished()));
    connect(ui->repeatSSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->repeatLengthSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->repeatLengthSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->repeatDistanceSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->repeatDistanceSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
	connect(ui->repeatTStartSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
	connect(ui->repeatTStartSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
	connect(ui->repeatTEndSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
	connect(ui->repeatTEndSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
	connect(ui->repeatWidthStartSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
	connect(ui->repeatWidthStartSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
	connect(ui->repeatWidthEndSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
	connect(ui->repeatWidthEndSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
	connect(ui->repeatHeightStartSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
	connect(ui->repeatHeightStartSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
	connect(ui->repeatHeightEndSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
	connect(ui->repeatHeightEndSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
	connect(ui->repeatZOffsetStartSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
	connect(ui->repeatZOffsetStartSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
	connect(ui->repeatZOffsetEndSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
	connect(ui->repeatZOffsetEndSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));

    connect(ui->textureLineEdit, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->textureLineEdit, SIGNAL(textChanged(const QString&)), this, SLOT(onValueChanged()));

	connect(ui->repeatButton, SIGNAL(clicked(bool)), this, SLOT(activateRepeatWidget(bool)));

    init_ = true;
}

ObjectSettings::~ObjectSettings()
{
    delete ui;
}

//################//
// FUNCTIONS      //
//################//

void
ObjectSettings::updateProperties()
{
    if (object_)
    {
        ui->nameBox->setText(object_->getModelFileName());
        ui->idLabel->setText(object_->getId());
        ui->sSpinBox->setValue(object_->getSStart());
        ui->tSpinBox->setValue(object_->getT());
        ui->zOffsetSpinBox->setValue(object_->getzOffset());
        ui->typeBox->setText(object_->getType());
        //	ui->typeComboBox->setCurrentIndex(object_->getType()-100001);

	    ui->heightSpinBox->setValue(object_->getHeight());
        ui->validLengthSpinBox->setValue(object_->getValidLength());
        ui->lengthSpinBox->setValue(object_->getLength());
        ui->widthSpinBox->setValue(object_->getWidth());
        ui->radiusSpinBox->setValue(object_->getRadius());
        ui->pitchSpinBox->setValue(object_->getPitch());
        ui->rollSpinBox->setValue(object_->getRoll());
        ui->hdgSpinBox->setValue(object_->getHeading());
        ui->orientationComboBox->setCurrentIndex(object_->getOrientation());
        ui->poleCheckBox->setChecked(object_->getPole());

        ui->repeatSSpinBox->setValue(object_->getRepeatS());
        ui->repeatLengthSpinBox->setValue(object_->getRepeatLength());
        ui->repeatDistanceSpinBox->setValue(object_->getRepeatDistance());
		ui->repeatTStartSpinBox->setValue(object_->getRepeatTStart());
		ui->repeatTEndSpinBox->setValue(object_->getRepeatTEnd());
		ui->repeatWidthStartSpinBox->setValue(object_->getRepeatWidthStart());
		ui->repeatWidthEndSpinBox->setValue(object_->getRepeatWidthEnd());
		ui->repeatHeightStartSpinBox->setValue(object_->getRepeatHeightStart());
		ui->repeatHeightEndSpinBox->setValue(object_->getRepeatHeightEnd());
		ui->repeatZOffsetStartSpinBox->setValue(object_->getRepeatZOffsetStart());
		ui->repeatZOffsetEndSpinBox->setValue(object_->getRepeatZOffsetEnd());

        ui->textureLineEdit->setText(object_->getTextureFileName());
    }
}

double ObjectSettings::
    objectT(double s, double t, double roadDistance)
{
    LaneSection *laneSection = object_->getParentRoad()->getLaneSection(s);
    double dist = 0.0;
    double sSection = s - laneSection->getSStart();

    if (t >= 0)
    {
        dist = laneSection->getLaneSpanWidth(0, laneSection->getLeftmostLaneId(), sSection) + roadDistance;
    }
    else
    {
        dist = -laneSection->getLaneSpanWidth(0, laneSection->getRightmostLaneId(), sSection) - roadDistance;
    }

    return dist;
}

void
ObjectSettings::updateProperties(QString country, ObjectContainer *objectProperties)
{
    double t = objectT(ui->sSpinBox->value(), ui->tSpinBox->value(), objectProperties->getObjectDistance());

    ui->tSpinBox->setValue(t);
    ui->countryBox->setText(country);

    ui->typeBox->setText(objectProperties->getObjectType());
    ui->lengthSpinBox->setValue(objectProperties->getObjectLength());
    ui->validLengthSpinBox->setValue(objectProperties->getObjectLength());
    ui->widthSpinBox->setValue(objectProperties->getObjectWidth());
    ui->radiusSpinBox->setValue(objectProperties->getObjectRadius());
    ui->repeatDistanceSpinBox->setValue(objectProperties->getObjectRepeatDistance());
    ui->heightSpinBox->setValue(objectProperties->getObjectHeight());
    ui->hdgSpinBox->setValue(objectProperties->getObjectHeading());
}

//################//
// SLOTS          //
//################//

void
ObjectSettings::onEditingFinished(int i)
{
    if ((ui->poleCheckBox->isChecked() != object_->getPole()) || (Object::ObjectOrientation)ui->orientationComboBox->currentIndex() != object_->getOrientation())
    {
        valueChanged_ = true;
        onEditingFinished();
    }
}

void
ObjectSettings::onEditingFinished()
{
    if (valueChanged_)
    {
        QString filename = ui->nameBox->text();
        QString newId = object_->getId();
        RSystemElementRoad * road = object_->getParentRoad();
        if (filename != object_->getName())
        {
            QStringList parts = object_->getId().split("_");

            if (parts.size() > 2)
            {
                newId = QString("%1_%2_%3").arg(parts.at(0)).arg(parts.at(1)).arg(filename); 
            }
            else
            {
                newId = road->getRoadSystem()->getUniqueId(object_->getId(), filename);
            }
        }

        double repeatLength = ui->repeatLengthSpinBox->value();
        if (repeatLength > road->getLength() - ui->repeatSSpinBox->value())
        {
            repeatLength = road->getLength() - ui->repeatSSpinBox->value();
        }
		
		Object::ObjectProperties objectProps{ ui->tSpinBox->value(), (Object::ObjectOrientation)ui->orientationComboBox->currentIndex(), ui->zOffsetSpinBox->value(), ui->typeBox->text(),
			ui->validLengthSpinBox->value(), ui->lengthSpinBox->value(), ui->widthSpinBox->value(), ui->radiusSpinBox->value(), ui->heightSpinBox->value(), ui->hdgSpinBox->value(), 
			ui->pitchSpinBox->value(), ui->rollSpinBox->value() };
		Object::ObjectRepeatRecord repeatProps{ ui->repeatSSpinBox->value(), repeatLength, ui->repeatDistanceSpinBox->value(), ui->repeatTStartSpinBox->value(), ui->repeatTEndSpinBox->value(),
			ui->repeatWidthStartSpinBox->value(), ui->repeatWidthEndSpinBox->value(), ui->repeatHeightStartSpinBox->value(), ui->repeatHeightEndSpinBox->value(),
			ui->repeatZOffsetStartSpinBox->value(), ui->repeatZOffsetEndSpinBox->value() };

		SetObjectPropertiesCommand *command = new SetObjectPropertiesCommand(object_, newId, filename, objectProps, repeatProps, ui->textureLineEdit->text());
        getProjectSettings()->executeCommand(command);

        valueChanged_ = false;
        QWidget * focusWidget = QApplication::focusWidget();
        if (focusWidget)
        {
            focusWidget->clearFocus();
        }
    }
}

void
ObjectSettings::onValueChanged()
{
	valueChanged_ = true;
}

void
ObjectSettings::on_repeatSSpinBox_editingFinished()
{
	if (valueChanged_)
    {
		updateS(ui->repeatSSpinBox->value());
	}
}

void
ObjectSettings::on_sSpinBox_editingFinished()
{
    if (valueChanged_)
    {
		updateS(ui->sSpinBox->value());
	}
}

void
	ObjectSettings::updateS(double s)
{

	MoveRoadSectionCommand *moveSectionCommand = new MoveRoadSectionCommand(object_, s, RSystemElementRoad::DRS_ObjectSection);
	if (moveSectionCommand->isValid())
	{
		getProjectData()->getUndoStack()->beginMacro(QObject::tr("Change Start Values"));
		getProjectSettings()->executeCommand(moveSectionCommand);

		Object::ObjectRepeatRecord repeatProps = object_->getRepeatProperties();
		repeatProps.s = object_->getSStart();
		SetObjectPropertiesCommand *setPropertiesCommand = new SetObjectPropertiesCommand(object_, object_->getId(), object_->getName(), object_->getProperties(), repeatProps, object_->getTextureFileName());
		getProjectSettings()->executeCommand(setPropertiesCommand);

		getProjectData()->getUndoStack()->endMacro();
	}

	valueChanged_ = false;

	QWidget * focusWidget = QApplication::focusWidget();
	if (focusWidget)
	{
		focusWidget->clearFocus();
	}
}

void
	ObjectSettings::activateRepeatWidget(bool activ)
{
	ui->repeatGroupBox->setVisible(activ);
}

//##################//
// Observer Pattern //
//##################//

void
ObjectSettings::updateObserver()
{

    // Parent //
    //
    SettingsElement::updateObserver();
    if (isInGarbage())
    {
        return; // no need to go on
    }

    // Object //
    //
    int changes = object_->getObjectChanges();

	if ((changes & Object::CEL_ParameterChange) || (changes & Object::CEL_TypeChange))
    {
        updateProperties();
    }
}
