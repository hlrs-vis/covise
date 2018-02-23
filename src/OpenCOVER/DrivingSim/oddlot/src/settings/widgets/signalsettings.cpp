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

#include "signalsettings.hpp"
#include "ui_signalsettings.h"

#include "src/mainwindow.hpp"

// Data //
//
#include "src/data/roadsystem/sections/signalobject.hpp"
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
#include "src/data/roadsystem/sections/signalobject.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"

// Editor //
//
#include "src/graph/editors/signaleditor.hpp"
#include "src/graph/editors/projecteditor.hpp"

// Qt //
//
#include <QInputDialog>
#include <QStringList>
#include <QLabel>
#include <QSpinBox>

// Utils //
//
#include "math.h"

//################//
// CONSTRUCTOR    //
//################//

SignalSettings::SignalSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, Signal *signal)
    : SettingsElement(projectSettings, parentSettingsElement, signal)
    , ui(new Ui::SignalSettings)
    , signal_(signal)
    , init_(false)
    , valueChanged_(false)
{
    signalManager_ = getProjectSettings()->getProjectWidget()->getMainWindow()->getSignalManager();
	signalEditor_ = dynamic_cast <SignalEditor *> (getProjectSettings()->getProjectWidget()->getProjectEditor());
    ui->setupUi(this);

    // Initial Values //
    //
    updateProperties();
	activateValidityWidget(false);

    connect(ui->sSpinBox, SIGNAL(editingFinished()), this, SLOT(on_sSpinBox_editingFinished()));
    connect(ui->sSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->nameBox, SIGNAL(editingFinished()), this, SLOT(onNameBoxEditingFinished()));
    connect(ui->tSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->tSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->zOffsetSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->zOffsetSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->countryBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->countryBox, SIGNAL(textChanged(const QString&)), this, SLOT(onValueChanged()));

    connect(ui->typeBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->typeBox, SIGNAL(textChanged(const QString&)), this, SLOT(onValueChanged()));
    connect(ui->subclassLineEdit, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->subclassLineEdit, SIGNAL(textChanged(const QString&)), this, SLOT(onValueChanged()));
    connect(ui->subtypeBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->subtypeBox, SIGNAL(textChanged(const QString&)), this, SLOT(onValueChanged()));
    connect(ui->valueSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->valueSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->hOffsetSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->hOffsetSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->pitchSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->pitchSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->rollSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->rollSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
    connect(ui->dynamicCheckBox, SIGNAL(stateChanged(int)), this, SLOT(onEditingFinished(int)));
    connect(ui->orientationComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onEditingFinished(int)));
    connect(ui->poleCheckBox, SIGNAL(stateChanged(int)), this, SLOT(onEditingFinished(int)));
    connect(ui->sizeComboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(onEditingFinished(int)));

	connect(ui->widthSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
	connect(ui->widthSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
	connect(ui->heightSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
	connect(ui->heightSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
	connect(ui->unitEdit, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
	connect(ui->unitEdit, SIGNAL(textChanged(const QString&)), this, SLOT(onValueChanged()));
	connect(ui->textEdit, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
	connect(ui->textEdit, SIGNAL(textChanged(const QString&)), this, SLOT(onValueChanged()));

    connect(ui->fromLaneSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->fromLaneSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onValueChanged()));
    connect(ui->toLaneSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
    connect(ui->toLaneSpinBox, SIGNAL(valueChanged(int)), this, SLOT(onValueChanged()));

	connect(ui->validityPushButton, SIGNAL(clicked(bool)), this, SLOT(activateValidityWidget(bool)));
	connect(ui->crossingPushButton, SIGNAL(clicked(bool)), this, SLOT(activateCrossingWidget(bool)));

	connect(ui->crossingSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
	connect(ui->crossingSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));
	connect(ui->resetTimeSpinBox, SIGNAL(editingFinished()), this, SLOT(onEditingFinished()));
	connect(ui->resetTimeSpinBox, SIGNAL(valueChanged(double)), this, SLOT(onValueChanged()));

	if (signal_->getType() != "293")
	{
		enableCrossingParams(false);
	}
	
    init_ = true;
}

SignalSettings::~SignalSettings()
{
    delete ui;
}

//################//
// FUNCTIONS      //
//################//

void
SignalSettings::updateProperties()
{
    if (signal_)
    {
        ui->nameBox->setText(signal_->getName());
        ui->idLabel->setText(signal_->getId().speakingName());
        ui->sSpinBox->setValue(signal_->getSStart());
        ui->tSpinBox->setValue(signal_->getT());
        ui->zOffsetSpinBox->setValue(signal_->getZOffset());
        ui->countryBox->setText(signal_->getCountry());

        ui->typeBox->setText(signal_->getType());
        ui->subclassLineEdit->setText(signal_->getTypeSubclass());
        ui->subtypeBox->setText(signal_->getSubtype());
		SignalContainer *signalContainer = signalManager_->getSignalContainer(signal_->getCountry(), signal_->getType(),signal_->getTypeSubclass(),signal_->getSubtype());
		if (signalContainer)
		{
			QIcon icon = signalContainer->getSignalIcon();
			ui->imageTextLabel->setPixmap(icon.pixmap(icon.availableSizes().first()).scaledToHeight(30));
		}
        ui->valueSpinBox->setValue(signal_->getValue());
        ui->hOffsetSpinBox->setValue(signal_->getHeading());
        ui->pitchSpinBox->setValue(signal_->getPitch());
        ui->rollSpinBox->setValue(signal_->getRoll());
        ui->dynamicCheckBox->setChecked(signal_->getDynamic());
        ui->orientationComboBox->setCurrentIndex(signal_->getOrientation());
        ui->poleCheckBox->setChecked(signal_->getPole());
        ui->sizeComboBox->setCurrentIndex(signal_->getSize() - 1);
		ui->unitEdit->setText(signal_->getUnit());
		ui->textEdit->setText(signal_->getText());
		ui->widthSpinBox->setValue(signal_->getWidth());
		ui->heightSpinBox->setValue(signal_->getHeight());

        ui->fromLaneSpinBox->setValue(signal_->getValidFromLane());
        ui->toLaneSpinBox->setValue(signal_->getValidToLane());

		//Pedestrian Crossing has ancillary data
		//
		if ((signal_->getType() == "293") && !ui->crossingPushButton->isVisible())
		{
			enableCrossingParams(true);
		}
		else if (ui->crossingPushButton->isVisible() && (signal_->getType() != "293"))
		{
			enableCrossingParams(false);
		}

		// User data //
		if (signal_->getType() == "293")
		{
			ui->crossingSpinBox->setValue(signal_->getCrossingProbability());
			ui->resetTimeSpinBox->setValue(signal_->getResetTime());
		}
    }
}

double SignalSettings::
    signalT(double s, double t, double roadDistance)
{
    LaneSection *laneSection = signal_->getParentRoad()->getLaneSection(s);
    double dist = 0.0;
    double sSection = s - laneSection->getSStart();

    if (t >= 0)
    {
        dist = laneSection->getLaneSpanWidth(0, laneSection->getLeftmostLaneId(), sSection) + roadDistance + signal_->getParentRoad()->getLaneOffset(s);
    }
    else
    {
        dist = -laneSection->getLaneSpanWidth(0, laneSection->getRightmostLaneId(), sSection) - roadDistance + signal_->getParentRoad()->getLaneOffset(s);
    }

    return dist;
}

void
SignalSettings::enableCrossingParams(bool value)
{
	ui->crossingPushButton->setVisible(value);
	activateCrossingWidget(false);

	
//    ui->poleCheckBox->setChecked(!value); this will send updateWidget and call signalsettings
}


//################//
// SLOTS          //
//################//

void
SignalSettings::onValueChanged()
{
    valueChanged_ = true;
}

void
SignalSettings::onEditingFinished(int i)
{
    if ((ui->dynamicCheckBox->isChecked() != signal_->getDynamic()) || (ui->poleCheckBox->isChecked() != signal_->getPole()) || ((Signal::OrientationType)ui->orientationComboBox->currentIndex() != signal_->getOrientation()) || (ui->sizeComboBox->currentIndex() + 1 != signal_->getSize()))
    {
        valueChanged_ = true;
        onEditingFinished();
    }
}

void
SignalSettings::onEditingFinished()
{
    if (valueChanged_)
    {
		double t = ui->tSpinBox->value();
        int fromLane = ui->fromLaneSpinBox->value();
        int toLane = ui->toLaneSpinBox->value();

		if (toLane > fromLane)
        {
            toLane = fromLane;
        }

        if (signal_->getType() != "293")
        {
            if (((t < 0) && ((fromLane > 0) || (fromLane < signal_->getParentRoad()->getLaneSection(signal_->getSStart())->getRightmostLaneId()))) || ((t > 0) && ((fromLane < 0) || (fromLane > signal_->getParentRoad()->getLaneSection(signal_->getSStart())->getLeftmostLaneId()))))
            {
                fromLane = signal_->getParentRoad()->getValidLane(signal_->getSStart(), t);
            }

            if (((t < 0) && ((toLane > 0) || (toLane < signal_->getParentRoad()->getLaneSection(signal_->getSStart())->getRightmostLaneId()))) || ((t > 0) && ((toLane < 0) || (toLane > signal_->getParentRoad()->getLaneSection(signal_->getSStart())->getLeftmostLaneId()))))
            {
                toLane = signal_->getParentRoad()->getValidLane(signal_->getSStart(), t);
            }
        }
        else
        {
			if ((fromLane != signal_->getValidFromLane()) || (toLane != signal_->getValidToLane()))
			{
				if ((fromLane >= 0) && (toLane >= 0) && (t < 0))
				{
					LaneSection *laneSection = signal_->getParentRoad()->getLaneSection(signal_->getSStart());
					t = laneSection->getLaneSpanWidth(fromLane, toLane, signal_->getSStart());
				}
				else if ((fromLane <= 0) && (toLane <= 0) && (t > 0))
				{
					LaneSection *laneSection = signal_->getParentRoad()->getLaneSection(signal_->getSStart());
					t = -laneSection->getLaneSpanWidth(fromLane, toLane, signal_->getSStart());
				}
			}

            if (fromLane < signal_->getParentRoad()->getLaneSection(signal_->getSStart())->getRightmostLaneId())
            {
                fromLane = signal_->getParentRoad()->getLaneSection(signal_->getSStart())->getRightmostLaneId();
            }
            else if (fromLane > signal_->getParentRoad()->getLaneSection(signal_->getSStart())->getLeftmostLaneId())
            {
                fromLane = signal_->getParentRoad()->getLaneSection(signal_->getSStart())->getLeftmostLaneId();
            }

            if (toLane < signal_->getParentRoad()->getLaneSection(signal_->getSStart())->getRightmostLaneId())
            {
                toLane = signal_->getParentRoad()->getLaneSection(signal_->getSStart())->getRightmostLaneId();
            }
            else if (toLane > signal_->getParentRoad()->getLaneSection(signal_->getSStart())->getLeftmostLaneId())
            {
                toLane = signal_->getParentRoad()->getLaneSection(signal_->getSStart())->getLeftmostLaneId();
            }
        }


		double crossingProb = 0.0;
		double resetTime = 0.0;
		if (ui->crossingProbLabel->isEnabled())
		{
			crossingProb = ui->crossingSpinBox->value();
			resetTime = ui->resetTimeSpinBox->value();
		}

        SetSignalPropertiesCommand *command = new SetSignalPropertiesCommand(signal_, signal_->getId(), signal_->getName(), t, ui->dynamicCheckBox->isChecked(), (Signal::OrientationType)ui->orientationComboBox->currentIndex(), ui->zOffsetSpinBox->value(), ui->countryBox->text(), ui->typeBox->text(), ui->subclassLineEdit->text(), ui->subtypeBox->text(), ui->valueSpinBox->value(), ui->hOffsetSpinBox->value(), ui->pitchSpinBox->value(), ui->rollSpinBox->value(), ui->unitEdit->text(), ui->textEdit->text(), ui->widthSpinBox->value(), ui->heightSpinBox->value(), ui->poleCheckBox->isChecked(), ui->sizeComboBox->currentIndex() + 1, fromLane, toLane, crossingProb, resetTime, NULL);
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
SignalSettings::onNameBoxEditingFinished()
{
    QString name = ui->nameBox->text();

    if (name != signal_->getName())
    {
		odrID newId = signal_->getId();
		newId.setName(name);
        SetSignalPropertiesCommand *command = new SetSignalPropertiesCommand(signal_, newId, name, signal_->getProperties(), signal_->getValidity(), signal_->getSignalUserData(), NULL);
        getProjectSettings()->executeCommand(command);
    }
}

void
SignalSettings::on_sSpinBox_editingFinished()
{
    if (valueChanged_)
    {

        MoveRoadSectionCommand *command = new MoveRoadSectionCommand(signal_, ui->sSpinBox->value(), RSystemElementRoad::DRS_SignalSection);
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
SignalSettings::activateValidityWidget(bool activ)
{
	ui->validityGroupBox->setVisible(activ);
	double y;
	if (activ)
	{
		y = ui->validityFrame->height() + ui->validityFrame->geometry().y();
	}
	else
	{
		y = ui->validityPushButton->height() + ui->validityFrame->geometry().y();
	}
	QRect geometry = ui->crossingFrame->geometry();
	geometry.setY(y + 6);
	ui->crossingFrame->setGeometry(geometry); 
}

void
SignalSettings::activateCrossingWidget(bool activ)
{
	ui->crossingGroupBox->setVisible(activ);
}
//##################//
// Observer Pattern //
//##################//

void
SignalSettings::updateObserver()
{

    // Parent //
    //
    SettingsElement::updateObserver();
    if (isInGarbage())
    {
        return; // no need to go on
    }

    // Signal //
    //
    int changes = signal_->getSignalChanges();

    if ((changes & Signal::CEL_ParameterChange) || (changes & Signal::CEL_TypeChange))
    {
        updateProperties();
    }
}
