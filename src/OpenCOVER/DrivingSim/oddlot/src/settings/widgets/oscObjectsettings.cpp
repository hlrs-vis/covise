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

#include "oscObjectSettings.hpp"

#include "src/mainwindow.hpp"

// OpenScenario //
//
#include "oscObjectBase.h"
#include "OpenScenarioBase.h"
#include "oscVariables.h"

// Settings //
//
#include "ui_oscObjectSettings.h"

// Commands //
//
#include "src/data/commands/osccommands.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"

// Data //
#include "src/data/oscsystem/oscelement.hpp"

// Qt //
//
#include <QInputDialog>
#include <QStringList>
#include <QSpinBox>
#include <QLineEdit>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QComboBox>
#include <QSignalMapper>
#include <QLabel>

// Utils //
//
#include "math.h"

using namespace OpenScenario;

//################//
// CONSTRUCTOR    //
//################//

oscObjectSettings::oscObjectSettings(ProjectSettings *projectSettings, SettingsElement *parentSettingsElement, OSCElement *element)
    : SettingsElement(projectSettings, parentSettingsElement, element)
    , ui(new Ui::oscObjectSettings)
    , init_(false)
    , valueChanged_(true)
	, element_(element)
{
	object_ = element_->getObject();

    ui->setupUi(this);

	uiInit();

    // Initial Values //
    //
    updateProperties();


    init_ = true;
}

oscObjectSettings::~oscObjectSettings()
{
    delete ui;
}

//################//
// FUNCTIONS      //
//################//

// Create generic interface //
//
void
oscObjectSettings::uiInit()
{
	int row = -1;
	QSignalMapper *signalMapper = new QSignalMapper();
	connect(signalMapper, SIGNAL(mapped(const QString&)), this, SIGNAL(onEditingFinished(const QString&)));

	OpenScenario::oscObjectBase::MemberMap members = object_->getMembers();
	for(OpenScenario::oscObjectBase::MemberMap::iterator it = members.begin();it != members.end();it++)
	{
		OpenScenario::oscMember *member = it->second;
		QString memberName = QString::fromStdString(member->getName());
		QLabel *label = new QLabel(memberName);
		ui->objectGridLayout->addWidget(label, ++row, 0);
		const OpenScenario::oscMemberValue::MemberTypes type = member->getType();

		if (type <= 3) // UINT = 0, INT = 1, USHORT = 2, SHORT = 3
		{
			QSpinBox * oscSpinBox = new QSpinBox();
			memberWidgets_.insert(memberName, oscSpinBox);	
			ui->objectGridLayout->addWidget(oscSpinBox, row, 1);
			connect(oscSpinBox, SIGNAL(editingFinished()), this, SLOT(map()));
			signalMapper->setMapping(oscSpinBox, memberName);
		}
		else if (type == OpenScenario::oscMemberValue::MemberTypes::STRING)
		{
			QLineEdit *oscLineEdit = new QLineEdit();
			memberWidgets_.insert(memberName, oscLineEdit);
			ui->objectGridLayout->addWidget(oscLineEdit, row, 1);
			connect(oscLineEdit, SIGNAL(editingFinished()), signalMapper, SLOT(map()));
			signalMapper->setMapping(oscLineEdit, memberName);
		}
		else if ((type == OpenScenario::oscMemberValue::MemberTypes::DOUBLE) || (type == OpenScenario::oscMemberValue::MemberTypes::FLOAT))
		{
			QDoubleSpinBox *oscSpinBox = new QDoubleSpinBox();
			memberWidgets_.insert(memberName, oscSpinBox);	
			ui->objectGridLayout->addWidget(oscSpinBox, row, 1);
			connect(oscSpinBox, SIGNAL(editingFinished()), this, SLOT(map()));
			signalMapper->setMapping(oscSpinBox, memberName);
		}
		else if (type == OpenScenario::oscMemberValue::MemberTypes::OBJECT)
		{
			// new settings
		}
		else if (type == OpenScenario::oscMemberValue::MemberTypes::ENUM)
		{
			QComboBox *oscComboBox = new QComboBox();
//			object_->getBase()->test->driver->body->sex.enumType->enumValues;

			OpenScenario::oscEnum *oscVar = dynamic_cast<OpenScenario::oscEnum *>(member);
			std::map<std::string,int> enumValues = oscVar->enumType->enumValues;
			for (std::map<std::string,int>::iterator it=enumValues.begin(); it!=enumValues.end(); ++it)
			{
				oscComboBox->addItem(QString::fromStdString(it->first));
			}

			memberWidgets_.insert(memberName, oscComboBox);
			ui->objectGridLayout->addWidget(oscComboBox, row, 1);
			connect(oscComboBox, SIGNAL(currentIndexChanged()), this, SLOT(map()));
			signalMapper->setMapping(oscComboBox, memberName);
		}
		else if (type == OpenScenario::oscMemberValue::MemberTypes::BOOL)
		{
			QCheckBox *oscCheckBox = new QCheckBox();
			memberWidgets_.insert(memberName, oscCheckBox);
			ui->objectGridLayout->addWidget(oscCheckBox, row, 1);
			connect(oscCheckBox, SIGNAL(stateChanged()), this, SLOT(map()));
			signalMapper->setMapping(oscCheckBox, memberName);

		}

	}

	// adjust the scroll area
	//
	ui->scrollAreaWidgetContents->setMinimumHeight(ui->scrollAreaWidgetContents->height());
}

void
oscObjectSettings::updateProperties()
{
    if (object_)
    {
		QMap<QString, QWidget *>::const_iterator it;
		for (it = memberWidgets_.constBegin(); it != memberWidgets_.constEnd(); it++)
		{
			OpenScenario::oscMember *member = object_->getMembers().at(it.key().toStdString());
			OpenScenario::oscMemberValue::MemberTypes type = member->getType();
			OpenScenario::oscMemberValue *value = member->getValue();
			QSpinBox * spinBox = dynamic_cast<QSpinBox *>(it.value());
			if (spinBox)
			{
				char buf[100]; 
				sprintf(buf, "%d", value);
				spinBox->setValue(atoi(buf));
				continue;
			}
			QDoubleSpinBox *doubleSpinBox = dynamic_cast<QDoubleSpinBox *>(it.value());
			if (doubleSpinBox)
			{
				char buf[100];
				sprintf(buf, "%f", value);
				doubleSpinBox->setValue(atof(buf));
				continue;
			}
			QLineEdit *lineEdit = dynamic_cast<QLineEdit *>(it.value());
			if (lineEdit)
			{
				char buf[100];
				sprintf(buf, "%c", value);
				lineEdit->setText(QString::fromStdString(buf));
				continue;
			}
			QComboBox *comboBox = dynamic_cast<QComboBox *>(it.value());
			if (comboBox)
			{
				char buf[100];
				sprintf(buf, "%c", value);
				OpenScenario::oscEnum *oscVar = dynamic_cast<OpenScenario::oscEnum *>(member);
				comboBox->setCurrentIndex(oscVar->enumType->getEnum(buf) - 1);
				continue;
			}
			QCheckBox *checkBox = dynamic_cast<QCheckBox *>(it.value());
			if (checkBox)
			{
				char buf[100];
				sprintf(buf, "%c", value);
				if (buf == "true")
				{
					checkBox->setChecked(true);
				}
				else
				{
					checkBox->setChecked(false);
				}
				continue;
			}
		}
    }
}

//################//
// SLOTS          //
//################//

void
oscObjectSettings::onEditingFinished(const QString &name)
{
//    if (valueChanged_)
 //   {
	QWidget *widget = memberWidgets_.value(name);
	OpenScenario::oscMember *member = object_->getMembers().at(name.toStdString());
	OpenScenario::oscMemberValue::MemberTypes type = member->getType();

	switch (type)
	{
	case OpenScenario::oscMemberValue::MemberTypes::INT:
		{
			QSpinBox * spinBox = dynamic_cast<QSpinBox *>(widget);
			int v = spinBox->value();
			OpenScenario::oscValue<int> value(v);
			SetOSCObjectPropertiesCommand<int> *command = new SetOSCObjectPropertiesCommand<int>(element_, name.toStdString(), value);
			break;
		}
	case OpenScenario::oscMemberValue::MemberTypes::UINT:
		{
			QSpinBox * spinBox = dynamic_cast<QSpinBox *>(widget);
			uint v = spinBox->value();
			OpenScenario::oscValue<uint> value(v);
			SetOSCObjectPropertiesCommand<uint> *command = new SetOSCObjectPropertiesCommand<uint>(element_, name.toStdString(), value);
			break;
		}
	case OpenScenario::oscMemberValue::MemberTypes::USHORT:
		{
			QSpinBox * spinBox = dynamic_cast<QSpinBox *>(widget);
			ushort v = spinBox->value();
			OpenScenario::oscValue<ushort> value(v);
			SetOSCObjectPropertiesCommand<ushort> *command = new SetOSCObjectPropertiesCommand<ushort>(element_, name.toStdString(), value);
			getProjectSettings()->executeCommand(command);
			break;
		}
	case OpenScenario::oscMemberValue::MemberTypes::SHORT:
		{
			QSpinBox * spinBox = dynamic_cast<QSpinBox *>(widget);
			short v = spinBox->value();
			OpenScenario::oscValue<short> value(v);
			SetOSCObjectPropertiesCommand<short> *command = new SetOSCObjectPropertiesCommand<short>(element_, name.toStdString(), value);
			getProjectSettings()->executeCommand(command);
			break;
		}
	case OpenScenario::oscMemberValue::MemberTypes::FLOAT:
		{
			QDoubleSpinBox * spinBox = dynamic_cast<QDoubleSpinBox *>(widget);
			float v = spinBox->value();
			OpenScenario::oscValue<float> value(v);
			SetOSCObjectPropertiesCommand<float> *command = new SetOSCObjectPropertiesCommand<float>(element_, name.toStdString(), value);
			getProjectSettings()->executeCommand(command);
			break;
		}
	case OpenScenario::oscMemberValue::MemberTypes::DOUBLE:
		{
			QDoubleSpinBox * spinBox = dynamic_cast<QDoubleSpinBox *>(widget);
			double v = spinBox->value();
			OpenScenario::oscValue<double> value(v);
			SetOSCObjectPropertiesCommand<double> *command = new SetOSCObjectPropertiesCommand<double>(element_, name.toStdString(), value);
			getProjectSettings()->executeCommand(command);
			break;
		}
	case OpenScenario::oscMemberValue::MemberTypes::STRING:
		{
			QLineEdit * lineEdit = dynamic_cast<QLineEdit *>(widget);
			QString v = lineEdit->text();
			OpenScenario::oscValue<std::string> value(v.toStdString());
			SetOSCObjectPropertiesCommand<std::string> *command = new SetOSCObjectPropertiesCommand<std::string>(element_, name.toStdString(), value);
			getProjectSettings()->executeCommand(command);
			break;
		}
	case OpenScenario::oscMemberValue::MemberTypes::BOOL:
		{
			QCheckBox *checkBox = dynamic_cast<QCheckBox *>(widget);
			bool v = checkBox->isChecked();
			OpenScenario::oscValue<bool> value(v);
			SetOSCObjectPropertiesCommand<bool> *command = new SetOSCObjectPropertiesCommand<bool>(element_, name.toStdString(), value);
			getProjectSettings()->executeCommand(command);
			break;
		}
	case OpenScenario::oscMemberValue::MemberTypes::ENUM:
		{
			QComboBox *comboBox = dynamic_cast<QComboBox *>(widget);
			int v = comboBox->currentIndex();
			OpenScenario::oscEnum *oscVar = dynamic_cast<OpenScenario::oscEnum *>(member);

/*			OpenScenario::oscEnumType *oscVar = dynamic_cast<OpenScenario::oscEnumType *>(member);

			OpenScenario::oscValue<oscEnum> value(v);
			SetOSCObjectPropertiesCommand<oscEnum> *command = new SetOSCObjectPropertiesCommand<oscEnum>(element_, name.toStdString(), value);
			getProjectSettings()->executeCommand(command);*/
			break;
		}

		//TODO: Date, time, obejct
	}


	QWidget * focusWidget = QApplication::focusWidget();
	if (focusWidget)
	{
		focusWidget->clearFocus();
	}
}


//##################//
// Observer Pattern //
//##################//

void
oscObjectSettings::updateObserver()
{

    // Parent //
    //
    SettingsElement::updateObserver();
    if (isInGarbage())
    {
        return; // no need to go on
    }

    // oscObject //
    //
/*    int changes = object_->getoscObjectChanges();

    if ((changes & Bridge::CEL_ParameterChange))
    {
        updateProperties();
    }*/
}
