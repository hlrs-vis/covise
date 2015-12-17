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

#include "oscobjectsettings.hpp"

#include "src/mainwindow.hpp"

// OpenScenario //
//
#include "oscObjectBase.h"
#include "OpenScenarioBase.h"
#include "oscVariables.h"

// Settings //
//
#include "ui_oscobjectsettings.h"

// Commands //
//
#include "src/data/commands/osccommands.hpp"
#include "src/data/commands/dataelementcommands.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"

// Data //
#include "src/data/oscsystem/oscelement.hpp"
#include "src/data/oscsystem/oscbase.hpp"

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
#include <QPushButton>

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
	QSignalMapper *signalMapper = new QSignalMapper(this);
	connect(signalMapper, SIGNAL(mapped(QString)), this, SLOT(onEditingFinished(QString)));
	QSignalMapper *signalPushMapper = new QSignalMapper(this);
	connect(signalPushMapper, SIGNAL(mapped(QString)), this, SLOT(onPushButtonPressed(QString)));

	OpenScenario::oscObjectBase::MemberMap members = object_->getMembers();
	for(OpenScenario::oscObjectBase::MemberMap::iterator it = members.begin();it != members.end();it++)
	{
		OpenScenario::oscMember *member = it->second;
		QString memberName = QString::fromStdString(member->getName());
		QLabel *label = new QLabel(memberName);
		if (memberName.size() > 16)
		{
			//QStringList list = memberName.split(QRegExp("[A-Z]"));
			QString temp = memberName;
			temp.truncate(16);
			QStringList list = memberName.split(temp);
			QString name = list.takeFirst() + "\n" + list.takeLast();
			label->setText(temp);
		}

		ui->objectGridLayout->addWidget(label, ++row, 0);
		const OpenScenario::oscMemberValue::MemberTypes type = member->getType();

		if (type <= 3) // UINT = 0, INT = 1, USHORT = 2, SHORT = 3
		{
			QSpinBox * oscSpinBox = new QSpinBox();
			memberWidgets_.insert(memberName, oscSpinBox);	
			ui->objectGridLayout->addWidget(oscSpinBox, row, 1);
			connect(oscSpinBox, SIGNAL(editingFinished()), signalMapper, SLOT(map()));
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
			connect(oscSpinBox, SIGNAL(editingFinished()), signalMapper, SLOT(map()));
			signalMapper->setMapping(oscSpinBox, memberName);
		}
		else if (type == OpenScenario::oscMemberValue::MemberTypes::OBJECT)
		{
			QPushButton *oscPushButton = new QPushButton();
			oscPushButton->setText("Edit");
			memberWidgets_.insert(memberName, oscPushButton);
			ui->objectGridLayout->addWidget(oscPushButton, row, 1);
			connect(oscPushButton, SIGNAL(pressed()), signalPushMapper, SLOT(map()));
			signalPushMapper->setMapping(oscPushButton, memberName);
		}
		else if (type == OpenScenario::oscMemberValue::MemberTypes::ENUM)
		{
			QComboBox *oscComboBox = new QComboBox();

			OpenScenario::oscEnum *oscVar = dynamic_cast<OpenScenario::oscEnum *>(member);
			std::map<std::string,int> enumValues = oscVar->enumType->enumValues;
			for (int i = 0; i < enumValues.size(); i++)
			{
				for (std::map<std::string,int>::iterator it=enumValues.begin(); it!=enumValues.end(); ++it)
				{
					if (it->second == i)
					{
						oscComboBox->addItem(QString::fromStdString(it->first));
						break;
					}
				}
			}

			memberWidgets_.insert(memberName, oscComboBox);
			ui->objectGridLayout->addWidget(oscComboBox, row, 1);
			connect(oscComboBox, SIGNAL(currentIndexChanged(int)), signalMapper, SLOT(map()));
		}
		else if (type == OpenScenario::oscMemberValue::MemberTypes::BOOL)
		{
			QCheckBox *oscCheckBox = new QCheckBox();
			memberWidgets_.insert(memberName, oscCheckBox);
			signalMapper->setMapping(oscCheckBox, memberName);
			ui->objectGridLayout->addWidget(oscCheckBox, row, 1);
			connect(oscCheckBox, SIGNAL(stateChanged(int)), signalMapper, SLOT(map()));
			signalMapper->setMapping(oscCheckBox, memberName);

		}

	}

	// adjust the scroll area
	//
//	ui->scrollAreaWidgetContents->setMinimumHeight(ui->scrollAreaWidgetContents->height());
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
				oscIntValue *iv = dynamic_cast<oscIntValue *>(it.value());
				if (iv)
				{
					spinBox->setValue(iv->getValue());
				}
				continue;
			}
			QDoubleSpinBox *doubleSpinBox = dynamic_cast<QDoubleSpinBox *>(it.value());
			if (doubleSpinBox)
			{
				oscFloatValue *fv = dynamic_cast<oscFloatValue *>(value);
				if(fv)
				{
				   doubleSpinBox->setValue(fv->getValue());
				}
				continue;
			}
			QLineEdit *lineEdit = dynamic_cast<QLineEdit *>(it.value());
			if (lineEdit)
			{
				oscStringValue *sv = dynamic_cast<oscStringValue *>(it.value());
				if (sv)
				{
					lineEdit->setText(QString::fromStdString(sv->getValue()));
				}
				continue;
			}
			QComboBox *comboBox = dynamic_cast<QComboBox *>(it.value());
			if (comboBox)
			{
				oscIntValue *iv = dynamic_cast<oscIntValue *>(it.value());
				if (iv)
				{
					comboBox->setCurrentIndex(iv->getValue());
				}

				continue;
			}
			QCheckBox *checkBox = dynamic_cast<QCheckBox *>(it.value());
			if (checkBox)
			{
				oscIntValue *iv = dynamic_cast<oscIntValue *>(it.value());
				if (iv)
				{
					if (iv->getValue() == 1)
					{
						checkBox->setChecked(true);
					}
					else
					{
						checkBox->setChecked(false);
					}
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
oscObjectSettings::onEditingFinished(QString name)
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
			SetOSCValuePropertiesCommand<int> *command = new SetOSCValuePropertiesCommand<int>(element_, name.toStdString(), v);
			break;
		}
	case OpenScenario::oscMemberValue::MemberTypes::UINT:
		{
			QSpinBox * spinBox = dynamic_cast<QSpinBox *>(widget);
			uint v = spinBox->value();
			SetOSCValuePropertiesCommand<uint> *command = new SetOSCValuePropertiesCommand<uint>(element_, name.toStdString(), v);
			break;
		}
	case OpenScenario::oscMemberValue::MemberTypes::USHORT:
		{
			QSpinBox * spinBox = dynamic_cast<QSpinBox *>(widget);
			ushort v = spinBox->value();
			SetOSCValuePropertiesCommand<ushort> *command = new SetOSCValuePropertiesCommand<ushort>(element_, name.toStdString(), v);
			getProjectSettings()->executeCommand(command);
			break;
		}
	case OpenScenario::oscMemberValue::MemberTypes::SHORT:
		{
			QSpinBox * spinBox = dynamic_cast<QSpinBox *>(widget);
			short v = spinBox->value();
			SetOSCValuePropertiesCommand<short> *command = new SetOSCValuePropertiesCommand<short>(element_, name.toStdString(), v);
			getProjectSettings()->executeCommand(command);
			break;
		}
	case OpenScenario::oscMemberValue::MemberTypes::FLOAT:
		{
			QDoubleSpinBox * spinBox = dynamic_cast<QDoubleSpinBox *>(widget);
			float v = spinBox->value();
			SetOSCValuePropertiesCommand<float> *command = new SetOSCValuePropertiesCommand<float>(element_, name.toStdString(), v);
			getProjectSettings()->executeCommand(command);
			break;
		}
	case OpenScenario::oscMemberValue::MemberTypes::DOUBLE:
		{
			QDoubleSpinBox * spinBox = dynamic_cast<QDoubleSpinBox *>(widget);
			double v = spinBox->value();
			SetOSCValuePropertiesCommand<double> *command = new SetOSCValuePropertiesCommand<double>(element_, name.toStdString(), v);
			getProjectSettings()->executeCommand(command);
			break;
		}
	case OpenScenario::oscMemberValue::MemberTypes::STRING:
		{
			QLineEdit * lineEdit = dynamic_cast<QLineEdit *>(widget);
			QString v = lineEdit->text();
			SetOSCValuePropertiesCommand<std::string> *command = new SetOSCValuePropertiesCommand<std::string>(element_, name.toStdString(), v.toStdString());
			getProjectSettings()->executeCommand(command);
			break;
		}
	case OpenScenario::oscMemberValue::MemberTypes::BOOL:
		{
			QCheckBox *checkBox = dynamic_cast<QCheckBox *>(widget);
			bool v = checkBox->isChecked();
			SetOSCValuePropertiesCommand<bool> *command = new SetOSCValuePropertiesCommand<bool>(element_, name.toStdString(), v);
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

void 
oscObjectSettings::onPushButtonPressed(QString name)
{
/*	if (element_ && element_->isElementSelected())
	{
		DeselectDataElementCommand *command = new DeselectDataElementCommand(element_, NULL);
		getProjectSettings()->executeCommand(command);
	}*/

	OpenScenario::oscMember *member = object_->getMembers().at(name.toStdString());

	OSCBase *base = element_->getOSCBase();
	OSCElement *subElement = base->getOSCElement(member->getObject());
	if (subElement)
	{
		const OpenScenario::oscObjectBase *object = subElement->getObject();
		OpenScenario::oscObjectBase::MemberMap members = object->getMembers();
		for(OpenScenario::oscObjectBase::MemberMap::iterator it = members.begin();it != members.end();it++)
		{
			if (!it->second->exists())
			{
				OpenScenario::oscMemberValue::MemberTypes memberType = it->second->getType();
				if (memberType == OpenScenario::oscMemberValue::MemberTypes::OBJECT)
				{
					OSCElement *memberElement = new OSCElement(QString::fromStdString(it->first));

					AddOSCObjectCommand *command = new AddOSCObjectCommand(object, base, it->first, memberElement, NULL);
					getProjectSettings()->executeCommand(command);
				}
				else if (memberType == OpenScenario::oscMemberValue::MemberTypes::BOOL)
				{
					bool v = true;
					AddOSCValueCommand<bool> *command = new AddOSCValueCommand<bool>(object, it->first, v);
					getProjectSettings()->executeCommand(command);
				}
				else if (memberType == OpenScenario::oscMemberValue::MemberTypes::STRING)
				{
					std::string v = "";
					AddOSCValueCommand<std::string> *command = new AddOSCValueCommand<std::string>(object, it->first, v);
					getProjectSettings()->executeCommand(command);
				}
				else if (memberType == OpenScenario::oscMemberValue::MemberTypes::ENUM)
				{
				}
				else
				{
					int v = 0;
					AddOSCValueCommand<int> *command = new AddOSCValueCommand<int>(object, it->first, v);
					getProjectSettings()->executeCommand(command);
				}
			}
		}

/*		SelectDataElementCommand *command = new SelectDataElementCommand(subElement, NULL);
		getProjectSettings()->executeCommand(command);*/
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
