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
#include "oscArrayMember.h"
#include "oscSourceFile.h"

// Settings //
//
#include "ui_oscobjectsettings.h"
#include "oscobjectsettingsstack.hpp"

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
#include <QSizePolicy>
#include <QScrollArea>
#include <QGridLayout> 

// Utils //
//
#include "math.h"

using namespace OpenScenario;

//################//
// CONSTRUCTOR    //
//################//

OSCObjectSettings::OSCObjectSettings(ProjectSettings *projectSettings, OSCObjectSettingsStack *parent, OSCElement *element)
    : QWidget()
	, Observer()
	, ui(new Ui::OSCObjectSettings)
    , init_(false)
    , valueChanged_(false)
	, element_(element)
	, projectSettings_(projectSettings)
	, parentStack_(parent)
{
	object_ = element_->getObject();
	base_ = element_->getOSCBase();
    ui->setupUi(this);

	//oscArrayMember
	//

	oscArrayMember_ = dynamic_cast<OpenScenario::oscArrayMember *>(object_->getOwnMember());

	if(oscArrayMember_)
	{
		uiInitArray();
	}
	else 
	{
		uiInit();
		// Initial Values //
		//
		updateProperties();
	}


    init_ = true;
	parentStack_->addWidget(this);

	// Observer //
    //
    element_->attachObserver(this);

}

OSCObjectSettings::~OSCObjectSettings()
{
	// Observer //
    //
    element_->detachObserver(this);
//    delete ui;
}

//################//
// FUNCTIONS      //
//################//

// Create generic interface //
//
void
OSCObjectSettings::uiInit()
{
	// Widget/Layout //
	//
	QGridLayout *objectGridLayout = new QGridLayout();
	objectGridLayout->setSizeConstraint(QLayout::SetMinAndMaxSize);

	int row = 1;
	// Close Button //
	// not for the first element in the stackedWidget //
	if (parentStack_->getStackSize() == 0)
	{
		objectGridLayout->setContentsMargins(4, 30, 4, 9);
	}
	else
	{
		QPushButton *closeButton = new QPushButton("close", ui->oscGroupBox);
		closeButton->setObjectName(QStringLiteral("close"));
        closeButton->setGeometry(QRect(90, 30, 75, 23));
		connect(closeButton, SIGNAL(pressed()), parentStack_, SLOT(removeWidget()));

		objectGridLayout->setContentsMargins(4, 60, 4, 9);
	}
	
	// Signal Mapper for the value input widgets //
	//
	QSignalMapper *signalMapper = new QSignalMapper(this);
	connect(signalMapper, SIGNAL(mapped(QString)), this, SLOT(onEditingFinished(QString)));

	// values changed mapper //
	//
	QSignalMapper *valueChangedMapper = new QSignalMapper(this);
	connect(valueChangedMapper, SIGNAL(mapped(QString)), this, SLOT(onValueChanged()));


	// Signal Mapper for the objects //
	//
	QSignalMapper *signalPushMapper = new QSignalMapper(this);
	connect(signalPushMapper, SIGNAL(mapped(QString)), this, SLOT(onPushButtonPressed(QString)));

	bool choice = object_->hasChoice();
	int choiceComboBoxRow = ++row;
	if (choice)
	{
		choiceComboBox_ = new QComboBox();
		objectGridLayout->addWidget(choiceComboBox_, choiceComboBoxRow, 0);
		connect(choiceComboBox_, SIGNAL(activated(const QString &)), this, SLOT(onChoiceChanged(const QString &)));

		QPushButton *oscPushButton = new QPushButton();
		oscPushButton->setText("Edit");
		memberWidgets_.insert("choice", oscPushButton);
		objectGridLayout->addWidget(oscPushButton, choiceComboBoxRow, 1);
		connect(oscPushButton, SIGNAL(pressed()), signalPushMapper, SLOT(map()));
		signalPushMapper->setMapping(oscPushButton, "choice");
	}

	OpenScenario::oscObjectBase::MemberMap members = object_->getMembers();
	for(OpenScenario::oscObjectBase::MemberMap::iterator it = members.begin();it != members.end();it++)
	{
		OpenScenario::oscMember *member = it->second;
		QString memberName = QString::fromStdString(member->getName());

		if (choice && object_->isMemberInChoice(member))
		{
			choiceComboBox_->addItem(memberName);
		}
		else
		{
			QLabel *label = new QLabel(memberName);
			formatLabel(label, memberName);
			objectGridLayout->addWidget(label, ++row, 0);


			const OpenScenario::oscMemberValue::MemberTypes type = member->getType();

			if (type <= 3) // UINT = 0, INT = 1, USHORT = 2, SHORT = 3
			{
				QSpinBox * oscSpinBox = new QSpinBox();
				switch (type)
				{
				case OpenScenario::oscMemberValue::MemberTypes::INT:
					{
						oscSpinBox->setMinimum(INT_MIN);
						oscSpinBox->setMaximum(INT_MAX);
						break;
					}
				case OpenScenario::oscMemberValue::MemberTypes::UINT:
					{
						oscSpinBox->setMinimum(0);
						oscSpinBox->setMaximum(INT_MAX);
						break;
					}
				case OpenScenario::oscMemberValue::MemberTypes::SHORT:
					{
						oscSpinBox->setMinimum(SHRT_MIN);
						oscSpinBox->setMaximum(SHRT_MAX);
						break;
					}
				case OpenScenario::oscMemberValue::MemberTypes::USHORT:
					{
						oscSpinBox->setMinimum(0);
						oscSpinBox->setMaximum(USHRT_MAX);
						break;
					}
				default:
					assert("member->getType() not handled"==0);
					break;
				}
				memberWidgets_.insert(memberName, oscSpinBox);	
				objectGridLayout->addWidget(oscSpinBox, row, 1);
				connect(oscSpinBox, SIGNAL(editingFinished()), signalMapper, SLOT(map()));
				signalMapper->setMapping(oscSpinBox, memberName);
				connect(oscSpinBox, SIGNAL(valueChanged(int)), valueChangedMapper, SLOT(map()));
				valueChangedMapper->setMapping(oscSpinBox, memberName);
			}
			else if (type == OpenScenario::oscMemberValue::MemberTypes::STRING)
			{
				QLineEdit *oscLineEdit = new QLineEdit();
				memberWidgets_.insert(memberName, oscLineEdit);
				objectGridLayout->addWidget(oscLineEdit, row, 1);
				connect(oscLineEdit, SIGNAL(editingFinished()), signalMapper, SLOT(map()));
				signalMapper->setMapping(oscLineEdit, memberName);
				connect(oscLineEdit, SIGNAL(textChanged(QString)), valueChangedMapper, SLOT(map()));
				valueChangedMapper->setMapping(oscLineEdit, memberName);
			}
			else if ((type == OpenScenario::oscMemberValue::MemberTypes::DOUBLE) || (type == OpenScenario::oscMemberValue::MemberTypes::FLOAT))
			{
				QDoubleSpinBox *oscSpinBox = new QDoubleSpinBox();
				switch (type)
				{
				case OpenScenario::oscMemberValue::MemberTypes::DOUBLE:
					{
						oscSpinBox->setMinimum(-1.0e+10);
						oscSpinBox->setMaximum(1.0e+10);
						break;
					}
				case OpenScenario::oscMemberValue::MemberTypes::FLOAT:
					{
						oscSpinBox->setMinimum(-1.0e+10);
						oscSpinBox->setMaximum(1.0e+10);
						break;
					}
				default:
					assert("member->getType() not handled"==0);
					break;
				}
				memberWidgets_.insert(memberName, oscSpinBox);	
				objectGridLayout->addWidget(oscSpinBox, row, 1);
				connect(oscSpinBox, SIGNAL(editingFinished()), signalMapper, SLOT(map()));
				signalMapper->setMapping(oscSpinBox, memberName);
				connect(oscSpinBox, SIGNAL(valueChanged(double)), valueChangedMapper, SLOT(map()));
				valueChangedMapper->setMapping(oscSpinBox, memberName);
			}
			else if (type == OpenScenario::oscMemberValue::MemberTypes::OBJECT)
			{
				QPushButton *oscPushButton = new QPushButton();
				oscPushButton->setText("Edit");
				memberWidgets_.insert(memberName, oscPushButton);
				objectGridLayout->addWidget(oscPushButton, row, 1);
				connect(oscPushButton, SIGNAL(pressed()), signalPushMapper, SLOT(map()));
				signalPushMapper->setMapping(oscPushButton, memberName);
			}
			else if (type == OpenScenario::oscMemberValue::MemberTypes::ENUM)
			{
				QComboBox *oscComboBox = new QComboBox();

				oscComboBox->addItem("Choose entry ...");

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
				objectGridLayout->addWidget(oscComboBox, row, 1);
				connect(oscComboBox, SIGNAL(activated(int)), signalMapper, SLOT(map()));
				signalMapper->setMapping(oscComboBox, memberName);
				connect(oscComboBox, SIGNAL(currentIndexChanged(int)), valueChangedMapper, SLOT(map()));
				valueChangedMapper->setMapping(oscComboBox, memberName);
			}
			else if (type == OpenScenario::oscMemberValue::MemberTypes::BOOL)
			{
				QCheckBox *oscCheckBox = new QCheckBox();
				memberWidgets_.insert(memberName, oscCheckBox);
				objectGridLayout->addWidget(oscCheckBox, row, 1);
				connect(oscCheckBox, SIGNAL(clicked(bool)), signalMapper, SLOT(map()));
				signalMapper->setMapping(oscCheckBox, memberName);
				connect(oscCheckBox, SIGNAL(stateChanged(int)), valueChangedMapper, SLOT(map()));
				valueChangedMapper->setMapping(oscCheckBox, memberName);

			}
		}

	}

	// Finish Layout //
	//
	objectGridLayout->setRowStretch(++row, 1); // row x fills the rest of the availlable space
	objectGridLayout->setColumnStretch(2, 1); // column 2 fills the rest of the availlable space

	ui->oscGroupBox->setLayout(objectGridLayout);


}

// Create generic interface for array members//
//
void
OSCObjectSettings::uiInitArray()
{
	// Widget/Layout //
	//
	QGridLayout *objectGridLayout = new QGridLayout();
	objectGridLayout->setSizeConstraint(QLayout::SetMinAndMaxSize);

	// Close Button //
	// not for the first element in the stackedWidget //
	if (parentStack_->getStackSize() == 0)
	{
		objectGridLayout->setContentsMargins(4, 30, 4, 9);
	}
	else
	{
		QPushButton *closeButton = new QPushButton("close", ui->oscGroupBox);
		closeButton->setObjectName(QStringLiteral("close"));
        closeButton->setGeometry(QRect(90, 30, 75, 23));
		connect(closeButton, SIGNAL(pressed()), parentStack_, SLOT(removeWidget()));

		objectGridLayout->setContentsMargins(4, 60, 4, 9);
	}

	QPixmap recycleIcon(":/icons/recycle.png");

	ArrayDropArea *recycleArea = new ArrayDropArea(this, &recycleIcon);
	objectGridLayout->addWidget(recycleArea, 0, 0);

	OpenScenario::oscObjectBase::MemberMap map = object_->getMembers();
	auto it = map.begin();
	memberName_ = QString::fromStdString(it->first);
	QLabel *label = new QLabel(memberName_);
	objectGridLayout->addWidget(label, 1, 0);

	// Tree for array objects
	//
	arrayTree_ = new QTreeWidget(this);
	arrayTree_->setObjectName("ArrayTree");
	arrayTree_->setHeaderHidden(true);
	arrayTree_->setDragEnabled(true);
	connect(arrayTree_, SIGNAL(itemClicked(QTreeWidgetItem *, int)), this, SLOT(onArrayElementClicked(QTreeWidgetItem *, int)));

	
	//oscArrayMember
	//
	// emtpy item to create new elements //
	//

	updateTree();
	
	objectGridLayout->addWidget(arrayTree_, 1, 0);


	// Finish Layout //
    //
    objectGridLayout->setColumnStretch(2, 1); // column 2 fills the rest of the availlable space

	ui->oscGroupBox->setLayout(objectGridLayout);


}

void OSCObjectSettings::updateTree()
{
	arrayTree_->clear();

	QTreeWidgetItem *item = new QTreeWidgetItem();
	item->setText(0, "New " + memberName_);
	arrayTree_->addTopLevelItem(item);

	//generate the children members
	for (int i = 0; i < oscArrayMember_->size(); i++)
	{
		addTreeItem(arrayTree_, i+1);
	}
}

void OSCObjectSettings::addTreeItem(QTreeWidget *arrayTree, int name)
{

	QTreeWidgetItem *item = new QTreeWidgetItem();
	item->setText(0, QString::number(name));
	arrayTree->addTopLevelItem(item);

}

void OSCObjectSettings::formatLabel(QLabel *label, const QString &memberName)
{
	label->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
	if (memberName.size() > 16)
	{
		QStringList list = memberName.split(QRegExp("[A-Z]"));
		QString line;
		int separator = 16;

		for (int i = 0; i < list.size(); i++)
		{
			QString temp = line + list.at(i) + " ";
			if (temp.size() > 16)
			{
				separator = line.size() - 1;
				break;
			}
			line = temp;
		}

		QString name = memberName.left(separator) + "\n" + memberName.right(memberName.size() - separator);
		label->setText(name);
		label->setFixedHeight(20);
	}
	else
	{
		label->setFixedHeight(25);
	}
}

void
OSCObjectSettings::updateProperties()
{

    if (object_)
    {
		QMap<QString, QWidget *>::const_iterator it;
		for (it = memberWidgets_.constBegin(); it != memberWidgets_.constEnd(); it++)
		{
			if (it.key() == "choice")
			{
				foreach (OpenScenario::oscMember *choiceMember, object_->getChoice())
				{
					if (choiceMember->exists())
					{
						loadProperties(choiceMember, it.value());
						break;
					}
				}
			}
			else
			{
				OpenScenario::oscMember *member = object_->getMember(it.key().toStdString());

				if (!member->exists())
				{
					continue;
				} 

				loadProperties(member, it.value());
			}
		}

	}
}

void
	OSCObjectSettings::loadProperties(OpenScenario::oscMember *member, QWidget *widget)
{
	if (object_->isMemberInChoice(member))
	{
		choiceComboBox_->setCurrentText(QString::fromStdString(member->getName()));
	}

	OpenScenario::oscMemberValue *value = member->getOrCreateValue();

	if (QSpinBox * spinBox = dynamic_cast<QSpinBox *>(widget))
	{

		if (oscIntValue *iv = dynamic_cast<oscIntValue *>(value))
		{
			spinBox->setValue(iv->getValue());
		}
		else if (oscUIntValue *uv = dynamic_cast<oscUIntValue *>(value))
		{
			spinBox->setValue(uv->getValue());
		}
		else if (oscShortValue *sv = dynamic_cast<oscShortValue *>(value))
		{
			spinBox->setValue(sv->getValue());
		}
		else if (oscUShortValue *usv = dynamic_cast<oscUShortValue *>(value))
		{
			spinBox->setValue(usv->getValue());
		}
	}
	else if (QDoubleSpinBox *doubleSpinBox = dynamic_cast<QDoubleSpinBox *>(widget))
	{
		if (oscFloatValue *fv = dynamic_cast<oscFloatValue *>(value))
		{
			doubleSpinBox->setValue(fv->getValue());
		}
		else if (oscDoubleValue *dv = dynamic_cast<oscDoubleValue *>(value))
		{
			doubleSpinBox->setValue(dv->getValue());
		}
	}
	else if (QLineEdit *lineEdit = dynamic_cast<QLineEdit *>(widget))
	{
		oscStringValue *sv = dynamic_cast<oscStringValue *>(value);
		if (sv)
		{
			lineEdit->setText(QString::fromStdString(sv->getValue()));
		}
	}
	else if (QComboBox *comboBox = dynamic_cast<QComboBox *>(widget))
	{
		oscIntValue *iv = dynamic_cast<oscIntValue *>(value);
		if (iv)
		{
			comboBox->setCurrentIndex(iv->getValue() + 1);
		}
	}
	else if (QCheckBox *checkBox = dynamic_cast<QCheckBox *>(widget))
	{
		oscBoolValue *iv = dynamic_cast<oscBoolValue *>(value);
		if (iv)
		{
			checkBox->setChecked(iv->getValue());
		}
	}
}

void 
OSCObjectSettings::onDeleteArrayElement()
{

	QTreeWidgetItem *item = arrayTree_->selectedItems().at(0);
	int j = arrayTree_->indexOfTopLevelItem(item);
	if (j > 0)
	{
		OpenScenario::oscObjectBase *object = oscArrayMember_->at(--j);
		RemoveOSCArrayMemberCommand *command = new RemoveOSCArrayMemberCommand(oscArrayMember_, object_, j, base_->getOSCElement(object));
		projectSettings_->executeCommand(command);
	}

}

//################//
// SLOTS          //
//################//

void
OSCObjectSettings::onValueChanged()
{
    valueChanged_ = true;
}


void
OSCObjectSettings::onEditingFinished(QString name)
{
	if (valueChanged_)
	{
		QWidget *widget = memberWidgets_.value(name);
		
		OpenScenario::oscMember *member = object_->getMember(name.toStdString());
		OpenScenario::oscMemberValue::MemberTypes type = member->getType();

		switch (type)
		{
		case OpenScenario::oscMemberValue::MemberTypes::INT:
			{
				QSpinBox * spinBox = dynamic_cast<QSpinBox *>(widget);
				int v = spinBox->value();
				SetOSCValuePropertiesCommand<int> *command = new SetOSCValuePropertiesCommand<int>(element_, object_, name.toStdString(), v);
				projectSettings_->executeCommand(command);
				break;
			}
		case OpenScenario::oscMemberValue::MemberTypes::UINT:
			{
				QSpinBox * spinBox = dynamic_cast<QSpinBox *>(widget);
				uint v = spinBox->value();
				SetOSCValuePropertiesCommand<uint> *command = new SetOSCValuePropertiesCommand<uint>(element_, object_, name.toStdString(), v);
				projectSettings_->executeCommand(command);
				break;
			}
		case OpenScenario::oscMemberValue::MemberTypes::USHORT:
			{
				QSpinBox * spinBox = dynamic_cast<QSpinBox *>(widget);
				ushort v = spinBox->value();
				SetOSCValuePropertiesCommand<ushort> *command = new SetOSCValuePropertiesCommand<ushort>(element_, object_, name.toStdString(), v);
				projectSettings_->executeCommand(command);
				break;
			}
		case OpenScenario::oscMemberValue::MemberTypes::SHORT:
			{
				QSpinBox * spinBox = dynamic_cast<QSpinBox *>(widget);
				short v = spinBox->value();
				SetOSCValuePropertiesCommand<short> *command = new SetOSCValuePropertiesCommand<short>(element_, object_, name.toStdString(), v);
				projectSettings_->executeCommand(command);
				break;
			}
		case OpenScenario::oscMemberValue::MemberTypes::FLOAT:
			{
				QDoubleSpinBox * spinBox = dynamic_cast<QDoubleSpinBox *>(widget);
				float v = spinBox->value();
				SetOSCValuePropertiesCommand<float> *command = new SetOSCValuePropertiesCommand<float>(element_, object_, name.toStdString(), v);
				projectSettings_->executeCommand(command);
				break;
			}
		case OpenScenario::oscMemberValue::MemberTypes::DOUBLE:
			{
				QDoubleSpinBox * spinBox = dynamic_cast<QDoubleSpinBox *>(widget);
				double v = spinBox->value();
				SetOSCValuePropertiesCommand<double> *command = new SetOSCValuePropertiesCommand<double>(element_, object_, name.toStdString(), v);
				projectSettings_->executeCommand(command);
				break;
			}
		case OpenScenario::oscMemberValue::MemberTypes::STRING:
			{
				QLineEdit * lineEdit = dynamic_cast<QLineEdit *>(widget);
				QString v = lineEdit->text();
				SetOSCValuePropertiesCommand<std::string> *command = new SetOSCValuePropertiesCommand<std::string>(element_, object_, name.toStdString(), v.toStdString());
				projectSettings_->executeCommand(command);
				break;
			}
		case OpenScenario::oscMemberValue::MemberTypes::BOOL:
			{
				QCheckBox *checkBox = dynamic_cast<QCheckBox *>(widget);
				bool v = checkBox->isChecked();
				SetOSCValuePropertiesCommand<bool> *command = new SetOSCValuePropertiesCommand<bool>(element_, object_, name.toStdString(), v);
				projectSettings_->executeCommand(command);
				break;
			}
		case OpenScenario::oscMemberValue::MemberTypes::ENUM:
			{
				QComboBox *comboBox = dynamic_cast<QComboBox *>(widget);
				int v = comboBox->currentIndex();
				SetOSCValuePropertiesCommand<int> *command = new SetOSCValuePropertiesCommand<int>(element_, object_, name.toStdString(), v - 1);
				projectSettings_->executeCommand(command);

				break;
			}
		case OpenScenario::oscMemberValue::MemberTypes::OBJECT:
		case OpenScenario::oscMemberValue::MemberTypes::DATE_TIME:
			//TODO
            break;
		}
		valueChanged_ = false;
	}


	QWidget * focusWidget = QApplication::focusWidget();
	if (focusWidget)
	{
		focusWidget->clearFocus();
	}
}

void
OSCObjectSettings::onChoiceChanged(const QString &memberName)
{
	if (memberName != lastComboBoxChoice_)
	{
		OpenScenario::oscMember *member = object_->getMember(memberName.toStdString());
		OpenScenario::oscObjectBase *obj = member->getOrCreateObject();

		lastComboBoxChoice_ = memberName;
	}


		// read default values
		//
	//	OpenScenario::oscSourceFile *sourceFile = object_->getSource();
	//	OpenScenario::oscObjectBase *obj = object_->readDefaultXMLObject( sourceFile->getSrcFileHref(), memberName.toStdString(), object_->getMember(memberName.toStdString())->getTypeName(), sourceFile);
	//	AddOSCObjectCommand *command = new AddOSCObjectCommand(object_, base_, 


}

OpenScenario::oscObjectBase * 
OSCObjectSettings::onPushButtonPressed(QString name)
{
	OpenScenario::oscObjectBase *object = NULL;

	if (oscArrayMember_)
	{
		object = oscArrayMember_->at(name.toInt()-1);
	}
	else if (name == "choice")
	{
		object = object_->getMember(choiceComboBox_->currentText().toStdString())->getOrCreateObject();

	}
	else
	{
		object = object_->getMember(name.toStdString())->getOrCreateObject();
	}

	OSCElement *memberElement = base_->getOSCElement(object);

	OSCObjectSettings *oscSettings = new OSCObjectSettings(projectSettings_, parentStack_, memberElement);

	return object;
}

void
OSCObjectSettings::onNewArrayElement()
{
	OSCElement *oscElement = new OSCElement(memberName_);
	if (oscElement)
	{
		OpenScenario::oscSourceFile *sourceFile = object_->getSource();
		OpenScenario::oscObjectBase *obj = object_->readDefaultXMLObject( sourceFile->getSrcFileHref(), memberName_.toStdString(), object_->getMember(memberName_.toStdString())->getTypeName(), sourceFile);

		AddOSCArrayMemberCommand *command = new AddOSCArrayMemberCommand(oscArrayMember_, object_, obj, memberName_.toStdString(), base_, oscElement);
		projectSettings_->executeCommand(command);

		OSCObjectSettings *oscSettings = new OSCObjectSettings(projectSettings_, parentStack_, oscElement);
	}

}

void
OSCObjectSettings::onArrayElementClicked(QTreeWidgetItem *item, int column)
{
	QString name = item->text(0);
	if (name.contains("New")) 
	{
		onNewArrayElement();
	}
	else
	{
		onPushButtonPressed(name);
	}
}


//##################//
// Observer Pattern //
//##################//

void
OSCObjectSettings::updateObserver()
{
	if (parentStack_->isInGarbage())
    {
        return; // no need to go on
    }

    // oscObject //
    //
	int changes = element_->getOSCElementChanges();

	if (changes & OSCElement::COE_ParameterChange)
    {
        updateProperties();
    }

	if (changes & OSCElement::COE_ChildChanged)
	{
		if (oscArrayMember_)
		{
			updateTree();
		}
	}

	changes = element_->getDataElementChanges();
	if ((changes & DataElement::CDE_DataElementAdded) || (changes & DataElement::CDE_DataElementRemoved))
	{

	}
}


//###############################//
// DropArea for the recycle bin //
//
//#############################//
ArrayDropArea::ArrayDropArea(OSCObjectSettings *settings, QPixmap *pixmap)
    : DropArea(pixmap)
	, settings_(settings)
{
}

//################//
// EVENTS         //
//################//

void 
ArrayDropArea::dropEvent(QDropEvent *event)
{
	settings_->onDeleteArrayElement();

	DropArea::dropEvent(event);
}
