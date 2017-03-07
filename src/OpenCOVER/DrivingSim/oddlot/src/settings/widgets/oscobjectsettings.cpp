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
#include "oscCatalog.h"

// Settings //
//
#include "ui_oscobjectsettings.h"
#include "oscobjectsettingsstack.hpp"

// Commands //
//
#include "src/data/commands/osccommands.hpp"
#include "src/data/commands/dataelementcommands.hpp"
#include "src/data/projectdata.hpp"
#include "src/data/changemanager.hpp"

// GUI //
//
#include "src/gui/projectwidget.hpp"
#include "src/gui/tools/osceditortool.hpp"
#include "src/gui/tools/toolmanager.hpp"

// Data //
#include "src/data/oscsystem/oscelement.hpp"
#include "src/data/oscsystem/oscbase.hpp"

// Graph //
#include "src/graph/editors/osceditor.hpp"

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
#include <QMessageBox>
#include <QDateTimeEdit>

// Utils //
//
#include "math.h"

using namespace OpenScenario;

//################//
// CONSTRUCTOR    //
//################//

OSCObjectSettings::OSCObjectSettings(ProjectSettings *projectSettings, OSCObjectSettingsStack *parent, OSCElement *element, OpenScenario::oscMember *member, OpenScenario::oscObjectBase *parentObject)
    : QWidget()
	, Observer()
	, ui(new Ui::OSCObjectSettings)
    , init_(false)
    , valueChanged_(false)
	, object_(NULL)
	, element_(element)
	, projectSettings_(projectSettings)
	, parentStack_(parent)
	, closeCount_(false)
	, member_(member)
	, oscArrayMember_(NULL)
	, parentObject_(parentObject)
{

	if (element_)
	{
		object_ = element_->getObject();
	}
	base_ = projectSettings_->getProjectData()->getOSCBase();
	oscBase_ = base_->getOpenScenarioBase();
	toolManager_ = projectSettings_->getProjectWidget()->getMainWindow()->getToolManager();
    ui->setupUi(this);

	//oscArrayMember
	//

	if (!object_ && member_)
	{
		oscArrayMember_ = dynamic_cast<OpenScenario::oscArrayMember *>(member_);
	}

    if (parentStack_)
	{
		OSCObjectSettings *lastSettings = static_cast<OSCObjectSettings *>(parentStack_->getLastWidget());
		if (lastSettings)
		{
			objectStackText_ = lastSettings->getStackText();
		}
	}

	if(oscArrayMember_)
	{
		uiInitArray();
	}
	else 
	{
		if (object_->getOwnMember())
		{
			member_ = object_->getOwnMember();
			objectStackText_ += QString::fromStdString(member_->getName()) + "/";
		}
		else
		{
			OpenScenario::oscCatalog *catalog = dynamic_cast<OpenScenario::oscCatalog *>(object_->getParentObj());
			if (catalog)
			{
				objectStackText_ += QString::fromStdString(catalog->getCatalogName()) + "/";
			}
		}

		uiInit();
		// Initial Values //
		//
		updateProperties();
	}


    init_ = true;
	parentStack_->addWidget(this);

	// Observer //
    //
	if (element_)
	{
		element_->attachObserver(this);
	}

}

OSCObjectSettings::~OSCObjectSettings()
{
	if (member_)
	{
		QString type = QString::fromStdString(member_->getTypeName());
		if (type == "oscTrajectory")
		{

			// Connect with the ToolManager to send the selected signal or object //
			//
			if (toolManager_)
			{
				toolManager_->enableOSCEditorToolButton(false);
			}
		}
	}

	memberWidgets_.clear();

	// Observer //
    //
	if (element_)
	{
		element_->detachObserver(this);
	}

    delete ui;
}

//################//
// FUNCTIONS      //
//################//

// Create generic interface //
//
void
OSCObjectSettings::uiInit()
{;
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
		objectStackTextlabel_ = new QLabel(objectStackText_, ui->oscGroupBox);
		int rows = formatDirLabel(objectStackTextlabel_, objectStackText_);
		objectStackTextlabel_->setGeometry(10, 50, ui->oscGroupBox->width(), objectStackTextlabel_->height());

		QPushButton *closeButton = new QPushButton("close", ui->oscGroupBox);
		closeButton->setObjectName(QStringLiteral("close"));
        closeButton->setGeometry(QRect(90, 30, 75, 23));
		connect(closeButton, SIGNAL(pressed()), this, SLOT(onCloseWidget()));

		objectGridLayout->setContentsMargins(4, 60 + (++rows)*20, 4, 9);
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

	// Signal mapper for arrays //
	//
	QSignalMapper *signalArrayPushMapper = new QSignalMapper(this);
    connect(signalArrayPushMapper, SIGNAL(mapped(QString)), this, SLOT(onArrayPushButtonPressed(QString)));


	OpenScenario::oscObjectBase::MemberMap members = object_->getMembers();
	for(OpenScenario::oscObjectBase::MemberMap::iterator it = members.begin();it != members.end();it++)
	{
		OpenScenario::oscMember *member = (*it).member;
		QString memberName = QString::fromStdString(member->getName());

		if (member->isChoice())
		{
			short int choiceNumber = member->choiceNumber();
			if (!choiceComboBox_.contains(choiceNumber))
			{
				int choiceComboBoxRow = ++row;

				QComboBox *comboBox = new QComboBox();
				objectGridLayout->addWidget(comboBox, choiceComboBoxRow, 0);
				connect(comboBox, SIGNAL(activated(const QString &)), this, SLOT(onChoiceChanged(const QString &)));

				QString name = "choice" + QString::number(choiceNumber);

				QPushButton *oscPushButton = new QPushButton();
				oscPushButton->setText("Edit");
				oscPushButton->setObjectName(name);

				memberWidgets_.insert(name, oscPushButton);
				objectGridLayout->addWidget(oscPushButton, choiceComboBoxRow, 1);
				connect(oscPushButton, SIGNAL(pressed()), signalPushMapper, SLOT(map()));
				signalPushMapper->setMapping(oscPushButton, name);
				choiceComboBox_.insert(choiceNumber, comboBox);
				choiceComboBox_.value(choiceNumber)->addItem(memberName);

				if (!object_->getChosenMember(choiceNumber))
				{
					member->setSelected(true);
				}
			}
			else
			{
				choiceComboBox_.value(choiceNumber)->addItem(memberName);
			}
		}
		else
		{
			QLabel *label = new QLabel(memberName);
			label->setObjectName(memberName);
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

				OpenScenario::oscArrayMember *arrayMember = dynamic_cast<OpenScenario::oscArrayMember *>(member);
				if (arrayMember)
				{
					oscPushButton->setText("Edit List");
					connect(oscPushButton, SIGNAL(pressed()), signalArrayPushMapper, SLOT(map()));
					signalArrayPushMapper->setMapping(oscPushButton, memberName);
				}
				else
				{
					oscPushButton->setText("Edit");
					connect(oscPushButton, SIGNAL(pressed()), signalPushMapper, SLOT(map()));
					signalPushMapper->setMapping(oscPushButton, memberName);
				}

				memberWidgets_.insert(memberName, oscPushButton);
				objectGridLayout->addWidget(oscPushButton, row, 1);
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
			else if (type == OpenScenario::oscMemberValue::MemberTypes::DATE_TIME)
			{
				QDateTimeEdit *oscDateTimeEdit = new QDateTimeEdit();
				memberWidgets_.insert(memberName, oscDateTimeEdit);
				objectGridLayout->addWidget(oscDateTimeEdit, row, 1);
				connect(oscDateTimeEdit, SIGNAL(editingFinished()), signalMapper, SLOT(map()));
				signalMapper->setMapping(oscDateTimeEdit, memberName);
				connect(oscDateTimeEdit, SIGNAL(dateTimeChanged(QDateTime)), valueChangedMapper, SLOT(map()));
				valueChangedMapper->setMapping(oscDateTimeEdit, memberName); 
			}
		}

	}

	// Finish Layout //
	//
	objectGridLayout->setRowStretch(++row, 1); // row x fills the rest of the availlable space
	objectGridLayout->setColumnStretch(2, 1); // column 2 fills the rest of the availlable space

	ui->oscGroupBox->setLayout(objectGridLayout);

	if (member_)
	{
		std::string type = member_->getTypeName();
		if (type == "oscTrajectory")
		{			
			OpenScenarioEditor *oscEditor = dynamic_cast<OpenScenarioEditor *>(projectSettings_->getProjectWidget()->getProjectEditor());
			if (oscEditor)
			{
				oscEditor->setTrajectoryElement(element_);
			}

			// Connect with the ToolManager to send the selected signal or object //
			//
			if (toolManager_)
			{
				toolManager_->enableOSCEditorToolButton(true);
			}

			element_->addOSCElementChanges(OSCElement::COE_SettingChanged);
		}
	}
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
		QString text = "List of " + objectStackText_ + QString::fromStdString(oscArrayMember_->getName());
		objectStackTextlabel_ = new QLabel(text, ui->oscGroupBox);
		int rows = formatDirLabel(objectStackTextlabel_, text);
		objectStackTextlabel_->setGeometry(10, 50, ui->oscGroupBox->width(), objectStackTextlabel_->height());

		QPushButton *closeButton = new QPushButton("close", ui->oscGroupBox);
		closeButton->setObjectName(QStringLiteral("close"));
        closeButton->setGeometry(QRect(90, 30, 75, 23));
		connect(closeButton, SIGNAL(pressed()), parentStack_, SLOT(stackRemoveWidget()));

		objectGridLayout->setContentsMargins(4, 60 + (++rows)*20, 4, 9);
	}

	QPixmap recycleIcon(":/icons/recycle.png");

	ArrayDropArea *recycleArea = new ArrayDropArea(this, &recycleIcon);
	objectGridLayout->addWidget(recycleArea, 0, 0);

	memberName_ = QString::fromStdString(member_->getName());
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
		OpenScenario::oscObjectBase *object = oscArrayMember_->at(i);

		addTreeItem(arrayTree_, i+1, object);
	}
}

void OSCObjectSettings::addTreeItem(QTreeWidget *arrayTree, int name, OpenScenario::oscObjectBase *object)
{

	QTreeWidgetItem *item = new QTreeWidgetItem();
	item->setText(0, QString::number(name));
	if (object->validate() == OpenScenario::oscObjectBase::VAL_valid)
	{
		item->setTextColor(0, QColor(128, 195, 66));   // lightgreen
	}
	else
	{
		item->setTextColor(0, Qt::white);
	}

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

int OSCObjectSettings::formatDirLabel(QLabel *label, const QString &memberName)
{
	int rows = 1;

	label->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
	if (memberName.size() > 30)
	{
		QStringList list = memberName.split(QRegExp("/"));
 
		QString line;
		QString name;

		for (int i = 0; i < list.size()-1; i++)
		{
			QString temp = line + list.at(i) + "/";
			if (temp.size() > 30)
			{
				name += line + "\n";
				line = list.at(i) + "/";
				rows++;
			}
			else
			{
				line = temp;
			}
		}
		name += line;
		rows++;

		label->setText(name);
		label->setFixedHeight(rows*25);
	}
	else
	{
		label->setFixedHeight(30);
	}

	return rows;
}

void
OSCObjectSettings::updateProperties()
{
	bool isValid = true;

    if (object_)
    {
		QMap<QString, QWidget *>::const_iterator it;
		for (it = memberWidgets_.constBegin(); it != memberWidgets_.constEnd(); it++)
		{
			if (it.key().contains("choice"))
			{
				foreach (OpenScenario::oscMember *choiceMember, object_->getChoice())
				{
					if (choiceMember->isSelected())
					{
						isValid = isValid & loadProperties(choiceMember, it.value());
					}
				}
			}
			else
			{
				OpenScenario::oscMember *member = object_->getMember(it.key().toStdString());

				if (!member->exists())
				{
					isValid = isValid & validateMember(member);
					continue;
				} 

				isValid = isValid & loadProperties(member, it.value());
			}
		}
		if (object_->getParentObj() == oscBase_)
		{
			if (isValid)
			{
				toolManager_->setPushButtonColor(QString::fromStdString(member_->getName()), QColor(128, 195, 66));
			}
			else
			{
				toolManager_->setPushButtonColor(QString::fromStdString(member_->getName()), Qt::white);
			}
			//		}
		}
	}
	else if (oscArrayMember_)
	{
		updateTree();
	}

}

bool
OSCObjectSettings::validateMember(OpenScenario::oscMember *member)
{
	QLabel *label = ui->oscGroupBox->findChild<QLabel *>(QString::fromStdString(member->getName()));
	QPushButton *button = NULL;
	if (!label && member->isChoice() && member->isSelected())
	{
		button = ui->oscGroupBox->findChild<QPushButton *>("choice" + QString::number(member->choiceNumber()));
	}

	if (label || button)
	{
		QString color;

		switch (member->validate())
		{
		case OpenScenario::oscObjectBase::VAL_valid:
			color = "lightgreen";
			break;

		case OpenScenario::oscObjectBase::VAL_optional:
			color = "yellow";
			break;

		default:
			color = "white";
		}

		if (label)
		{
			QString s = "QLabel{ color: " + color + "; }";
			label->setStyleSheet(s);
		}
		else if (button)
		{
			QString s = "QPushButton{ color: " + color + "; }";
			button->setStyleSheet(s);
		}

		if (color == "white")
		{
			return false;
		}
	}


	return true;
}

bool
OSCObjectSettings::loadProperties(OpenScenario::oscMember *member, QWidget *widget)
{
	if (member->isChoice())
	{
		if (member->isSelected())
		{
			choiceComboBox_.value(member->choiceNumber())->setCurrentText(QString::fromStdString(member->getName()));
		}
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
	else if (QDateTimeEdit *dateTimeEdit = dynamic_cast<QDateTimeEdit *>(widget))
	{
		oscDateTimeValue *iv = dynamic_cast<oscDateTimeValue *>(value);
		if (iv)
		{
			dateTimeEdit->setDateTime(QDateTime::fromTime_t(iv->getValue()));
		}
	}

	return validateMember(member);
}

void 
OSCObjectSettings::onDeleteArrayElement()
{

	QTreeWidgetItem *item = arrayTree_->selectedItems().at(0);
	int j = arrayTree_->indexOfTopLevelItem(item);
	if (j > 0)
	{
		OpenScenario::oscObjectBase *object = oscArrayMember_->at(--j);
		RemoveOSCArrayMemberCommand *command = new RemoveOSCArrayMemberCommand(oscArrayMember_, j, base_->getOSCElement(object));
		projectSettings_->executeCommand(command);

		updateTree();
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
			{
				QDateTimeEdit *dateTimeEdit = dynamic_cast<QDateTimeEdit *>(widget);
				QDateTime v = dateTimeEdit->dateTime();
				SetOSCValuePropertiesCommand<time_t> *command = new SetOSCValuePropertiesCommand<time_t>(element_, object_, name.toStdString(), v.toTime_t());
				projectSettings_->executeCommand(command);

				break;
			}
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
	OpenScenario::oscMember *member = object_->getMember(memberName.toStdString());
	short int nr = member->choiceNumber();
	if (memberName != lastComboBoxChoice_.value(nr))
	{
		ChangeOSCObjectChoiceCommand *command = new ChangeOSCObjectChoiceCommand(object_->getChosenMember(nr), member, element_);
		projectSettings_->executeCommand(command);

		lastComboBoxChoice_[nr] = memberName;
	}


}

OpenScenario::oscObjectBase * 
OSCObjectSettings::onPushButtonPressed(QString name)
{
	OpenScenario::oscObjectBase *object = NULL;

	if (oscArrayMember_)
	{
		object = oscArrayMember_->at(name.toInt()-1);
	}
	else if (name.contains("choice"))
	{
		QComboBox *comboBox = choiceComboBox_.value(name.remove("choice").toInt());
		object = object_->getMember(comboBox->currentText().toStdString())->getOrCreateObjectBase();

	}
	else
	{
		object = object_->getMember(name.toStdString())->getOrCreateObjectBase();
	}

	OSCElement *memberElement = base_->getOrCreateOSCElement(object);

	// Reset change //
    //
    projectSettings_->getProjectData()->getChangeManager()->notifyObservers();

	OSCObjectSettings *oscSettings = new OSCObjectSettings(projectSettings_, parentStack_, memberElement, NULL, object_);

	return object;
}

OpenScenario::oscMember * 
OSCObjectSettings::onArrayPushButtonPressed(QString name)
{
	OpenScenario::oscMember *member = object_->getMember(name.toStdString());

	OSCElement *memberElement = base_->getOrCreateOSCElement(NULL);

	// Reset change //
    //
    projectSettings_->getProjectData()->getChangeManager()->notifyObservers();

	OSCObjectSettings *oscSettings = new OSCObjectSettings(projectSettings_, parentStack_, NULL, member, object_);

	return member;
}

OpenScenario::oscObjectBase * 
OSCObjectSettings::onGraphElementChosen(QString name)
{
    // Set a tool //
    //
    OpenScenarioEditorToolAction *action = new OpenScenarioEditorToolAction(ODD::TOS_GRAPHELEMENT, name);
    emit toolAction(action);
    delete action;

    OpenScenario::oscObjectBase *object = onPushButtonPressed(name);

    return object;
}

void
OSCObjectSettings::onNewArrayElement()
{
	OSCElement *oscElement = new OSCElement(memberName_);
	if (oscElement)
	{
//		OpenScenario::oscSourceFile *sourceFile = parentObject_->getSource();

		OpenScenario::oscObjectBase *obj = NULL;
		if (OSCSettings::instance()->loadDefaults())
		{
	//		obj = object_->readDefaultXMLObject( sourceFile->getSrcFileHref(), memberName_.toStdString(), object_->getMember(memberName_.toStdString())->getTypeName(), sourceFile);
		}

		AddOSCArrayMemberCommand *command = new AddOSCArrayMemberCommand(oscArrayMember_, parentObject_, obj, memberName_.toStdString(), base_, oscElement);
		projectSettings_->executeCommand(command);

		updateTree();

		OSCObjectSettings *oscSettings = new OSCObjectSettings(projectSettings_, parentStack_, oscElement, NULL, parentObject_);
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

void
OSCObjectSettings::onCloseWidget()
{

	std::string errorMessage;
//	object_->validate(&errorMessage);
	if ( !closeCount_ && (errorMessage != ""))
	{
		// Ask user //
		/*			QMessageBox::StandardButton ret = QMessageBox::warning(this, tr("ODD"),
		tr("Errors in OpenScenario elements: '%1'.\nDo you want to close anyway?")
		.arg(QString::fromStdString(errorMessage)),
		QMessageBox::Close | QMessageBox::Cancel); 

		// Close //
		//
		if (ret == QMessageBox::Close)
		parentStack_->removeWidget(); */

		formatDirLabel(objectStackTextlabel_, "! Errors in: " + objectStackText_ + " !");
		projectSettings_->printErrorMessage(objectStackText_ + ": " + QString::fromStdString(errorMessage));

		closeCount_ = true;
	}
	else
	{
		parentStack_->stackRemoveWidget();

		closeCount_ = false;
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
	else if (changes & OSCElement::COE_ChildChanged)
	{
		if (oscArrayMember_)
		{
			updateTree();
		}
	}
	else if (changes & OSCElement::COE_ChoiceChanged)
	{
/*		QWidget *lastWidget;
		do
		{
			lastWidget = parentStack_->getLastWidget();
			if (lastWidget != this)
			{
				parentStack_->stackRemoveWidget();
			}
		} while(lastWidget != this); */

		updateProperties();

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
