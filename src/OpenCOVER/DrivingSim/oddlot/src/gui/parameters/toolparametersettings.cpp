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

#include "toolparametersettings.hpp"


// GUI //
//
#include "src/gui/tools/toolmanager.hpp"
#include "toolparameter.hpp"
#include "toolvalue.hpp"

// Data //
//
#include "src/data/roadsystem/rsystemelementroad.hpp"

// Graph //
#include "src/graph/editors/projecteditor.hpp"

// Utils //
//
#include "src/util/odd.hpp"
#include "src/util/colorpalette.hpp"


// Qt //
//
#include <QFrame>
#include <QDialogButtonBox>
#include <QButtonGroup>
#include <QToolButton>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QPushButton>
#include <QCloseEvent>
#include <QComboBox>


//#######################//
// ToolParameterSettings //
//#######################//

ToolParameterSettings::ToolParameterSettings(ToolManager *toolmanager, const ODD::EditorId &editorID)
	:tool_(NULL)
	, editorID_(editorID)
	, paramList_(NULL)
	, buttonGroup_(NULL)
	, layout_(NULL)
{

	// Connect //
	//
	connect(this, SIGNAL(toolAction(ToolAction *)), toolmanager, SLOT(toolActionSlot(ToolAction *)));
}

ToolParameterSettings::~ToolParameterSettings()
{
	deleteUI();
}

void 
ToolParameterSettings::updateTool()
{
	tool_ = NULL;
	paramList_ = NULL;
}

void
ToolParameterSettings::setTool(Tool *tool)
{
	if (tool)
	{
		tool_ = tool;
		paramList_ = tool_->getParamList();
		params_ = tool_->getParams();
	}
}

void
ToolParameterSettings::addMultiSelectUI(unsigned int paramIndex, const QString &text, int count)
{
	QToolButton *button = new QToolButton();
	button->setText(text);
	button->setCheckable(true);

	QString name = QString::number(paramIndex);
	buttonGroup_->addButton(button, paramIndex); // name, id
	button->setObjectName(name);

	int row = layout_->rowCount();
	layout_->addWidget(button, row, 0);

	QLabel *label = new QLabel();
	label->setText(QString("%1  elements selected").arg(count));
	layout_->addWidget(label, row, 1);

	memberWidgets_.insert(name, label);
}

void
ToolParameterSettings::addParamUI(unsigned int paramIndex, ToolParameter *p)
{
	QString name = QString::number(paramIndex);

	if ((p->getType() == ToolParameter::ParameterTypes::OBJECT_LIST) || (p->getType() == ToolParameter::ParameterTypes::OBJECT))
	{
		QToolButton *button = new QToolButton();
		button->setText(p->getText());
		button->setCheckable(true);

		buttonGroup_->addButton(button, paramIndex); // name, id
		button->setObjectName(name);

		int row = layout_->rowCount();
		layout_->addWidget(button, row, 0);

		QLabel *label = new QLabel();
		label->setText(p->getValueDisplayed());
		layout_->addWidget(label, row, 1);

		memberWidgets_.insert(name, label);

		if (p->isActive())
		{
			button->setChecked(true);
			currentParamId_ = paramIndex;
		}
	}
	else if ((p->getType() == ToolParameter::ParameterTypes::DOUBLE) || (p->getType() == ToolParameter::ParameterTypes::INT))
	{
		QLabel *label = new QLabel(p->getText());
		layout_->addWidget(label);

		QAbstractSpinBox *spinBox;
		if (p->getType() == ToolParameter::ParameterTypes::DOUBLE)
		{
			 QDoubleSpinBox *doubleSpinBox = new QDoubleSpinBox();

			 ToolValue<double> *v = static_cast<ToolValue<double> *>(p);
			 if (v->getValue())
			 {
				 doubleSpinBox->setValue(*v->getValue());
				 p->setValid(true);
			 }

			 connect(doubleSpinBox, QOverload<const QString &>::of(&QDoubleSpinBox::valueChanged), [=](const QString &text) { onEditingFinished(name); });
			 spinBox = doubleSpinBox;
		}
		else
		{
			QSpinBox *intSpinBox = new QSpinBox();
			intSpinBox->setRange(-50, 50);

			ToolValue<int> *v = static_cast<ToolValue<int> *>(p);
			if (v->getValue())
			{
				intSpinBox->setValue(*v->getValue());
				p->setValid(true);
			}

			connect(intSpinBox, QOverload<const QString &>::of(&QSpinBox::valueChanged), [=](const QString &text) { onEditingFinished(name); });
			spinBox = intSpinBox;
		}

		spinBox->setObjectName(name);
		memberWidgets_.insert(name, spinBox);
		memberWidgets_.insertMulti(name, label);

		spinBox->installEventFilter(this);
		layout_->addWidget(spinBox);
		currentParamId_ = paramIndex;
	}
	else if (p->getType() == ToolParameter::ParameterTypes::ENUM)
	{
		QComboBox *comboBox = new QComboBox();
		comboBox->setObjectName(name);
		memberWidgets_.insert(name, comboBox);

		if (!p->getLabelText().isEmpty())
		{
			QLabel *label = new QLabel(p->getLabelText());
			layout_->addWidget(label);
			memberWidgets_.insertMulti(name, label);
		}

		// List //
		//
		QStringList roadTypeNames = p->getText().split(",");
		bool color = false;
		if (roadTypeNames.at(0) == "color")
		{
			color = true;
			roadTypeNames.removeAt(0);
		}

		// Settings //
		//
		comboBox->addItems(roadTypeNames);
		ToolValue<int> *v = static_cast<ToolValue<int> *>(p);
		if (v->getValue())
		{
			comboBox->setCurrentIndex(*v->getValue());
			p->setValid(true);
		}

		if (color)
		{
			QList<QColor> colors;
			colors
				<< ODD::instance()->colors()->brightGrey()
				<< ODD::instance()->colors()->brightOrange()
				<< ODD::instance()->colors()->brightRed()
				<< ODD::instance()->colors()->brightGreen()
				<< ODD::instance()->colors()->brightCyan()
				<< ODD::instance()->colors()->brightBlue();

			for (int i = 0; i < roadTypeNames.count(); ++i)
			{
				QPixmap icon(12, 12);
				icon.fill(colors.at(i));
				comboBox->setItemIcon(i, icon);
			}
		}

		connect(comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(handleComboBoxSelection(int)));
		layout_->addWidget(comboBox);
		currentParamId_ = paramIndex;
	}
}

void
ToolParameterSettings::setComboBoxIndex(ToolParameter *p, const QString &text)
{
	QString name = QString::number(params_->key(p));
	QList<QWidget *> widgetList = memberWidgets_.values(name);
	for (int i = 0; i < widgetList.size();)
	{
		QWidget *widget = widgetList.takeAt(i);
		QComboBox *comboBox = dynamic_cast<QComboBox *>(widget);
		if (comboBox)
		{
			int index = comboBox->findText(text);
			comboBox->setCurrentIndex(index);
			return;
		}
	}
}

void 
ToolParameterSettings::addComboBoxEntry(ToolParameter *p, int index, const QString &text)
{
	QString name = QString::number(params_->key(p));
	QList<QWidget *> widgetList = memberWidgets_.values(name);
	for (int i = 0; i < widgetList.size();)
	{
		QWidget *widget = widgetList.takeAt(i);
		QComboBox *comboBox = dynamic_cast<QComboBox *>(widget);
		if (comboBox)
		{
			comboBox->blockSignals(true);
			comboBox->insertItem(index, text);
			comboBox->blockSignals(false);
			comboBox->setCurrentIndex(index);
			return;
		}
	}
}

void
ToolParameterSettings::updateSpinBoxAndLabels(ToolParameter *p)
{
	int paramIndex = params_->key(p);
	QString name = QString::number(paramIndex);
	QList<QWidget *> widgetList = memberWidgets_.values(name);
	for (int i = 0; i < widgetList.size(); i++)
	{
		QAbstractSpinBox *spinBox = dynamic_cast<QAbstractSpinBox *>(widgetList.at(i));
		if (spinBox)
		{
			if (p->getType() == ToolParameter::INT)
			{
				ToolValue<int> *v = dynamic_cast<ToolValue<int> *>(p);
				QSpinBox *spin = dynamic_cast<QSpinBox *>(spinBox);
				spin->blockSignals(true);
				spin->setValue(*v->getValue());
				spin->blockSignals(false);
			}
		}
	}
}

void
ToolParameterSettings::deleteButtonsAndLabels(QAbstractButton *button)
{
	QString name = button->objectName();
	QLabel *label = dynamic_cast<QLabel*>(memberWidgets_.value(name));
	memberWidgets_.remove(name);

	buttonGroup_->removeButton(button);
	layout_->removeWidget(button);
	layout_->removeWidget(label);
	delete button;
	delete label;
}

void
ToolParameterSettings::deleteSpinBoxAndLabels(int paramIndex)
{
	QString name = QString::number(paramIndex);
	QList<QWidget *> widgetList = memberWidgets_.values(name);
	for (int i = 0; i < widgetList.size();)
	{
		QWidget *widget = widgetList.takeAt(i);
		layout_->removeWidget(widget);
		delete widget;
	}
	memberWidgets_.remove(name);
}

void
ToolParameterSettings::deleteCombobox(int paramIndex)
{
	QString name = QString::number(paramIndex);
	QList<QWidget *> widgetList = memberWidgets_.values(name);
	for (int i = 0; i < widgetList.size();)
	{
		QWidget *widget = widgetList.takeAt(i);
		layout_->removeWidget(widget);
		delete widget;
	}

	memberWidgets_.remove(name);
}

void
ToolParameterSettings::removeUI(unsigned int paramIndex)
{
	if (paramList_->contains(paramIndex))
	{
		ToolParameter *p = tool_->getLastParam(paramIndex);
		int objectCount = tool_->getObjectCount(p->getToolId(), p->getParamToolId());
		setLabels(paramIndex, QString("%1 elements selected").arg(objectCount), buttonGroup_->button(paramIndex)->text());
	}
	else
	{
		QList<QAbstractButton *> buttonList = buttonGroup_->buttons();
		if (paramList_->size() + params_->size() > memberWidgets_.uniqueKeys().size())
		{
			for (int i = 0; i < buttonList.size();)
			{
				QAbstractButton *button = buttonList.takeAt(i);
				deleteButtonsAndLabels(button);
			}
			QMap<QString, QWidget *>::iterator it = memberWidgets_.begin();
			while (it != memberWidgets_.end())
			{
				QWidget *widget = it.value();
				layout_->removeWidget(widget);
				delete widget;
				it++;
			}
			memberWidgets_.clear();

			QMap<unsigned int, ToolParameter *>::const_iterator paramIt = params_->constBegin();
			while (paramIt != params_->constEnd())
			{
				addParamUI(paramIt.key(), paramIt.value());
				paramIt++;
			}
		}
		else
		{
			deleteButtonsAndLabels(buttonGroup_->button(paramIndex));
		}
	}
}

void
ToolParameterSettings::addUI(unsigned int paramIndex, ToolParameter *p)
{
	if (p->getType() == ToolParameter::ParameterTypes::OBJECT_LIST)
	{
		int objectCount = tool_->getObjectCount(p->getToolId(), p->getParamToolId());
		if (objectCount == tool_->getListSize())
		{
			QList<QAbstractButton *> buttonList = buttonGroup_->buttons();
			for (int i = 0; i < buttonList.size();)
			{
				QAbstractButton *button = buttonList.at(i);
				ToolParameter *param = tool_->getLastParam(button->objectName().toInt());
				if (!param || (param->getToolId() == p->getToolId()))
				{
					buttonList.removeAt(i);
					deleteButtonsAndLabels(button);
				}
				else
				{
					i++;
				}
			}

			QMap<unsigned int, QList<ToolParameter *>>::const_iterator it = paramList_->constBegin();
			while (it != paramList_->constEnd())
			{
				addMultiSelectUI(it.key(), "Remove Object", it.value().size());
				it++;
			}

			QMap<unsigned int, ToolParameter *>::const_iterator paramIt = params_->constEnd();
			do
			{
				paramIt--;
				if (paramIt.value()->getToolId() == p->getToolId())
				{
					addParamUI(paramIndex, p);
					break;
				}
			} while (paramIt != params_->constBegin());

		}
		else if (objectCount > tool_->getListSize())
		{
			paramList_ = tool_->getParamList();
			QMap<unsigned int, QList<ToolParameter *>>::const_iterator it = paramList_->constBegin();
			while (it != paramList_->constEnd())
			{
				if (it.value().size() > 1)
				{
					unsigned int i = it.key();
					setLabels(i, QString("%1 elements selected").arg(objectCount), buttonGroup_->button(i)->text());
					break;
				}
				it++;
			}
		}
		else
		{
			addParamUI(paramIndex, p);
		}
	}
	else
	{
		addParamUI(paramIndex, p);
	}
}



void
ToolParameterSettings::generateUI(QFrame *box)
{
	layout_ = new QGridLayout;
	layout_->setColumnStretch(1, 10);

	buttonGroup_ = new QButtonGroup();
	QMap<unsigned int, ToolParameter *>::const_iterator paramIt = params_->constBegin();
	while (paramIt != params_->constEnd())
	{
		addUI(paramIt.key(), paramIt.value());
		paramIt++;
	}

	QMap<unsigned int, QList<ToolParameter *>>::const_iterator it = paramList_->constBegin();
	while (it != paramList_->constEnd())
	{
		addUI(it.key(), it.value().first());
		it++;
	}

	if (buttonGroup_->buttons().size() < 2)
	{
		buttonGroup_->setExclusive(false);
	}
	connect(buttonGroup_, SIGNAL(buttonPressed(int)), this, SLOT(onButtonPressed(int)));

	int checkedButton = buttonGroup_->checkedId(); // now the checked button can send an action
	if (checkedButton != -1)
	{
		onButtonPressed(checkedButton);
	}

	box->setLayout(layout_);
}

void
ToolParameterSettings::updateUI(ToolParameter *param)
{
	if ((param->getType() == ToolParameter::DOUBLE) || (param->getType() == ToolParameter::INT))
	{
		updateSpinBoxAndLabels(param);
	}
}

void
ToolParameterSettings::deleteUI()
{
	if (params_)
	{
		QMap<unsigned int, ToolParameter *>::const_iterator paramIt = params_->constBegin();
		while (paramIt != params_->constEnd())
		{
			if ((paramIt.value()->getType() == ToolParameter::OBJECT) || (paramIt.value()->getType() == ToolParameter::OBJECT_LIST))
			{
				deleteButtonsAndLabels(buttonGroup_->button(paramIt.key()));
			}
			else if ((paramIt.value()->getType() == ToolParameter::DOUBLE) || (paramIt.value()->getType() == ToolParameter::INT))
			{
				deleteSpinBoxAndLabels(paramIt.key());
			}
			else if (paramIt.value()->getType() == ToolParameter::ENUM)
			{
				deleteCombobox(paramIt.key());
			}
			paramIt++;
		}
	}
	if (paramList_)
	{
		QMap<unsigned int, QList<ToolParameter *>>::const_iterator it = paramList_->constBegin();
		while (it != paramList_->constEnd())
		{
			deleteButtonsAndLabels(buttonGroup_->button(it.key()));
			it++;
		}
	}

	if (buttonGroup_)
	{
		delete buttonGroup_;
	}

	if (layout_)
	{
		delete layout_;
	}

}

bool 
ToolParameterSettings::eventFilter(QObject* object, QEvent* event)
{
	for (int i = 0; i < memberWidgets_.values().size();)
	{
		QWidget* widget = memberWidgets_.values().takeAt(i);
		if (object == widget) {
			if (event->type() == QEvent::FocusIn) {
				QAbstractButton *checkedButton = buttonGroup_->checkedButton();
				if (checkedButton)
				{
					checkedButton->setChecked(false);
				}
				return true;
			}
			else {
				return false;
			}
		}
		else
		{
			i++;
		}
	}

	// pass the event on to the parent class
	return QObject::eventFilter(object, event);
}

void
ToolParameterSettings::onEditingFinished(const QString &objectName)
{

	int i = objectName.toInt();
	ToolParameter *p;
	if (params_->contains(i))
	{
		p = params_->value(i);
	}
/*	else
	{
		p = paramList_->value(i).first();
	} */

	if (p)
	{
		double val;
		if (p->getType() == ToolParameter::ParameterTypes::DOUBLE)
		{
			ToolValue<double> *v = dynamic_cast<ToolValue<double> *>(p);
			foreach(QWidget *widget, memberWidgets_.values(objectName))
			{
				QDoubleSpinBox *spinBox = dynamic_cast<QDoubleSpinBox*>(widget);
				if (spinBox)
				{
					val = spinBox->value();
					break;
				}
			}

			v->setValue(val);
		}
		else if (p->getType() == ToolParameter::ParameterTypes::INT)
		{
			ToolValue<int> *v = dynamic_cast<ToolValue<int> *>(p);
			foreach(QWidget *widget, memberWidgets_.values(objectName))
			{
				QSpinBox *spinBox = dynamic_cast<QSpinBox*>(widget);
				if (spinBox)
				{
					val = spinBox->value();
					break;
				}
			}

			v->setValue(val);
		}

		ODD::ToolId paramToolId = p->getParamToolId();
		if (paramToolId == ODD::TPARAM_VALUE)
		{
			ParameterToolAction *action = new ParameterToolAction(editorID_, p->getToolId(), paramToolId, val, true);
			emit toolAction(action);
			delete action;
		}
		else
		{
			ParameterToolAction *action = new ParameterToolAction(editorID_, p->getToolId(), paramToolId, i, true);
			emit toolAction(action);
			delete action;
		}
	}
}

void
ToolParameterSettings::handleComboBoxSelection(int index)
{
	QComboBox *combo = dynamic_cast<QComboBox *>(sender());
	currentParamId_ = memberWidgets_.key(combo).toInt();
	ToolParameter *p = params_->value(currentParamId_);

	ParameterToolAction *action = new ParameterToolAction(editorID_, p->getToolId(), p->getParamToolId(), index, false);
	emit toolAction(action);
	delete action;
}

void
ToolParameterSettings::onButtonPressed(int paramId)
{
	QAbstractButton *button = buttonGroup_->button(paramId);
	ToolParameter *p;
	if (params_->contains(paramId))
	{
		p = params_->value(paramId);
	}
	else
	{
		p = paramList_->value(paramId).first();
	}

	if ((p->getType() == ToolParameter::ParameterTypes::OBJECT_LIST) || (p->getType() == ToolParameter::ParameterTypes::OBJECT))
	{
		int oldParamId = currentParamId_;
		currentParamId_ = paramId;
		if (button->text() == "Remove Object")
		{
			QString name = button->objectName();

			ParameterToolAction *action = new ParameterToolAction(editorID_, p->getToolId(), p->getParamToolId(), paramId, false);
			emit toolAction(action);
			delete action;

			removeUI(paramId);

			currentParamId_ = oldParamId;
		}
		else 
		{
			ParameterToolAction *action = new ParameterToolAction(editorID_, p->getToolId(), p->getParamToolId(), paramId, true);
			emit toolAction(action);
			delete action;
		}

	}

}

void 
ToolParameterSettings::setLables(QList<ToolParameter*>& paramList)
{
	foreach(ToolParameter *param, paramList)
	{
		unsigned int id = tool_->getParamId(param);
		QAbstractButton* button = buttonGroup_->button(id);
		button->setText(param->getText());
		if (param->isActive())
		{
			button->setChecked(true);
			currentParamId_ = id;
		}

		QString name = QString::number(id);
		QLabel* label = dynamic_cast<QLabel*>(memberWidgets_.value(name));
		label->setText(param->getValueDisplayed());
	}
}

void 
ToolParameterSettings::setLabels(int id, const QString &objectName, const QString &buttonText)
{
	QAbstractButton *button = buttonGroup_->button(id);
	//	buttonGroup_->setExclusive(false);
	//button->setChecked(false);
	button->setText(buttonText);
	//	buttonGroup_->setExclusive(true);

	QString name = QString::number(id);
	QLabel *label = dynamic_cast<QLabel*>(memberWidgets_.value(name));
	label->setText(objectName);
}

void
ToolParameterSettings::setObjectSelected(int id, const QString &objectName, const QString &buttonText)
{
	currentParamId_ = id;

	setLabels(id, objectName, buttonText);

	if (!buttonGroup_->exclusive() && buttonGroup_->checkedButton())
	{
		buttonGroup_->checkedButton()->setChecked(false);
	}
}

void 
ToolParameterSettings::hide()
{
	ParameterToolAction *action = new ParameterToolAction(editorID_, ODD::TNO_TOOL, ODD::TNO_TOOL, 0, false);
	emit toolAction(action);
	delete action;
}



//####################################//
// ToolParameterSettingsDialogBox    //
//##################################//

ToolParameterSettingsApplyBox::ToolParameterSettingsApplyBox(ProjectEditor *editor, ToolManager *toolManager, const ODD::EditorId &editorID, QFrame *dBox)
	:ToolParameterSettings(toolManager, editorID),
	editor_(editor)
{

	createDialogBox(dBox);
}

ToolParameterSettingsApplyBox::~ToolParameterSettingsApplyBox()
{
	deleteDialogBox();
}



QDialogButtonBox *
ToolParameterSettingsApplyBox::createDialogBox(QFrame *dBox)
{
	dialogLayout_ = new QGridLayout;
	dialogBox_ = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Apply| QDialogButtonBox::Cancel);
	setApplyButtonVisible(false);
	ok_ = dialogBox_->button(dialogBox_->Ok);
	ok_->setObjectName("okButton");
	ok_->setDefault(true);
	dialogLayout_->addWidget(dialogBox_, 0, 0, Qt::AlignBottom);
	dBox->setLayout(dialogLayout_);
	dBox->show();

	connect(dialogBox_->button(dialogBox_->Apply), SIGNAL(pressed()), this, SLOT(apply()));
	connect(dialogBox_, SIGNAL(accepted()), this, SLOT(ok()));
	connect(dialogBox_, SIGNAL(rejected()), this, SLOT(cancel()));

	return dialogBox_;
}

void
ToolParameterSettingsApplyBox::deleteDialogBox()
{
	delete dialogBox_;
	dialogBox_ = NULL;
	delete dialogLayout_;
}

void 
ToolParameterSettingsApplyBox::generateUI(QFrame *box)
{
	ToolParameterSettings::generateUI(box);
	dialogBox_->button(dialogBox_->Apply)->setVisible(tool_->verify());
	dialogBox_->button(dialogBox_->Ok)->setVisible(tool_->verify());
}

void
ToolParameterSettingsApplyBox::deleteUI()
{
	ToolParameterSettings::deleteUI();
	setApplyButtonVisible(false);
}

void
ToolParameterSettingsApplyBox::setApplyButtonVisible(bool visible)
{
	dialogBox_->button(dialogBox_->Apply)->setVisible(visible);
	dialogBox_->button(dialogBox_->Ok)->setVisible(visible);
	if (visible)
	{
		dialogBox_->button(dialogBox_->Ok)->setDefault(true);
	}
	else
	{
		dialogBox_->button(dialogBox_->Cancel)->setDefault(true);
	}
}

void ToolParameterSettingsApplyBox::onEditingFinished(const QString &objectName)
{
	ToolParameterSettings::onEditingFinished(objectName);
	if (tool_->verify())
	{
		dialogBox_->button(dialogBox_->Apply)->setVisible(true);
	}
	else
	{
		dialogBox_->button(dialogBox_->Apply)->setVisible(false);
	}
}

void
ToolParameterSettingsApplyBox::cancel()
{
	editor_->reject();
}

void
ToolParameterSettingsApplyBox::apply()
{

	editor_->apply();
}

void
ToolParameterSettingsApplyBox::ok()
{
	apply();
	cancel();
}

void 
ToolParameterSettingsApplyBox::enterEvent(QEvent* event)
{
	ok_->setFocus(Qt::TabFocusReason);
}

void 
ToolParameterSettingsApplyBox::focus(short state)
{
	if (state)
	{
		ok_->setFocus();
	}
	else
	{
		ok_->clearFocus();
	}
}




//######################//
//                     //
// ParameterToolAction //
//                    //
//####################//

ParameterToolAction::ParameterToolAction(const ODD::EditorId &editorID, ODD::ToolId toolId, ODD::ToolId paramToolId)
	:ToolAction(editorID, toolId, paramToolId)
{
}

ParameterToolAction::ParameterToolAction(const ODD::EditorId &editorID, ODD::ToolId toolId, ODD::ToolId paramToolId, int paramId, bool state)
	:ToolAction(editorID, toolId, paramToolId),
	state_(state),
	paramId_(paramId)
{
}

ParameterToolAction::ParameterToolAction(const ODD::EditorId &editorID, ODD::ToolId toolId, ODD::ToolId paramToolId, double value, bool state)
	:ToolAction(editorID, toolId, paramToolId),
	state_(state),
	value_(value)
{
}

