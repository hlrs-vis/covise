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

#ifndef TOOLPARAMETERSETTINGS_HPP
#define TOOLPARAMETERSETTINGS_HPP

#include "tool.hpp"
#include "src/gui/tools/toolaction.hpp"
#include <QObject>

class ToolParameter;
class ToolManager;

class QFrame;
class QButtonGroup;
class QGridLayout;
class QAbstractButton;
class QAbstractSpinBox;
class QDialogButtonBox;
class QHideEvent;

class ToolParameterSettings : public QObject
{
	Q_OBJECT

public:
	explicit ToolParameterSettings(ToolManager *toolmanager, const ODD::EditorId &editorID);
	virtual ~ToolParameterSettings();

	void updateTool();
	void setTool(Tool *tool);
	virtual void generateUI(QFrame *box);
	virtual void updateUI(ToolParameter *param);
	void addMultiSelectUI(unsigned int paramIndex, const QString &text, int count);
	void addUI(unsigned int paramIndex, ToolParameter *p, bool active = false);
	void addParamUI(unsigned int paramIndex, ToolParameter *p, bool active = false);
	void addComboBoxEntry(ToolParameter *p, int index, const QString &text);
	void setComboBoxIndex(ToolParameter *p, const QString &text);
	void updateSpinBoxAndLabels(ToolParameter *p);
	void deleteButtonsAndLabels(QAbstractButton *button);
	void deleteSpinBoxAndLabels(int paramIndex);
	void deleteCombobox(int paramIndex);
	void removeUI(unsigned int paramIndex);
	virtual void deleteUI();

	void hide();

	QList<ToolParameter *> getCurrentParameter()
	{
		return paramList_->value(currentParamId_);
	}

	int getCurrentParameterID()
	{
		return currentParamId_;
	}


	void setLabels(int id, const QString &objectName, const QString &buttonText);
	void setObjectSelected(int id, const QString &objectName, const QString &buttonText);

	//################//
	// SIGNALS        //
	//################//

signals:
	void toolAction(ToolAction *);

protected slots:
	void onEditingFinished(const QString &objectName);
	void handleComboBoxSelection(int);

private slots:
	void onButtonPressed(int toolId);

private:
	ToolParameterSettings(); /* not allowed */
	ToolParameterSettings(const ToolParameterSettings &); /* not allowed */
	ToolParameterSettings &operator=(const ToolParameterSettings &); /* not allowed */

protected:
	Tool *tool_;
	ODD::EditorId editorID_;

private:

	QMap<unsigned int, QList<ToolParameter *>> *paramList_;
	QMap<unsigned int, ToolParameter *> *params_;
	QMap<QString, QWidget *> memberWidgets_;
	int currentParamId_;

	QButtonGroup *buttonGroup_;
	QGridLayout *layout_;
};

class ToolParameterSettingsApplyBox : public ToolParameterSettings
{
	Q_OBJECT

		//################//
		// FUNCTIONS      //
		//################//
public:
	explicit ToolParameterSettingsApplyBox(ProjectEditor *editor, ToolManager *toolmanager, const ODD::EditorId &editorID, QFrame *dBox);
	virtual ~ToolParameterSettingsApplyBox();

	virtual void generateUI(QFrame *box);
	virtual void deleteUI();
	QDialogButtonBox *createDialogBox(QFrame *box);
	void deleteDialogBox();
	void setApplyButtonVisible(bool);

//################//
// SIGNALS        //
//################//
private slots:
	void onEditingFinished(const QString &objectName);
	void apply();
	void reset();

private:
	ToolParameterSettingsApplyBox(); /* not allowed */
	ToolParameterSettingsApplyBox(const ToolParameterSettingsApplyBox &); /* not allowed */
	ToolParameterSettingsApplyBox &operator=(const ToolParameterSettingsApplyBox &); /* not allowed */

private:
	ProjectEditor *editor_;

	QDialogButtonBox *dialogBox_;
	QGridLayout *dialogLayout_;

};


class ParameterToolAction : public ToolAction
{

	//################//
	// FUNCTIONS      //
	//################//

public:
	explicit ParameterToolAction(const ODD::EditorId &editorID, ODD::ToolId toolId, ODD::ToolId paramToolId);
	explicit ParameterToolAction(const ODD::EditorId &editorID, ODD::ToolId toolId, ODD::ToolId paramToolId, int paramId, bool state);
	explicit ParameterToolAction(const ODD::EditorId &editorID, ODD::ToolId toolId, ODD::ToolId paramToolId, double value, bool state);

	virtual ~ParameterToolAction()
	{ /* does nothing */
	}

	bool getState()
	{
		return state_;
	}

	int getParamId()
	{
		return paramId_;
	}

	double getValue()
	{
		return value_;
	}

protected:
private:
	ParameterToolAction(); /* not allowed */
	ParameterToolAction(const ParameterToolAction &); /* not allowed */
	ParameterToolAction &operator=(const ParameterToolAction &); /* not allowed */


protected:
private:
	bool state_;
	int paramId_;
	double value_;
};

#endif // TOOLPARAMETERSETTINGS_HPP
