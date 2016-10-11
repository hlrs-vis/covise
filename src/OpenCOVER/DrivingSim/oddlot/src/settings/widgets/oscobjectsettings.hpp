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

#ifndef OSCOBJECTSETTINGS_HPP
#define OSCOBJECTSETTINGS_HPP

#include <QWidget>
#include "src/data/observer.hpp"
#include "src/util/droparea.hpp"

#include "oscMemberValue.h"


namespace Ui
{
class OSCObjectSettings;
}

namespace OpenScenario
{
class oscObjectBase;
class oscArrayMember;
class oscMember;
}

class OSCElement;
class OSCBase;
class OpenScenarioEditorToolAction;
class OSCObjectSettingsStack;
class ProjectSettings;

class QLabel;
class QTreeWidget;
class QTreeWidgetItem;
class QSignalMapper;
class QComboBox;

#include <QMap>


class OSCObjectSettings: public QWidget, public Observer
{
    Q_OBJECT

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit OSCObjectSettings(ProjectSettings *projectSettings, OSCObjectSettingsStack *parent, OSCElement *element);
    virtual ~OSCObjectSettings();

    // Observer Pattern //
    //
    virtual void updateObserver();

	// Initialize generic user interface //
	//
	void uiInit();
	void uiInitArray();
	void onDeleteArrayElement();

	OpenScenario::oscArrayMember *getArrayMember()
	{
		return oscArrayMember_;
	}

	void updateTree();

private:
    void updateProperties();
	void loadProperties(OpenScenario::oscMember *member, QWidget *widget);
	void formatLabel(QLabel *label, const QString &memberName);
	int formatDirLabel(QLabel *label, const QString &memberName);
	void addTreeItem(QTreeWidget *arrayTree, int name);
	QString getStackText()
	{
		return objectStackText_;
	}

	//################//
	// SIGNALS        //
	//################//

signals:

    //################//
    // SLOTS          //
    //################//

private slots:
    void onEditingFinished(QString name);
	OpenScenario::oscObjectBase * onPushButtonPressed(QString name);
	void onArrayElementClicked(QTreeWidgetItem *item, int column);
	void onNewArrayElement();
	void onValueChanged();
	void onChoiceChanged(const QString &text);
	void onCloseWidget();

    //################//
    // PROPERTIES     //
    //################//

private:
	Ui::OSCObjectSettings *ui;
	ProjectSettings *projectSettings_;
	OSCObjectSettingsStack *parentStack_;

    OpenScenario::oscObjectBase *object_;
	QString objectStackText_;
	QLabel *objectStackTextlabel_;

	OpenScenario::oscArrayMember *oscArrayMember_;
	QTreeWidget *arrayTree_;
	QComboBox *choiceComboBox_;
	OSCElement *element_;
	OSCBase *base_;

    bool init_;

	QMap<QString, QWidget*> memberWidgets_;
	QString memberName_;
	QString lastComboBoxChoice_;

    bool valueChanged_;

	bool closeCount_;

};

class ArrayDropArea : public DropArea
{
	//################//
    // FUNCTIONS      //
    //################//

public:
	explicit ArrayDropArea(OSCObjectSettings *settings, QPixmap *pixmap);

private:
    ArrayDropArea(); /* not allowed */
    ArrayDropArea(const OSCObjectSettings &, QPixmap *pixmap); /* not allowed */
    ArrayDropArea &operator=(const OSCObjectSettings &); /* not allowed */

	//################//
    // SLOTS          //
    //################//
protected:
    void dropEvent(QDropEvent *event);

private:
	OSCObjectSettings *settings_;
};


#endif // OSCOBJECTSETTINGS_HPP
