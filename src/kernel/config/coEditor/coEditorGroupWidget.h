/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef GROUPWIDGET
#define GROUPWIDGET

#include <QWidget>
#include <QList>
#include <QHash>

//the observer template
#include <config/coConfigEditorController.h>
#include <config/coConfigSchemaInfos.h>
#include <config/coConfigSchemaInfosList.h>
#include <config/coConfigEntry.h>

class QVBoxLayout;
class QLineEdit;
class coEditorEntryWidget;

namespace covise
{
class coConfigEntry;
class coConfigSchemaInfos;
class coConfigSchemaInfosList;
}

class coEditorGroupWidget : public QWidget, covise::Observer<covise::coConfigEntry>
{
    Q_OBJECT
public:
    coEditorGroupWidget(QWidget *parent, const QString &name,
                        covise::coConfigEntry *entry = 0, covise::coConfigSchemaInfosList *infoList = 0);
    ~coEditorGroupWidget()
    {
    }

    void addEntries(QHash<QString, covise::coConfigEntry *> entriesList, bool overwrite = false);
    void update(covise::coConfigEntry *subject); //observer Part
    bool outOfDate();

    void setOutOfDate();
    //check if this name is free or ocupied
    bool nameIsFree(const QString &newName, coEditorEntryWidget *entryWid);
    void removeFromInfoWidgetList(covise::coConfigSchemaInfos *info);

private:
    void addEntry(const QString name, covise::coConfigEntry *entry, bool overwrite = false);
    void createInfoWidgets();
    void createInfoWidget(covise::coConfigSchemaInfos *info);

public slots:
    void showInfoWidget(bool show = 0);
    // create new InfoWidget because for the old one "name" has been set.
    // called by coEditorEntryWidget::nameadded
    void updateInfoWidgets(const QString &newName, covise::coConfigSchemaInfos *info);
    // delete a coEditorEntryWidget
    void deleteRequest(coEditorEntryWidget *widget);

private slots:
    void saveValueEntry(const QString &variable,
                        const QString &value, const QString &section,
                        const QString &host = QString::null);
    void deleteValueEntry(const QString &variable, const QString &section);

signals:
    void saveValue(const QString &variable, const QString &value,
                   const QString &section, const QString &targetHost = QString::null);
    void deleteValue(const QString &variable, const QString &section,
                     const QString &targetHost = QString::null);
    void showStatusBar(const QString &message, int timeout = 0);
    void hideYourselfInfoWidgets();
    void showYourselfInfoWidgets();

private:
    QWidget *mainWin;
    QString groupName;

    bool fOutOfDate, addedInfoWidgets;
    QVBoxLayout *groupGroupLayout;
    QWidget *groupEntryWidget;
    covise::coConfigEntry *rootEntry;
    QHash<QString, covise::coConfigEntry *> entries; // holds all entries for that group that are presented by a widget
    QStringList infoWidgetList; // holds names of elements for which a infoWidget is created.
    covise::coConfigSchemaInfosList *groupList;
};
#endif
