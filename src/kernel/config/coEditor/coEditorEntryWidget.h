/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ENTRYWIDGET
#define ENTRYWIDGET

#include <QWidget>
#include <config/coConfigEntry.h>
//the observer
// #include "config/coConfigEditorEntry.h"

// class QListWidget;
class QVBoxLayout;
class QSignalMapper;
class coEditorGroupWidget;
class QGroupBox;

class coEditorEntryWidget : public QWidget
{

    Q_OBJECT
    //    Q_PROPERTY (bool infoWidget);

public:
    coEditorEntryWidget(QWidget *parent, coEditorGroupWidget *group,
                        covise::coConfigEntry *entry, const QString &name = QString::null);
    coEditorEntryWidget(QWidget *parent, coEditorGroupWidget *group,
                        covise::coConfigSchemaInfos *infos, const QString &name = QString::null);

    ~coEditorEntryWidget()
    {
    }

    covise::coConfigEntry *getEntry();
    covise::coConfigSchemaInfos *getSchemaInfo();

signals:
    void helpRequest(/*QString & text*/);
    void saveValue(const QString &variable, const QString &value,
                   const QString &section, const QString &host = QString::null);
    void deleteValue(const QString &variable, const QString &section);
    void nameAdded(const QString &newName, covise::coConfigSchemaInfos *infos);
    void deleteRequest(coEditorEntryWidget *widget);
    void nameDeleted(coEditorEntryWidget *widget);

public slots:
    void refresh(covise::coConfigEntry *subject);

private slots:

    void saveValue(const QString &variable, const QString &value);
    void nameAddedSlot(const QString &value);
    void deleteValue(const QString &variable);
    void moveToHost();
    void suicide();
    void explainShowInfoButton();

private:
    void createBooleanValue(const QString &valueName, const QString &value,
                            const QString &attributeDescription = QString::null,
                            bool empty = false, bool required = false);
    void createQregXpValue(const QString &valueName, const QString &value,
                           const QRegExp &rx = QRegExp("^.*"),
                           const QString &readableAttrRule = QString::null,
                           const QString &attributeDescription = QString::null,
                           bool empty = false, bool required = false);

    void createConstruct();
    void examineEntry();
    void tranformToRealEntry();
    bool setName(const QString &name); // set objectName if possible

    covise::coConfigEntry *rootEntry;
    covise::coConfigSchemaInfos *info;
    coEditorGroupWidget *groupWidget;
    QGroupBox *valuesOfEntryGroup;
    bool singleEntry; // true, when this entry has only one attribute "value"
    QString section;

protected:
    QAction *deleteAction;
    QAction *moveToHostAction;
    QVBoxLayout *entryWidgetLayout;
    QStringList valuesList; // holds all attributes that we have an coEditorValueWidget for
    QString entryWidgetName;
    bool modified;
    QSignalMapper *signalMapper;
};
#endif
