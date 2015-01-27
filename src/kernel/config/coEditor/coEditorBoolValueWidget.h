/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef BOOLVALUEWIDGET
#define BOOLVALUEWIDGET
#include "coEditorValueWidget.h"
#include <QString>
#include <QRegExp>

class QCheckBox;

class coEditorBoolValueWidget : public coEditorValueWidget
{
    Q_OBJECT
public:
    coEditorBoolValueWidget(QWidget *parent, const QString name, Type type);
    ~coEditorBoolValueWidget();

signals:
    void saveValue(const QString &variable, const QString &value);
    void deleteValue(const QString &variable);

public slots:
    void setValue(const QString &valueName, const QString &value,
                  const QString &readableAttrRule = QString::null,
                  const QString &attributeDescription = QString::null,
                  bool required = false, const QRegExp &rx = QRegExp("^.*"));
    void suicide();
    void undo();

private slots:
    void save(int state);

private:
    QString widgetName, variable;
    QAction *deleteValueAction;
    QCheckBox *aValuesCheckBox;
};
#endif
