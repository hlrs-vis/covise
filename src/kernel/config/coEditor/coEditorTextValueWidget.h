/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TEXTVALUEWIDGET
#define TEXTVALUEWIDGET
#include "coEditorValueWidget.h"
#include "coEditorValidatedQLineEdit.h"
#include <QString>
#include <QRegExp>

class coEditorTextValueWidget : public coEditorValueWidget
{
    Q_OBJECT
public:
    coEditorTextValueWidget(QWidget *parent, const QString name, Type type);
    ~coEditorTextValueWidget();

signals:
    void saveValue(const QString &variable, const QString &value);
    void deleteValue(const QString &variable);
    void notValid();
    void nameAdded(const QString &value);

public slots:
    void setValue(const QString &valueName, const QString &value,
                  const QString &readableAttrRule = QString::null,
                  const QString &attributeDescription = QString::null,
                  bool required = false, const QRegExp &rx = QRegExp("^.*"));
    void suicide();
    void undo(); // relay to QlineEdit

private slots:
    void save();

    // virtual setValue(const QString & variable, const QString & value, const QString & section);

private:
    QString widgetName, variable, defaultValue;
    coEditorValidatedQLineEdit *valueLineEdit;
    QAction *deleteValueAction;
};
#endif
