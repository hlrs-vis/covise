/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TUIUIWIDGETSET_H
#define TUIUIWIDGETSET_H

#include <QObject>
#include <QMap>
#include <QVariant>

class TUIUIWidgetSet : public QObject
{
    Q_OBJECT
public:
    TUIUIWidgetSet(QList<QWidget *> vectorValues, QWidget *control,
                   QList<QWidget *> views, QObject *parent = 0);

    TUIUIWidgetSet(QWidget *value, QWidget *control,
                   QList<QWidget *> views, QObject *parent = 0);

signals:
    void parameterChanged(QString moduleID, QString name, QString type, QString value, bool variantLinked);
    void evaluateScript(QString script);

public slots:

    void setValue(const QVariant &value, const QList<QVariant> &helpers);
    QVariant getValue() const;
    const QMap<QString, QVariant> &getHelpers() const
    {
        return this->helpers;
    }

private slots:
    void checkableValueChanged(bool);
    void textValueChanged(QString);
    void sliderValueChanged(int);
    void itemSelectionChanged();

    void commit();

private:
    void initValue(QWidget *valueWidget);
    void initViews();
    void initCommit();

    QList<QWidget *> valueWidgets;
    QWidget *commitWidget;
    QList<QWidget *> viewWidgets;

    QMap<QString, QVariant> helpers;

    QVariant value;
    QString type;

    bool blockCommit;
};

#endif // TUIUIWIDGETSET_H
