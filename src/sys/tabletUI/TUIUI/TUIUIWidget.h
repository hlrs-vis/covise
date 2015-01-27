/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TUIUIWIDGET_H
#define TUIUIWIDGET_H
#include <util/coTypes.h>
#include <QWidget>
#include <QMap>

class QDomNode;
class TUIUIWidgetSet;
class TUIUITab;
class TUIUIScriptWidget;

class TUIUIWidget : public QWidget
{
    Q_OBJECT
public:
    explicit TUIUIWidget(const QString &description, TUIUITab *parent);
    virtual ~TUIUIWidget();

    virtual void addModule(const QDomNode &moduleDescription);
    virtual void setParameter(const QString &module, const QString &parameter,
                              const QVariant &value, const QList<QVariant> &helpers);

    virtual void setParameter(const QString &module, const QString &parameter,
                              const QString &value, bool variantLinked);

public slots:
    void boolParameterChanged(const QString &module, const QString &parameter, bool value, bool variantLinked = false);
    void intParameterChanged(const QString &module, const QString &parameter, int value, bool variantLinked = false);
    void floatParameterChanged(const QString &module, const QString &parameter, float value, bool variantLinked = false);
    void intBoundedParameterChanged(const QString &module, const QString &parameter,
                                    int value, int minimum, int maximum, bool variantLinked = false);
    void floatBoundedParameterChanged(const QString &module, const QString &parameter,
                                      float value, float minimum, float maximum, bool variantLinked = false);
    void stringParameterChanged(const QString &module, const QString &parameter, const QString &value, bool variantLinked = false);

    void processMessage(const QString &message);

signals:
    void command(const QString &target, const QString &command);

private slots:
    void onWidgetSetParameterChange(QString moduleID, QString name, QString type,
                                    QString value, bool linked);
    void onEvaluateScript(QString script);

    void scriptWidgetActivated();

private:
    QString uiFilename;
    QWidget *moduleWidget;
    TUIUITab *parentTab;

    QString mapfile;
    QString mapname;

    bool inMapLoading;

    QMap<QString, TUIUIWidgetSet *> widgetSets;
    QMap<QString, QList<QVariant> > vectors;

    QList<TUIUIScriptWidget *> scriptWidgets;
};

#endif // TUIUIWIDGET_H
