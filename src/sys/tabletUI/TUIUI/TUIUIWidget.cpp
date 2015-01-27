/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TUIUIWidget.h"

#include <QtUiTools>
#include <QBuffer>
#include <QByteArray>
#include <QVBoxLayout>

#include "TUIUIWidgetSet.h"
#include "TUIUIScriptWidget.h"
#include "../TUIUITab.h"

#include "QtXml"

TUIUIWidget::TUIUIWidget(const QString &description, TUIUITab *parent)
    : QWidget(parent->getWidget())
{

    QUiLoader loader;
    QByteArray ba(description.toUtf8());
    QBuffer buffer(&ba);

    QVBoxLayout *layout = new QVBoxLayout(this);
    this->moduleWidget = loader.load(&buffer, this);
    layout->addWidget(moduleWidget);

    QList<QWidget *> children = this->moduleWidget->findChildren<QWidget *>();
    foreach (QWidget *child, children)
    {
        if (child->property("COVERScript").isValid())
        {
            TUIUIScriptWidget *scriptWidget = new TUIUIScriptWidget(child);
            connect(scriptWidget, SIGNAL(scriptActivated(QString)), this, SLOT(onEvaluateScript(QString)));
            this->scriptWidgets.append(scriptWidget);
        }
    }
}

TUIUIWidget::~TUIUIWidget()
{
}

void TUIUIWidget::addModule(const QDomNode &module)
{
    QString moduleID = module.firstChildElement("id").text();
    QString moduleTitle = module.firstChildElement("title").text();
    QString baseID = moduleID.section("_", 0, -2);

    QDomNodeList parameters = module.firstChildElement("parameters").childNodes();

    for (int ctr = 0; ctr < parameters.count(); ++ctr)
    {
        QDomElement parameter = parameters.at(ctr).toElement();

        if (parameter.isNull())
            continue;

        QString type = parameter.firstChildElement("type").text();
        QString name = parameter.firstChildElement("name").text();

        QStringList names;

        if (type.endsWith("Vector"))
        {
            QDomNodeList values = parameter.firstChildElement("values").elementsByTagName("value");
            for (unsigned int v = 0; v < values.length(); ++v)
                names << (parameter.firstChildElement("name").text() + "_" + QString::number(v));
        }
        else
        {
            names << name;
        }

        QList<QWidget *> valueWidgets;
        QWidget *valueWidget;
        int currentNameIndex = -1;

        foreach (QString pname, names)
        {
            valueWidget = this->moduleWidget->findChild<QWidget *>(baseID + "_" + pname + "_value");

            if (valueWidget == 0)
                valueWidget = this->moduleWidget->findChild<QWidget *>(moduleTitle + "_" + pname + "_value");

            ++currentNameIndex;
            if (valueWidget != 0)
            {
                std::cerr << "TUIUIWidget::addModule info: found widget " << qPrintable(valueWidget->objectName()) << std::endl;
                if (type.endsWith("Slider"))
                {
                    QVariant minimum = parameter.firstChildElement("min").text();
                    QVariant maximum = parameter.firstChildElement("max").text();
                    valueWidget->setProperty("parameterMin", minimum);
                    valueWidget->setProperty("parameterMax", maximum);
                }
                else if (type.endsWith("Vector"))
                {
                    valueWidget->setProperty("parameterIndex", currentNameIndex);
                }

                valueWidget->setProperty("moduleID", moduleID);
                valueWidget->setProperty("parameterName", name);
                valueWidget->setProperty("parameterType", type);
                valueWidgets << valueWidget;
            }
            else
            {
                std::cerr << "TUIUIWidget::addModule info: widget " << qPrintable(pname)
                          << " does not exist" << std::endl;
            }
        }

        QWidget *commitWidget = this->moduleWidget->findChild<QWidget *>(baseID + "_" + name + "_commit");
        QList<QWidget *> viewWidgets = this->moduleWidget->findChildren<QWidget *>(baseID + "_" + name + "_view");

        if (commitWidget)
            commitWidget->setProperty("moduleID", moduleID);
        foreach (QWidget *viewWidget, viewWidgets)
            viewWidget->setProperty("moduleID", moduleID);

        if (valueWidgets.empty())
            continue;

        TUIUIWidgetSet *widgetSet;
        if (type.endsWith("Vector"))
            widgetSet = new TUIUIWidgetSet(valueWidgets, commitWidget, viewWidgets, this);
        else
            widgetSet = new TUIUIWidgetSet(valueWidget, commitWidget, viewWidgets, this);

        this->widgetSets[moduleID + "_" + name] = widgetSet;

        connect(widgetSet, SIGNAL(parameterChanged(QString, QString, QString, QString, bool)),
                this, SLOT(onWidgetSetParameterChange(QString, QString, QString, QString, bool)));

        connect(widgetSet, SIGNAL(evaluateScript(QString)), this, SLOT(onEvaluateScript(QString)));

        QList<QVariant> helpers;
        QVariant value;

        if (type == "Boolean")
        {
            value = (parameter.firstChildElement("value").text() == "true");
        }
        else if (type == "String" || type == "FileBrowser" || type.endsWith("Scalar") || type.endsWith("Slider"))
        {
            value = parameter.firstChildElement("value").text();
        }
        else if (type.endsWith("Vector"))
        {
            QDomNodeList values = parameter.firstChildElement("values").elementsByTagName("value");
            QList<QVariant> valueList;
            for (unsigned int ctr = 0; ctr < values.length(); ++ctr)
                valueList << values.at(ctr).toElement().text();
            value = valueList;
        }

        widgetSet->setValue(value, helpers);
    }
}

void TUIUIWidget::setParameter(const QString &moduleID, const QString &parameter,
                               const QVariant &value, const QList<QVariant> &helpers)
{
    if (this->widgetSets.contains(moduleID + "_" + parameter))
    {
        this->widgetSets[moduleID + "_" + parameter]->setValue(value, helpers);
    }
}

void TUIUIWidget::onWidgetSetParameterChange(QString moduleID, QString name, QString type, QString v, bool linked)
{
    QVariant value(v);
    QMap<QString, QVariant> helpers = qobject_cast<TUIUIWidgetSet *>(sender())->getHelpers();

    if (type == "Boolean")
        emit boolParameterChanged(moduleID, name, value.toBool(), linked);
    else if (type == "String" || type == "FileBrowser" || type.endsWith("Vector"))
        emit stringParameterChanged(moduleID, name, value.toString(), linked);
    else if (type == "IntScalar")
        emit intParameterChanged(moduleID, name, value.toInt(), linked);
    else if (type == "FloatScalar")
        emit floatParameterChanged(moduleID, name, (float)value.toDouble(), linked);
    else if (type == "IntSlider")
        emit intBoundedParameterChanged(moduleID, name, value.toInt(), helpers["min"].toInt(), helpers["max"].toInt(), linked);
    else if (type == "FloatSlider")
        emit floatBoundedParameterChanged(moduleID, name, (float)value.toDouble(), helpers["min"].toDouble(), helpers["max"].toDouble(), linked);
}

void TUIUIWidget::onEvaluateScript(QString script)
{
    emit command("de.hlrs.opencover", "<evaluateScript><![CDATA[" + script + "]]></evaluateScript>");
}

void TUIUIWidget::processMessage(const QString &m)
{

    QDomDocument doc;
    doc.setContent(m);

    QDomElement message = doc.firstChildElement();

    if (message.tagName() == "coviseEvent")
    {
        QString type = message.attribute("type");
        if (type == "OpenNet")
        {

            this->mapfile = QDir::fromNativeSeparators(message.firstChildElement("mapname").text());

            this->mapname = this->mapfile.section("/", -1);
            if (this->mapname.endsWith(".net"))
                this->mapname.chop(4);

            this->inMapLoading = true;
        }
        if (type == "ModuleAdd")
        {
            QDomElement module = message.firstChildElement("module");
            addModule(module);
        }
        else if (type == "ParameterChange")
        {
            QString id = message.firstChildElement("moduleID").text();
            QDomNode parameter = message.firstChildElement("parameter");
            QString name = parameter.firstChildElement("name").text();
            QVariant value = parameter.firstChildElement("value").text();
            QString type = parameter.firstChildElement("type").text();
            QList<QVariant> helpers;
            if (type.endsWith("Slider"))
            {
                if (parameter.firstChildElement("min").text() != "" && parameter.firstChildElement("max").text() != "")
                {
                    helpers << parameter.firstChildElement("min").text();
                    helpers << parameter.firstChildElement("max").text();
                }
            }
            if (type.endsWith("Vector"))
            {

                QList<QVariant> valueList;
                QDomNodeList values = parameter.firstChildElement("values").elementsByTagName("value");

                for (unsigned int ctr = 0; ctr < values.length(); ++ctr)
                {
                    valueList << values.at(ctr).toElement().text();
                }

                value = valueList;
            }
            setParameter(id, name, value, helpers);
        }
        else if (type == "LinkAdd")
        {
        }
        else if (type == "OpenNetDone")
        {
            if (this->inMapLoading)
            {
                this->inMapLoading = false;
                emit(command("covise", "<executeNet/>"));
            }
        }
        else
        {
            std::cerr << "TUIUIWidget::processMessages err: unexpected event - "
                      << qPrintable(type) << std::endl;
        }
    }
}

void TUIUIWidget::boolParameterChanged(const QString &module, const QString &parameter, bool v, bool variantLinked)
{
    QString value = (v ? "true" : "false");
    value = "<value>" + value + "</value>";
    setParameter(module, parameter, value, variantLinked);
}

void TUIUIWidget::stringParameterChanged(const QString &module, const QString &parameter, const QString &value, bool variantLinked)
{
    setParameter(module, parameter, "<value>" + value + "</value>", variantLinked);
}

void TUIUIWidget::intParameterChanged(const QString &module, const QString &parameter, int v, bool variantLinked)
{
    QString value = QString::number(v);
    value = "<value>" + value + "</value>";
    setParameter(module, parameter, value, variantLinked);
}

void TUIUIWidget::floatParameterChanged(const QString &module, const QString &parameter, float v, bool variantLinked)
{
    QString value = QString::number(v);
    value = "<value>" + value + "</value>";
    setParameter(module, parameter, value, variantLinked);
}

void TUIUIWidget::intBoundedParameterChanged(const QString &module, const QString &parameter,
                                             int v, int minimum, int maximum, bool variantLinked)
{
    QString value = QString::number(v);
    value = "<value>" + QString::number(minimum) + " " + QString::number(maximum) + " " + value + "</value>";
    setParameter(module, parameter, value, variantLinked);
}

void TUIUIWidget::floatBoundedParameterChanged(const QString &module, const QString &parameter,
                                               float v, float minimum, float maximum, bool variantLinked)
{
    QString value = QString::number(v);
    value = "<value>" + QString::number(minimum) + " " + QString::number(maximum) + " " + value + "</value>";
    setParameter(module, parameter, value, variantLinked);
}

void TUIUIWidget::setParameter(const QString &module, const QString &parameter,
                               const QString &valueString, bool variantLinked)
{

    (void)variantLinked;

    QStringList moduleIDs;

    moduleIDs << module;

    foreach (QString moduleID, moduleIDs)
    {
        emit(command("de.hlrs.covise",
                     "<setParameter><moduleID>" + moduleID + "</moduleID><name>" + parameter + "</name>" + valueString + "</setParameter>"));

        if (!this->inMapLoading)
        {
            emit(command("de.hlrs.covise",
                         "<execute><moduleID>" + moduleID + "</moduleID></execute>"));
        }
    }
}

void TUIUIWidget::scriptWidgetActivated()
{
}
