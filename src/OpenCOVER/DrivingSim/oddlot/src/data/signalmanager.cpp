/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   12.05.2010
**
**************************************************************************/

#include "signalmanager.hpp"

// Data //
//

// IO //
//
#include "src/io/domparser.hpp"

// Qt //
//
#include <QFile>
#include <QMessageBox>
#include <QApplication>

#include "math.h"

//####################//
//                    //
// SIGNALMANAGER   //
//                    //
//####################//

SignalManager::SignalManager(QObject *parent)
    : QObject(parent),
      selectedSignalContainer_(NULL)
{
}

SignalManager::~SignalManager()
{
    foreach (SignalContainer *signalType, signals_)
    {
        delete signalType;
    }
}

void
SignalManager::addSignal(const QString &country, const QString &name, const QIcon &icon, const QString &categoryName, const QString &type, const QString &typeSubclass, const QString &subType, double value, double distance, double heightOffset, const QString &unit, const QString text, double width, double height)
{
    signals_.insert(country, new SignalContainer(name, icon, categoryName, type, typeSubclass, subType, value, distance, heightOffset, unit, text, width, height));
}

void
SignalManager::addObject(const QString &country, const QString &name, const QString &file, const QIcon &icon, const QString &categoryName, const QString &type, double length, double width, double radius, double height, double distance, double heading, double repeatDistance, const QList<ObjectCorner *> &corners)
{
    objects_.insert(country, new ObjectContainer(name, file, icon, categoryName, type, length, width, radius, height, distance, heading, repeatDistance, corners));
}

void
SignalManager::addCountry(const QString &country)
{
    if (!countries_.contains(country))
    {
        countries_.append(country);
    }
}

void
SignalManager::addCategory(const QString &category)
{
    if (!categories_.contains(category))
    {
        categories_.append(category);
    }
}

ObjectContainer *
SignalManager::getObjectContainer(const QString &type)
{
    QMultiMap<QString, ObjectContainer *>::const_iterator iter = objects_.constBegin();
    while (iter != objects_.constEnd())
    {
        if (iter.value()->getObjectType() == type)
        {
            return iter.value();
        }
        iter++;
    }

    return NULL;
}

QString
SignalManager::getCountry(SignalContainer *signalContainer)
{
    return signals_.key(signalContainer, "");

}

QString
SignalManager::getCountry(ObjectContainer *objectContainer)
{
    return objects_.key(objectContainer, "");

}

SignalContainer *
SignalManager::getSignalContainer(const QString &country, const QString &type, const QString &typeSubclass, const QString &subType)
{
	QList<SignalContainer *>countryList = signals_.values(country);

	for (auto i = 0; i < countryList.size(); i++)
    {
		SignalContainer *signal = countryList.at(i);
        if ((signal->getSignalType() == type) && (signal->getSignalSubType() == subType) && (signal->getSignalTypeSubclass() == typeSubclass))
        {
            return signal;
        }
    }

    return NULL;
}

SignalContainer *
SignalManager::getSignalContainer(const QString &country, const QString &name)
{
	QList<SignalContainer *>countryList = signals_.values(country);

	for (auto i = 0; i < countryList.size(); i++)
	{
		SignalContainer *signal = countryList.at(i);
		if (signal->getSignalName() == name)
		{
			return signal;
		}
	}

    return NULL;
}


bool
SignalManager::loadSignals(const QString &fileName)
{
    // Print //
    //
    qDebug("Loading file: %s", fileName.toUtf8().constData());

    // Open file //
    //
    QFile file(fileName);
    if (!file.open(QFile::ReadOnly | QFile::Text))
    {
        //		QMessageBox::warning(this, tr("ODD"), tr("Cannot read file %1:\n%2.")
        //		.arg(fileName)
        //		.arg(file.errorString()));
        qDebug("Loading file failed: %s", fileName.toUtf8().constData());
        return false;
    }

    // Parse file //
    //
    QApplication::setOverrideCursor(Qt::WaitCursor);
    DomParser *parser = new DomParser(NULL);
    parser->parseSignals(&file); // parser calls addPrototype()
    delete parser;

    // Close file //
    //
    QApplication::restoreOverrideCursor();
    file.close();
    return true;
}
