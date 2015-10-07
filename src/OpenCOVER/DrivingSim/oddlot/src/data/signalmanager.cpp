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
    : QObject(parent)
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
SignalManager::addSignal(const QString &country, const QString &name, const QIcon &icon, const QString &categoryName, int type, const QString &typeSubclass, int subType, double value, double distance, double height)
{
    signals_.insert(country, new SignalContainer(name, icon, categoryName, type, typeSubclass, subType, value, distance, height));
}

void
SignalManager::addObject(const QString &country, const QString &name, const QString &file, const QIcon &icon, const QString &type, double length, double width, double radius, double height, double distance, double heading, double repeatDistance, const QList<ObjectCorner *> &corners)
{
    objects_.insert(country, new ObjectContainer(name, file, icon, type, length, width, radius, height, distance, heading, repeatDistance, corners));
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

SignalContainer * 
SignalManager::getSignalContainer(int type, const QString &typeSubclass, int subType)
{
   QMultiMap<QString, SignalContainer *>::const_iterator iter = signals_.constBegin();
   while (iter != signals_.constEnd())
   {
	   if ((iter.value()->getSignalType() == type) && (iter.value()->getSignalSubType() == subType) && (iter.value()->getSignalTypeSubclass() == typeSubclass))
      {
         return iter.value();
      }
   iter++;
   }

   return NULL;
} 

SignalContainer * 
SignalManager::getSignalContainer(const QString &name)
{
   QMultiMap<QString, SignalContainer *>::const_iterator iter = signals_.constBegin();
   while (iter != signals_.constEnd())
   {
	   if (iter.value()->getSignalName() == name)
      {
         return iter.value();
      }
   iter++;
   }

   return NULL;
} 


bool
SignalManager::loadSignals(const QString &fileName)
{
    // Print //
    //
    qDebug("Loading file: " + fileName.toUtf8());

    // Open file //
    //
    QFile file(fileName);
    if (!file.open(QFile::ReadOnly | QFile::Text))
    {
        //		QMessageBox::warning(this, tr("ODD"), tr("Cannot read file %1:\n%2.")
        //		.arg(fileName)
        //		.arg(file.errorString()));
        qDebug("Loading file failed: " + fileName.toUtf8());
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
