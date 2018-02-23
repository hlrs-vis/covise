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

#ifndef SIGNAL_HPP
#define SIGNAL_HPP

#include <QObject>

#include <QString>
#include <QIcon>
#include <QMultiMap>

class ObjectCorner;

class SignalContainer
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SignalContainer(const QString &name, const QIcon &icon, const QString &category, const QString &type, const QString &typeSubclass, const QString &subType, double value, double distance, double heightOffset, const QString &unit, const QString &text, double width, double height)
        : signalName_(name)
        , signalIcon_(icon)
		, signalCategory_(category)
        , signalType_(type)
        , signalTypeSubclass_(typeSubclass)
        , signalSubType_(subType)
        , signalValue_(value)
        , signalDistance_(distance)
        , signalheightOffset_(heightOffset)
		, signalUnit_(unit)
		, signalText_(text)
		, signalWidth_(width)
		, signalHeight_(height)
    {
        /* does nothing */
    }

    virtual ~SignalContainer()
    { /* does nothing */
    }

    QString getSignalName() const
    {
        return signalName_;
    }
    QIcon getSignalIcon() const
    {
        return signalIcon_;
    }
    QString getSignalType() const
    {
        return signalType_;
    }
    QString getSignalTypeSubclass() const
    {
        return signalTypeSubclass_;
    }
	QString getSignalSubType() const
    {
        return signalSubType_;
    }
    double getSignalValue() const
    {
        return signalValue_;
    }
    double getSignalDistance() const
    {
        return signalDistance_;
    }
	double getSignalheightOffset() const
	{
		return signalheightOffset_;
	}
	QString getSignalUnit() const
	{
		return signalUnit_;
	}
	QString getSignalText() const
	{
		return signalText_;
	}
	double getSignalWidth() const
	{
		return signalWidth_;
	}
	double getSignalHeight() const
	{
		return signalHeight_;
	}
	const QString &getSignalCategory() const
    {
        return signalCategory_;
    }

protected:
private:
    SignalContainer(); /* not allowed */
    SignalContainer(const SignalContainer &); /* not allowed */
    SignalContainer &operator=(const SignalContainer &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    QString signalName_;
    QIcon signalIcon_;
	QString signalCategory_;
	QString signalType_;
    QString signalTypeSubclass_;
	QString signalSubType_;
    double signalValue_;
    double signalDistance_;
	double signalheightOffset_;
	QString signalUnit_;
	QString signalText_;
	double signalWidth_;
    double signalHeight_;
};

class ObjectContainer
{

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit ObjectContainer(const QString &name, const QString &file, const QIcon &icon, const QString &objectCategory, const QString &type, double length, double width, double radius, double height, double distance, double heading, double repeatDistance, const QList<ObjectCorner *> &corners)
        : objectName_(name)
		, objectFile_(file)
        , objectIcon_(icon)
		, objectCategory_(objectCategory)
        , objectType_(type)
        , objectLength_(length)
        , objectWidth_(width)
        , objectRadius_(radius)
        , objectHeight_(height)
        , objectDistance_(distance)
        , objectHeading_(heading)
        , objectRepeatDistance_(repeatDistance)
        , objectCorners_(corners)
    {
        /* does nothing */
    }

    virtual ~ObjectContainer()
    { /* does nothing */
    }

    QString getObjectName() const
    {
        return objectName_;
    }
    QString getObjectFile() const
    {
        return objectFile_;
    }
    QIcon getObjectIcon() const
    {
        return objectIcon_;
    }
    QString getObjectType() const
    {
        return objectType_;
    }
    double getObjectLength() const
    {
        return objectLength_;
    }
    double getObjectWidth() const
    {
        return objectWidth_;
    }
    double getObjectHeight() const
    {
        return objectHeight_;
    }
    double getObjectRadius() const
    {
        return objectRadius_;
    }
    double getObjectDistance() const
    {
        return objectDistance_;
    }
    double getObjectHeading() const
    {
        return objectHeading_;
    }
    double getObjectRepeatDistance() const
    {
        return objectRepeatDistance_;
    };
    QList<ObjectCorner *> getObjectCorners() const
    {
        return objectCorners_;
    };
	const QString &getObjectCategory() const
    {
        return objectCategory_;
    }

protected:
private:
    ObjectContainer(); /* not allowed */
    ObjectContainer(const ObjectContainer &); /* not allowed */
    ObjectContainer &operator=(const ObjectContainer &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
    QString objectName_;
    QString objectFile_;
    QIcon objectIcon_;
	QString objectCategory_;
    QString objectType_;
    double objectLength_;
    double objectWidth_;
    double objectRadius_;
    double objectHeading_;
    double objectDistance_;
    double objectHeight_;
    double objectRepeatDistance_;
    QList<ObjectCorner *> objectCorners_;
};

class SignalManager : public QObject
{
    Q_OBJECT

    //################//
    // STATIC         //
    //################//

public:
    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit SignalManager(QObject *parent);
    virtual ~SignalManager();

    // User Prototypes //
    //
    bool loadSignals(const QString &fileName);

	void addSignal(const QString &country, const QString &name, const QIcon &icon, const QString &categoryName, const QString &type, const QString &typeSubclass, const QString &subType, double value, double distance, double heightOffset, const QString &unit, const QString text, double width, double height);
    void addObject(const QString &country, const QString &name, const QString &file, const QIcon &icon,  const QString &categoryName, const QString &type, double length, double width, double radius, double height, double distance, double heading, double repeatDistance, const QList<ObjectCorner *> &corners);
    QList<SignalContainer *> getSignals(QString country) const
    {
        return signals_.values(country);
    };
	SignalContainer * getSignalContainer(const QString &country, const QString &type, const QString &typeSubclass, const QString &subType);
	SignalContainer * getSignalContainer(const QString &country,const QString &name);

	SignalContainer *getSelectedSignalContainer()
	{
		return selectedSignalContainer_;
	}
	void setSelectedSignalContainer(SignalContainer *signalContainer)
	{
		selectedSignalContainer_ = signalContainer;
	}

    QList<ObjectContainer *> getObjects(QString country) const
    {
        return objects_.values(country);
    };

	ObjectContainer * getObjectContainer(const QString &type);

	ObjectContainer *getSelectedObjectContainer()
	{
		return selectedObjectContainer_;
	}
	void setSelectedObjectContainer(ObjectContainer *objectContainer)
	{
		selectedObjectContainer_ = objectContainer;
	}


	QString getCountry(SignalContainer *signalContainer);
	QString getCountry(ObjectContainer *objectContainer);
	QList<QString> getCountries() const
    {
		return countries_;
    };

	void addCountry(const QString &country);

	void addCategory(const QString &category);

	int getCategoriesSize()
	{
		return categories_.size() + 2;  // add bridge and tunnel
	}

	int getCategoryNumber(const QString &category)
	{
		return categories_.indexOf(category);      //location of the category
	}

protected:
private:
    SignalManager(); /* not allowed */
    SignalManager(const SignalManager &); /* not allowed */
    SignalManager &operator=(const SignalManager &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

protected:
private:
	// countries//
	//
	QList<QString> countries_;

    // User Signals //
    //
    QMultiMap<QString, SignalContainer *> signals_;

    // User Objects //
    //
    QMultiMap<QString, ObjectContainer *> objects_;

	// Signal Categories //
	//
	QList<QString> categories_;

	// Selected signal //
	//
	SignalContainer *selectedSignalContainer_;

	// Selected object //
	//
	ObjectContainer *selectedObjectContainer_;


};

#endif // SIGNAL_HPP
