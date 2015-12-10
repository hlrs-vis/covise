/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   02.02.2010
**
**************************************************************************/

#ifndef OSCPARSER_HPP
#define OSCPARSER_HPP

#include <QObject>
// because of tr()
#include <QMap>
#include <QStringList>

/*class QIODevice;
class QDomElement;*/
class QDomDocument;

namespace OpenScenario
{
class OpenScenarioBase;
}


class OSCParser : public QObject
{
    Q_OBJECT

    //################//
    // STATIC         //
    //################//

    enum Mode
    {
        MODE_NONE,
        MODE_XOSC,
        MODE_PROTOTYPES
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
	explicit OSCParser(OpenScenario::OpenScenarioBase *base, QObject *parent = NULL);
    ~OSCParser();

    // XODR //
    //
    bool parseXOSC(const QString &filename);

protected:
    //	OSCParser(){ /* not allowed */ };

    //################//
    // PROPERTIES     //
    //################//

private:
 //   QDomDocument *doc_;

/*    bool check(bool success, const QDomElement &element, const QString &attributeName, const QString &type);
    void setTile(const QString &id, QString &oldId);*/

    OpenScenario::OpenScenarioBase *base_;
	OSCParser::Mode mode_;

};

#endif // OSCPARSER_HPP
