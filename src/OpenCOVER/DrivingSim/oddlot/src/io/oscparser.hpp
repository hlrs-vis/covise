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
class ProjectData;
class OSCBase;

namespace OpenScenario
{
class OpenScenarioBase;
class oscObjectBase;
class oscObject;
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
	explicit OSCParser(OpenScenario::OpenScenarioBase *openScenarioBase, QObject *parent = NULL);
    ~OSCParser();

    // XODR //
    //
    bool parseXOSC(const QString &filename, const QString &nodeName = "OpenSCENARIO", const QString &fileType = "OpenSCENARIO");
	void createElements(const OpenScenario::oscObjectBase *oscObjectBase);

protected:
    //	OSCParser(){ /* not allowed */ };

    //################//
    // PROPERTIES     //
    //################//

private:
 //   QDomDocument *doc_;

/*    bool check(bool success, const QDomElement &element, const QString &attributeName, const QString &type);
    void setTile(const QString &id, QString &oldId);*/
	ProjectData *projectData_;

	OSCBase *oscBase_; // ODDLOT OpenScenario base element
    OpenScenario::OpenScenarioBase *openScenarioBase_; // OpenScenario base object
	OSCParser::Mode mode_;

};

#endif // OSCPARSER_HPP
