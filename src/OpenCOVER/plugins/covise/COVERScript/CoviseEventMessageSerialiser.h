/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVISEEVENTMESSAGESERIALISER_H
#define COVISEEVENTMESSAGESERIALISER_H
#include <util/coTypes.h>
#include <QString>

#include <QtXml>

namespace covise
{
class covise__Event;
class covise__LinkAddEvent;
class covise__ModuleAddEvent;
class covise__ModuleDelEvent;
class covise__ModuleDiedEvent;
class covise__ParameterChangeEvent;
class covise__OpenNetEvent;
class covise__OpenNetDoneEvent;
class covise__QuitEvent;

class covise__Module;
class covise__Port;
class covise__Link;
class covise__Parameter;
class covise__Point;
}

class CoviseEventMessageSerialiser
{

public:
    static QDomElement serialise(const covise::covise__Event *event, QDomDocument &doc);
    static QDomElement serialise(const covise::covise__LinkAddEvent *event, QDomDocument &doc);
    static QDomElement serialise(const covise::covise__ModuleAddEvent *event, QDomDocument &doc);
    static QDomElement serialise(const covise::covise__ModuleDelEvent *event, QDomDocument &doc);
    static QDomElement serialise(const covise::covise__ModuleDiedEvent *event, QDomDocument &doc);
    static QDomElement serialise(const covise::covise__ParameterChangeEvent *event, QDomDocument &doc);
    static QDomElement serialise(const covise::covise__OpenNetEvent *event, QDomDocument &doc);
    static QDomElement serialise(const covise::covise__OpenNetDoneEvent *event, QDomDocument &doc);
    static QDomElement serialise(const covise::covise__QuitEvent *event, QDomDocument &doc);

    static QDomElement serialise(const covise::covise__Module *module, QDomDocument &doc);
    static QDomElement serialise(const covise::covise__Port *port, QDomDocument &doc);
    static QDomElement serialise(const covise::covise__Link *link, QDomDocument &doc);
    static QDomElement serialise(const covise::covise__Parameter *parameter, QDomDocument &doc);
    static QDomElement serialise(const covise::covise__Point *point, QDomDocument &doc);

private:
    CoviseEventMessageSerialiser()
    {
    }
    ~CoviseEventMessageSerialiser()
    {
    }
};

#endif // COVISEEVENTMESSAGESERIALISER_H
