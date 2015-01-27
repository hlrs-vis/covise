/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COCONFIGXERCESENTRY_H
#define COCONFIGXERCESENTRY_H

#include <config/coConfigEntry.h>

#include <QObject>
#include <QString>
#include <QRegExp>
#include <QList>

#include <util/coTypes.h>

#include <xercesc/util/XercesDefs.hpp>
#include <config/coConfigEditorController.h>

// this class is not in coConfig lib  (leads to duplicate symbols)#include "coConfigEntryToEditor.h" //the friend
//#include "config/coEditor/mainwindow.h"

namespace XERCES_CPP_NAMESPACE
{
class DOMDocument;
class DOMElement;
class DOMNode;
};

namespace covise
{

class coConfigXercesEntry : public coConfigEntry
{
public:
    coConfigXercesEntry();
    virtual ~coConfigXercesEntry();

    static coConfigEntry *restoreFromDom(xercesc::DOMElement *node, const QString &configName);
    xercesc::DOMNode *storeToDom(xercesc::DOMDocument &document, int indent = 2);

    virtual coConfigEntry *clone() const;

protected:
    coConfigXercesEntry(const coConfigXercesEntry *source);
};
}
#endif
