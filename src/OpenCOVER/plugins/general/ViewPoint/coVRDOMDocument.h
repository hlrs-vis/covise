/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_VR_DOM_DOC_H
#define CO_VR_DOM_DOC_H

#include <QDomDocument>
#include <QDomElement>

#include <util/coTypes.h>
class coVRDOMDocument
{
public:
    // read
    static coVRDOMDocument *instance();
    ~coVRDOMDocument();

    // 1. cover->vp_path[0]='0'
    //    - create root element
    //    - create dom
    // 2. cover->vp_path given, but does not yet exist:
    //    if !root element
    //      - create root element, create dom
    //    - open file for write
    //    - save dom
    //    - close file
    // 3. cover->vp_path given, file exists
    //    - open file for read
    //    - set dom content from file
    //    - close file
    bool read(const char *file = NULL);

    // save dom content to file
    bool save();

    // return the root element <coverstate>
    QDomElement getRootElement()
    {
        return rootElement;
    };
    QDomDocument getDocument()
    {
        return doc;
    };
    ////addElement();
protected:
    coVRDOMDocument();

private:
    static coVRDOMDocument *domDoc;
    QDomDocument doc;
    QDomElement rootElement;
    const char *vwpPath;
    const char *readVwpPath;
};
#endif
