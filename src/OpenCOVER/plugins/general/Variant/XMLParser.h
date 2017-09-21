/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* 
 * File:   XMLParser.h
 * Author: hpcagott
 *
 * Created on 18. August 2009, 10:23
 */

#ifndef _XMLPARSER_H
#define _XMLPARSER_H

#include <QtXml>

class SaveData
{
public:
    SaveData();
    ~SaveData();

private:
    QDomDocument *File;
};

#endif /* _XMLPARSER_H */
