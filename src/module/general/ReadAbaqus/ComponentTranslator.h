/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS ComponentTranslator
//
//  Abstraction for the mapping of component positions as read from
//  a database file onto component positions in the COVISE representation.
//
//  Initial version: 25.09.2003, Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef _COMPONENT_TRANSLATOR_H_
#define _COMPONENT_TRANSLATOR_H_

#include "odb_Types.h"

class ComponentTranslator
{
public:
    ComponentTranslator(const odb_SequenceString &secStr);
    virtual ~ComponentTranslator();
    int operator[](int i) const;

protected:
private:
    int *_translator;
    int _size;
};
#endif
