/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __Element_h__
#define __Element_h__

class Element
{
public:
    int numberofNode[8];
    int number;

public:
    Element(void);

public:
    ~Element(void);
};

#endif
