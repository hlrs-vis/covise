/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ELEMENT_DATABASE_H
#define _ELEMENT_DATABASE_H

#include <iostream>
#include <vector>

struct Element
{
    Element(int _number, std::string _symbol, std::string _name, int _protons, int _neutrons, int _e0, int _e1, int _e2, int _e3);

    std::string symbol, name;
    int number, protons, neutrons;
    int electrons[4];
};

class ElementDatabase
{
public:
    static ElementDatabase *Instance();

    Element findBySymbol(std::string symbol);

    std::vector<Element> elements;

private:
    ElementDatabase(); // constructor
    static ElementDatabase *instance_;
};

#endif
