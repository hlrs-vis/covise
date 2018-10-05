/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_CHEMICAL_ELEMENT_H
#define CO_CHEMICAL_ELEMENT_H

#include <util/coExport.h>
#include <string>
#include <vector>
#include <map>

/// list of all chemical elements

namespace covise
{

    class ALGEXPORT coChemicalElement
    {
        public:
        int number;
        std::string name;
        std::string symbol;
        float radius;
        float color[4];
        coChemicalElement();
        coChemicalElement(int number, std::string n, std::string sym, float r, float re, float g, float b);
        coChemicalElement(const coChemicalElement &ce);
    };

    class ALGEXPORT coAtomInfo
    {

    public:
        static coAtomInfo* instance();
        int getType(const std::string &type);
        float getRadius(int type);
        void getColor(int type, float (&color)[4]);
        std::vector<coChemicalElement> all;
        std::map<std::string, int> idMap;


    private:
        coAtomInfo();
        static bool initialized;
        static const int numStaticAtoms = 118;
        static coChemicalElement allStatic[numStaticAtoms];

        static coAtomInfo* myInstance;
    };
}

#endif
