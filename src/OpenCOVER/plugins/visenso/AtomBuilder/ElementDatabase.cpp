/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/CoviseConfig.h>
#include "cover/coTranslator.h"
#include "ElementDatabase.h"

#include <cover/coVRPluginSupport.h>
using namespace opencover;
using namespace covise;
ElementDatabase *ElementDatabase::instance_ = NULL;

ElementDatabase::ElementDatabase()
{
    elements.push_back(Element(0, "", "ERROR", 0, 0, 0, 0, 0, 0));
    elements.push_back(Element(1, coTranslator::coTranslate("H"), coTranslator::coTranslate("Wasserstoff"), 1, 0, 1, 0, 0, 0));
    elements.push_back(Element(2, coTranslator::coTranslate("He"), coTranslator::coTranslate("Helium"), 2, 2, 2, 0, 0, 0));
    elements.push_back(Element(3, coTranslator::coTranslate("Li"), coTranslator::coTranslate("Lithium"), 3, 4, 2, 1, 0, 0));
    elements.push_back(Element(4, coTranslator::coTranslate("Be"), coTranslator::coTranslate("Beryllium"), 4, 5, 2, 2, 0, 0));
    elements.push_back(Element(5, coTranslator::coTranslate("B"), coTranslator::coTranslate("Bor"), 5, 6, 2, 3, 0, 0));
    elements.push_back(Element(6, coTranslator::coTranslate("C"), coTranslator::coTranslate("Kohlenstoff"), 6, 6, 2, 4, 0, 0));
    elements.push_back(Element(7, coTranslator::coTranslate("N"), coTranslator::coTranslate("Stickstoff"), 7, 7, 2, 5, 0, 0));
    elements.push_back(Element(8, coTranslator::coTranslate("O"), coTranslator::coTranslate("Sauerstoff"), 8, 8, 2, 6, 0, 0));
    elements.push_back(Element(9, coTranslator::coTranslate("F"), coTranslator::coTranslate("Fluor"), 9, 10, 2, 7, 0, 0));
    elements.push_back(Element(10, coTranslator::coTranslate("Ne"), coTranslator::coTranslate("Neon"), 10, 10, 2, 8, 0, 0));
    elements.push_back(Element(11, coTranslator::coTranslate("Na"), coTranslator::coTranslate("Natrium"), 11, 12, 2, 8, 1, 0));
    elements.push_back(Element(12, coTranslator::coTranslate("Mg"), coTranslator::coTranslate("Magnesium"), 12, 12, 2, 8, 2, 0)); // ???
    elements.push_back(Element(13, coTranslator::coTranslate("Al"), coTranslator::coTranslate("Aluminium"), 13, 14, 2, 8, 3, 0));
    elements.push_back(Element(14, coTranslator::coTranslate("Si"), coTranslator::coTranslate("Silizium"), 14, 14, 2, 8, 4, 0));
    elements.push_back(Element(15, coTranslator::coTranslate("P"), coTranslator::coTranslate("Phosphor"), 15, 16, 2, 8, 5, 0));
    elements.push_back(Element(16, coTranslator::coTranslate("S"), coTranslator::coTranslate("Schwefel"), 16, 16, 2, 8, 6, 0));
    elements.push_back(Element(17, coTranslator::coTranslate("Cl"), coTranslator::coTranslate("Chlor"), 17, 18, 2, 8, 7, 0)); // ???
    elements.push_back(Element(18, coTranslator::coTranslate("Ar"), coTranslator::coTranslate("Argon"), 18, 22, 2, 8, 8, 0));
}

ElementDatabase *ElementDatabase::Instance()
{
    if (instance_ == NULL)
        instance_ = new ElementDatabase();
    return instance_;
}

Element ElementDatabase::findBySymbol(std::string symbol)
{
    for (std::vector<Element>::iterator it = elements.begin(); it < elements.end(); ++it)
    {
        if (it->symbol.compare(symbol) == 0)
            return (*it);
    }
    return elements[0];
}

Element::Element(int _number, std::string _symbol, std::string _name, int _protons, int _neutrons, int _e0, int _e1, int _e2, int _e3)
    : symbol(_symbol)
    , name(_name)
    , number(_number)
    , protons(_protons)
    , neutrons(_neutrons)
{
    electrons[0] = _e0;
    electrons[1] = _e1;
    electrons[2] = _e2;
    electrons[3] = _e3;
}
