/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <config/CoviseConfig.h>
#include "cover/coTranslator.h"
#include "AlkaneDatabase.h"
#include <stdio.h>

AlkaneDatabase *AlkaneDatabase::instance_ = NULL;

AlkaneDatabase::AlkaneDatabase()
{
    alkanes.push_back(Alkane("", "ERROR", true, 0, 0));
    alkanes.push_back(Alkane("CH4", coTranslator::coTranslate("Methan"), true, 1, 4));
    alkanes.push_back(Alkane("C2H6", coTranslator::coTranslate("Ethan"), true, 2, 6));
    alkanes.push_back(Alkane("C3H8", coTranslator::coTranslate("Propan"), true, 3, 8));
    alkanes.push_back(Alkane("n-C4H10", coTranslator::coTranslate("n-Butan"), true, 4, 10));
    alkanes.push_back(Alkane("i-C4H10", coTranslator::coTranslate("i-Butan"), false, 4, 10));
    alkanes.push_back(Alkane("n-C5H12", coTranslator::coTranslate("n-Pentan"), true, 5, 12));
    alkanes.push_back(Alkane("i-C5H12", coTranslator::coTranslate("i-Pentan"), false, 5, 12));
    alkanes.push_back(Alkane("n-C6H14", coTranslator::coTranslate("n-Hexan"), true, 6, 14));
    alkanes.push_back(Alkane("i-C6H14", coTranslator::coTranslate("i-Hexan"), false, 6, 14));
    alkanes.push_back(Alkane("n-C7H16", coTranslator::coTranslate("n-Heptan"), true, 7, 16));
    alkanes.push_back(Alkane("i-C7H16", coTranslator::coTranslate("i-Heptan"), false, 7, 16));
    alkanes.push_back(Alkane("n-C8H18", coTranslator::coTranslate("n-Octan"), true, 8, 18));
    alkanes.push_back(Alkane("i-C8H18", coTranslator::coTranslate("i-Octan"), false, 8, 18));
    alkanes.push_back(Alkane("n-C9H20", coTranslator::coTranslate("n-Nonan"), true, 9, 20));
    alkanes.push_back(Alkane("i-C9H20", coTranslator::coTranslate("i-Nonan"), false, 9, 20));
    alkanes.push_back(Alkane("n-C10H22", coTranslator::coTranslate("n-Decan"), true, 10, 22));
    alkanes.push_back(Alkane("i-C10H22", coTranslator::coTranslate("i-Decan"), false, 10, 22));
    alkanes.push_back(Alkane("n-C11H24", coTranslator::coTranslate("n-Undecan"), true, 11, 24));
    alkanes.push_back(Alkane("i-C11H24", coTranslator::coTranslate("i-Undecan"), false, 11, 24));
    alkanes.push_back(Alkane("n-C12H26", coTranslator::coTranslate("n-Dodecan"), true, 12, 26));
    alkanes.push_back(Alkane("i-C12H26", coTranslator::coTranslate("i-Dodecan"), false, 12, 26));
}

AlkaneDatabase *AlkaneDatabase::Instance()
{
    if (instance_ == NULL)
        instance_ = new AlkaneDatabase();
    return instance_;
}

Alkane AlkaneDatabase::findByFormula(std::string formula)
{
    for (std::vector<Alkane>::iterator it = alkanes.begin(); it < alkanes.end(); ++it)
    {
        if (it->formula.compare(formula) == 0)
            return (*it);
    }
    return alkanes[0];
}

Alkane::Alkane(std::string f, std::string n, bool l, int c, int h)
    : formula(f)
    , name(n)
    , linear(l)
    , carbons(c)
    , hydrogens(h)
{
}
Alkane
AlkaneDatabase::findByAtoms(int numc, int numh)
{
    fprintf(stderr, "AlkaneDatabase::findByAtoms %d %d\n", numc, numh);
    for (std::vector<Alkane>::iterator it = alkanes.begin(); it < alkanes.end(); ++it)
    {
        fprintf(stderr, "testing %s\n", it->name.c_str());
        if ((it->carbons == numc) && (it->hydrogens == numh))
        {
            fprintf(stderr, "found alkane %s\n", it->name.c_str());
            return (*it);
        }
    }
    return alkanes[0];
}

Alkane
AlkaneDatabase::findByAtoms(int numc, int numh, bool linear)
{
    fprintf(stderr, "AlkaneDatabase::findByAtoms %d %d\n", numc, numh);
    for (std::vector<Alkane>::iterator it = alkanes.begin(); it < alkanes.end(); ++it)
    {
        fprintf(stderr, "testing %s\n", it->name.c_str());
        if ((it->carbons == numc) && (it->hydrogens == numh) && (it->linear == linear))
        {
            fprintf(stderr, "found alkane %s\n", it->name.c_str());
            return (*it);
        }
    }
    return alkanes[0];
}
