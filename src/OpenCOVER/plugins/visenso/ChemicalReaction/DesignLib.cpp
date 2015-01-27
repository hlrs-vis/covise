/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <config/CoviseConfig.h>
#include "cover/coTranslator.h"

#include "DesignLib.h"
#include "Elements.h"

DesignLib *DesignLib::instance = NULL;

DesignLib::DesignLib()
{
    Design *design;

    // IMPORTANT: - Designs on top of the list will be preferred.
    //            - List same elements in a design from left to right (in case groups are moving from one molecule to another).

    design = new Design(coTranslator::coTranslate("H2O"), coTranslator::coTranslate("Wasser-Molek.")); // hoch, damit Wasserstoff und Sauerstoff in erster Linie zu Wasser reagieren
    design->addAtomConfig(ELEM_O, 0, osg::Vec3(0.0f, 0.0f, 0.25f));
    design->addAtomConfig(ELEM_H, 0, osg::Vec3(-0.65f, 0.0f, -0.3f));
    design->addAtomConfig(ELEM_H, 0, osg::Vec3(0.65f, 0.0f, -0.3f));
    designList.push_back(design);

    design = new Design(coTranslator::coTranslate("SO2"), coTranslator::coTranslate("Schwefeldioxid-Molek."));
    design->addAtomConfig(ELEM_S, 0, osg::Vec3(0.0f, 0.0f, 0.25f));
    design->addAtomConfig(ELEM_O, 0, osg::Vec3(-0.85f, 0.0f, -0.25f));
    design->addAtomConfig(ELEM_O, 0, osg::Vec3(0.85f, 0.0f, -0.25f));
    designList.push_back(design);

    design = new Design(coTranslator::coTranslate("Al2O3"), coTranslator::coTranslate("Aluminiumoxid")); // über Eisenoxid, damit Thermitverfahren möglich
    design->addAtomConfig(ELEM_Al, +3, osg::Vec3(0.0f, 0.0f, 0.68f));
    design->addAtomConfig(ELEM_Al, +3, osg::Vec3(0.0f, 0.0f, -0.68f));
    design->addAtomConfig(ELEM_O, -2, osg::Vec3(-0.6f, -0.3f, 0.0f));
    design->addAtomConfig(ELEM_O, -2, osg::Vec3(0.0f, 0.7f, 0.0f));
    design->addAtomConfig(ELEM_O, -2, osg::Vec3(0.6f, -0.3f, 0.0f));
    designList.push_back(design);

    design = new Design(coTranslator::coTranslate("Fe2O3"), coTranslator::coTranslate("Eisen(III)-Oxid"));
    design->addAtomConfig(ELEM_Fe, +3, osg::Vec3(0.0f, 0.0f, 0.65f));
    design->addAtomConfig(ELEM_Fe, +3, osg::Vec3(0.0f, 0.0f, -0.65f));
    design->addAtomConfig(ELEM_O, -2, osg::Vec3(-0.7f, -0.35f, 0.0f));
    design->addAtomConfig(ELEM_O, -2, osg::Vec3(0.0f, 0.75f, 0.0f));
    design->addAtomConfig(ELEM_O, -2, osg::Vec3(0.7f, -0.35f, 0.0f));
    designList.push_back(design);

    design = new Design(coTranslator::coTranslate("Al2S3"), coTranslator::coTranslate("Aluminiumsulfid"));
    design->addAtomConfig(ELEM_Al, +3, osg::Vec3(0.0f, 0.0f, 0.7f));
    design->addAtomConfig(ELEM_Al, +3, osg::Vec3(0.0f, 0.0f, -0.7f));
    design->addAtomConfig(ELEM_S, -2, osg::Vec3(-0.6f, -0.35f, 0.0f));
    design->addAtomConfig(ELEM_S, -2, osg::Vec3(0.0f, 0.7f, 0.0f));
    design->addAtomConfig(ELEM_S, -2, osg::Vec3(0.6f, -0.35f, 0.0f));
    designList.push_back(design);

    design = new Design(coTranslator::coTranslate("CuS"), coTranslator::coTranslate("Kupfer(II)-Sulfid"));
    design->addAtomConfig(ELEM_Cu, +2, osg::Vec3(-0.6f, 0.0f, 0.0f));
    design->addAtomConfig(ELEM_S, -2, osg::Vec3(0.55f, 0.0f, 0.0f));
    designList.push_back(design);

    design = new Design(coTranslator::coTranslate("NaCl"), coTranslator::coTranslate("Natriumchlorid"));
    design->addAtomConfig(ELEM_Na, +1, osg::Vec3(-0.5f, 0.0f, 0.0f));
    design->addAtomConfig(ELEM_Cl, -1, osg::Vec3(0.6f, 0.0f, 0.0f));
    designList.push_back(design);

    design = new Design(coTranslator::coTranslate("MgO"), coTranslator::coTranslate("Magnesiumoxid"));
    design->addAtomConfig(ELEM_Mg, +2, osg::Vec3(-0.55f, 0.0f, 0.0f));
    design->addAtomConfig(ELEM_O, -2, osg::Vec3(0.45f, 0.0f, 0.0f));
    designList.push_back(design);

    design = new Design(coTranslator::coTranslate("MgS"), coTranslator::coTranslate("Magnesiumsulfid"));
    design->addAtomConfig(ELEM_Mg, +2, osg::Vec3(-0.5f, 0.0f, 0.0f));
    design->addAtomConfig(ELEM_S, -2, osg::Vec3(0.55f, 0.0f, 0.0f));
    designList.push_back(design);

    design = new Design(coTranslator::coTranslate("CuO"), coTranslator::coTranslate("Kupfer(II)-Oxid"));
    design->addAtomConfig(ELEM_Cu, +2, osg::Vec3(-0.55f, 0.0f, 0.0f));
    design->addAtomConfig(ELEM_O, -2, osg::Vec3(0.55f, 0.0f, 0.0f));
    designList.push_back(design);

    design = new Design(coTranslator::coTranslate("FeS"), coTranslator::coTranslate("Eisen(II)-Sulfid"));
    design->addAtomConfig(ELEM_Fe, +2, osg::Vec3(-0.6f, 0.0f, 0.0f));
    design->addAtomConfig(ELEM_S, -2, osg::Vec3(0.55f, 0.0f, 0.0f));
    designList.push_back(design);

    design = new Design(coTranslator::coTranslate("CO2"), coTranslator::coTranslate("Kohlenstoffdioxid-Molek.")); // niedrig, damit sich die Verbindung zu Gunsten von Magnesiumoxid auflösen kann
    design->addAtomConfig(ELEM_C, 0, osg::Vec3(0.0f, 0.0f, 0.0f));
    design->addAtomConfig(ELEM_O, 0, osg::Vec3(-0.95f, 0.0f, 0.0f));
    design->addAtomConfig(ELEM_O, 0, osg::Vec3(0.95f, 0.0f, 0.0f));
    designList.push_back(design);

    design = new Design(coTranslator::coTranslate("O2"), coTranslator::coTranslate("Sauerstoff-Molek."));
    design->addAtomConfig(ELEM_O, 0, osg::Vec3(-0.45f, 0.0f, 0.0f));
    design->addAtomConfig(ELEM_O, 0, osg::Vec3(0.45f, 0.0f, 0.0f));
    designList.push_back(design);

    design = new Design(coTranslator::coTranslate("H2"), coTranslator::coTranslate("Wasserstoff-Molek."));
    design->addAtomConfig(ELEM_H, 0, osg::Vec3(-0.35f, 0.0f, 0.0f));
    design->addAtomConfig(ELEM_H, 0, osg::Vec3(0.35f, 0.0f, 0.0f));
    designList.push_back(design);

    design = new Design(coTranslator::coTranslate("N2"), coTranslator::coTranslate("Stickstoff-Molek."));
    design->addAtomConfig(ELEM_N, 0, osg::Vec3(-0.5f, 0.0f, 0.0f));
    design->addAtomConfig(ELEM_N, 0, osg::Vec3(0.5f, 0.0f, 0.0f));
    designList.push_back(design);

    design = new Design(coTranslator::coTranslate("Cl2"), coTranslator::coTranslate("Chlor-Molek."));
    design->addAtomConfig(ELEM_Cl, 0, osg::Vec3(-0.45f, 0.0f, 0.0f));
    design->addAtomConfig(ELEM_Cl, 0, osg::Vec3(0.45f, 0.0f, 0.0f));
    designList.push_back(design);

    for (int i = 1; i <= ELEMENT_MAX; ++i)
    {
        design = new Design(coTranslator::coTranslate(ELEMENT_SYMBOLS[i]), coTranslator::coTranslate(ELEMENT_NAMES[i] + "-Atom"));
        design->addAtomConfig(i, 0, osg::Vec3(0.0f, 0.0f, 0.0f));
        designList.push_back(design);
    }
}

Design *DesignLib::getDesign(std::string symbol)
{
    for (std::vector<Design *>::iterator it = designList.begin(); it < designList.end(); ++it)
    {
        if ((*it)->symbol.compare(symbol) == 0)
            return (*it);
    }
    return NULL;
}

DesignLib *DesignLib::Instance()
{
    if (instance == NULL)
        instance = new DesignLib();
    return instance;
}
