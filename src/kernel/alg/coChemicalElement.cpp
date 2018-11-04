/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coChemicalElement.h"
#include <config/CoviseConfig.h>
#include <config/coConfigGroup.h>
#include <config/coConfig.h>
#include <iostream>

using namespace covise;
coAtomInfo* coAtomInfo::myInstance = nullptr;


coChemicalElement coAtomInfo::allStatic[coAtomInfo::numStaticAtoms] = {
coChemicalElement( 1, "Hydrogen", "H"         ,1.20f  ,0.8f ,0.8f ,0.8f ),
coChemicalElement( 2, "Helium", "He"          ,1.22f  ,1.0f ,0.0f ,0.0f ),
coChemicalElement( 3, "Lithium", "Li"         ,0.00f  ,1.0f ,0.5f ,0.0f ),
coChemicalElement( 4, "Beryllium", "Be"       ,0.00f  ,1.0f ,1.0f ,0.0f ),
coChemicalElement( 5, "Boron", "B"            ,2.08f  ,0.7f ,1.0f ,0.0f ),
coChemicalElement( 6, "Carbon", "C"           ,1.85f  ,0.0f ,1.0f ,0.0f ),
coChemicalElement( 7, "Nitrogen", "N"         ,1.54f  ,0.0f ,1.0f ,0.5f ),
coChemicalElement( 8, "Oxygen", "O"           ,1.40f  ,1.0f ,0.0f ,0.0f ),
coChemicalElement( 9, "Fluorine", "F"         ,1.35f  ,0.0f ,0.5f ,1.0f ),
coChemicalElement( 10, "Neon", "Ne"           ,1.60f  ,0.0f ,0.0f ,1.0f ),
coChemicalElement( 11, "Sodium", "Na"         ,2.31f  ,0.5f ,0.0f ,1.0f ),
coChemicalElement( 12, "Magnesium", "Mg"      ,0.99f  ,1.0f ,0.0f ,1.0f ),
coChemicalElement( 13, "Aluminum", "Al"       ,2.05f  ,1.0f ,0.0f ,0.5f ),
coChemicalElement( 14, "Silicon", "Si"        ,2.00f  ,0.0f ,0.0f ,1.0f ),
coChemicalElement( 15, "Phosphorus", "P"      ,1.90f  ,0.0f ,0.0f ,1.0f ),
coChemicalElement( 16, "Sulfur", "S"          ,1.85f  ,0.0f ,0.0f ,1.0f ),
coChemicalElement( 17, "Chlorine", "Cl"       ,1.81f  ,0.0f ,0.0f ,1.0f ),
coChemicalElement( 18, "Argon", "Ar"          ,1.91f  ,1.0f ,0.0f ,0.0f ),
coChemicalElement( 19, "Potassium", "K"       ,2.31f  ,0.0f ,0.0f ,1.0f ),
coChemicalElement( 20, "Calcium", "Ca"        ,0.99f  ,0.0f ,0.0f ,1.0f ),
coChemicalElement( 21, "Scandium", "Sc"       ,0.99f  ,0.0f ,0.0f ,1.0f ),
coChemicalElement( 22, "Titanium", "Ti"       ,0.99f  ,0.0f ,0.0f ,1.0f ),
coChemicalElement( 23, "Vanadium", "V"        ,0.99f  ,0.0f ,0.0f ,1.0f ),
coChemicalElement( 24, "Chromium", "Cr"       ,0.99f  ,0.0f ,0.0f ,1.0f ),
coChemicalElement( 25, "Manganese", "Mn"      ,0.99f  ,0.0f ,0.0f ,1.0f ),
coChemicalElement( 26, "Iron", "Fe"           ,0.99f  ,0.0f ,0.0f ,1.0f ),
coChemicalElement( 27, "Cobalt", "Co"         ,0.99f  ,0.0f ,0.0f ,1.0f ),
coChemicalElement( 28, "Nickel", "Ni"         ,0.99f  ,0.0f ,0.0f ,1.0f ),
coChemicalElement( 29, "Copper", "Cu"         ,0.99f  ,0.0f ,0.0f ,1.0f ),
coChemicalElement( 30, "Zinc", "Zn"           ,0.99f  ,0.0f ,0.0f ,1.0f ),
coChemicalElement( 31, "Gallium", "Ga"        ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 32, "Germanium", "Ge"      ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 33, "Arsenic", "As"        ,2.00f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 34, "Selenium", "Se"       ,2.00f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 35, "Bromine", "Br"        ,1.95f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 36, "Krypton", "Kr"        ,1.98f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 37, "Rubidium", "Rb"       ,2.44f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 38, "Strontium", "Sr"      ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 39, "Yttrium", "Y"         ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 40, "Zirconium", "Zr"      ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 41, "Niobium", "Nb"        ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 42, "Molybdenum", "Mo"     ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 43, "Technetium", "Tc"     ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 44, "Ruthenium", "Ru"      ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 45, "Rhodium", "Rh"        ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 46, "Palladium", "Pd"      ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 47, "Silver", "Ag"         ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 48, "Cadmium", "Cd"        ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 49, "Indium", "In"         ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 50, "Tin", "Sn"            ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 51, "Antimony", "Sb"       ,2.20f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 52, "Tellurium", "Te"      ,2.20f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 53, "Iodine", "I"          ,2.15f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 54, "Xenon", "Xe"          ,2.16f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 55, "Cesium", "Cs"         ,2.62f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 56, "Barium", "Ba"         ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 57, "Lanthanum", "La"      ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 58, "Cerium", "Ce"         ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 59, "Praseodymium","Pr"    ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 60, "Neodymium", "Nd"      ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 61, "Promethium", "Pm"     ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 62, "Samarium", "Sm"       ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 63, "Europium", "Eu"       ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 64, "Gadolinium", "Gd"     ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 65, "Terbium", "Tb"        ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 66, "Dysprosium", "Dy"     ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 67, "Holmium", "Ho"        ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 68, "Erbium", "Er"         ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 69, "Thulium", "Tm"        ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 70, "Ytterbium", "Yb"      ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 71, "Lutetium", "Lu"       ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 72, "Hafnium", "Hf"        ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 73, "Tantalum", "Ta"       ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 74, "Tungsten", "W"        ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 75, "Rhenium", "Re"        ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 76, "Osmium", "Os"         ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 77, "Iridium", "Ir"        ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 78, "Platinum", "Pt"       ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 79, "Gold", "Au"           ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 80, "Mercury", "Hg"        ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 81, "Thallium", "Tl"       ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 82, "Lead", "Pb"           ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 83, "Bismuth", "Bi"        ,2.40f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 84, "Polonium", "Po"       ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 85, "Astatine", "At"       ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 86, "Radon", "Rn"          ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 87, "Francium", "Fr"       ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 88, "Radium", "Ra"         ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 89, "Actinium", "Ac"       ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 90, "Thorium", "Th"        ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 91, "Protactinium","Pa"    ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 92, "Uranium", "U"         ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 93, "Neptunium", "Np"      ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 94, "Plutonium", "Pu"      ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 95, "Americium", "Am"      ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 96, "Curium", "Cm"         ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 97, "Berkelium", "Bk"      ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 98, "Californium","Cf"     ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 99, "Einsteinium","Es"     ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 100, "Fermium", "Fm"       ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 101, "Mendelevium","Md"    ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 102, "Nobelium",  "No"     ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 103, "Lawrencium", "Lr"    ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 104, "Rutherfordium", "Rf" ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 105, "Dubnium", "Db"       ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 106, "Seaborgium", "Sg"    ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 107, "Bohrium", "Bh"       ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 108, "Hassium", "Hs"       ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 109, "Meitnerium", "Mt"    ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 110, "Darmstadtium", "Ds"  ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 111, "Roentgenium", "Rg"   ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 112, "Ununbium", "Uub"     ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 113, "Ununtrium", "Uut"    ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 114, "Ununquadium", "Uuq"  ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 115, "Ununpentium", "Uup"  ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 116, "Ununhexium", "Uuh"   ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 117, "Ununseptium", "Uus"  ,0.99f  ,1.0f ,1.0f ,1.0f ),
coChemicalElement( 118, "Ununoctium", "Uuo"   ,0.99f  ,1.0f ,1.0f ,1.0f )
};

coAtomInfo * covise::coAtomInfo::instance()
{
    if (myInstance == nullptr)
    {
        myInstance = new coAtomInfo();
    }
    return myInstance;
}

covise::coAtomInfo::coAtomInfo()
{
    myInstance = this;
    for (int i = 0; i < numStaticAtoms; i++)
    {
        all.push_back(allStatic[i]);
        idMap[allStatic[i].symbol] = i;
    }

    coConfigGroup *m_mapConfig;
    // try to add local atommapping.xml to current coviseconfig
    m_mapConfig = new coConfigGroup("Module.AtomColors");
    m_mapConfig->addConfig(coConfigDefaultPaths::getDefaultLocalConfigFilePath() + "atommapping.xml", "local", true);
    coConfig::getInstance()->addConfig(m_mapConfig);

    coCoviseConfig::ScopeEntries mappingEntries = coCoviseConfig::getScopeEntries("Module.AtomMapping");
    if (mappingEntries.getValue() == NULL)
    {
        // add global atommapping.xml to current coviseconfig
        m_mapConfig->addConfig(coConfigDefaultPaths::getDefaultGlobalConfigFilePath() + "atommapping.xml", "global", true);
        coConfig::getInstance()->addConfig(m_mapConfig);
        // retrieve the values of atommapping.xml and build the GUI
    }
    coCoviseConfig::ScopeEntries mappingEntries2 = coCoviseConfig::getScopeEntries("Module.AtomMapping");

    const char **mapEntry = mappingEntries2.getValue();
    if (mapEntry == NULL)
        std::cout << "AtomMapping is NULL" << std::endl;
    int iNrCurrent = 0;
    float radius;
    char cAtomName[256];

    if (mapEntry == NULL || *mapEntry == NULL)
        std::cout << "The scope Module.AtomMapping is not available in your covise.config file!" << std::endl;

    const char **curEntry = mapEntry;
    while (curEntry && *curEntry)
    {
        float ac[4];
        char acType[100];
        coChemicalElement ce;
        int iScanResult = sscanf(curEntry[1], "%s %s %f %f %f %f %f", acType, cAtomName, &radius, &ac[0], &ac[1], &ac[2], &ac[3]);
        if (iScanResult == 7)
        {
            ce.symbol = acType;
            ce.name = cAtomName;
            ce.radius = radius;
            ce.color[0] = ac[0];
            ce.color[1] = ac[1];
            ce.color[2] = ac[2];
            ce.color[3] = ac[3];
            //try to find this atom, if found overwrite the built in data, otherwise append it
            auto elem = idMap.find(ce.symbol);
            if (elem != idMap.end())
            {
                all[elem->second] = ce;
            }
            else
            {
                size_t num = all.size();
                all.push_back(ce);
                idMap[all[num].symbol] = (int)num;
            }
        }
        curEntry += 2;
    }
}

covise::coChemicalElement::coChemicalElement()
{
    number = (int)coAtomInfo::instance()->all.size();
    radius = 1.0;
    color[0] = 1.0;
    color[1] = 0.0;
    color[2] = 0.0;
    color[3] = 1.0;
}

covise::coChemicalElement::coChemicalElement(int num, std::string n, std::string sym, float r, float re, float g, float b)
{
    number = num;
    name = n;
    symbol = sym;
    radius = r;
    color[0] = re;
    color[1] = g;
    color[2] = b;
    color[3] = 1.0;
}

covise::coChemicalElement::coChemicalElement(const coChemicalElement & ce)
{
    number = ce.number;
    name = ce.name;
    symbol = ce.symbol;
    radius = ce.radius;
    color[0] = ce.color[0];
    color[1] = ce.color[1];
    color[2] = ce.color[2];
    color[3] = ce.color[3];
}
