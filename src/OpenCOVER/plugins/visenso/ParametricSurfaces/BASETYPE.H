/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __BASETYPE_H__
#define __BASETYPE_H__

/*
	basetype.h
	Die Klasse basetype definiert die Struktur eines "algebraischen"
	Listenelments.
	In der Symboltabelle, auf die mit _sym verwiesen wird, stehen naehere
	Informationen, wie Typ (IDENT, NUMBER, SIN, PLUS, ...), bei Typ IDENT
	Bindungen wie zugeordneter Ausdruck oder Variablenwert.
*/

#include "SYMBOL.H"

class HlBaseType
{

public:
    // Konstruktor
    HlBaseType()
        : mSym(NULL)
        , mVal(0.0)
    {
    }

    // Doublewert lesen und setzen
    double getDoubleVal()
    {
        return mVal;
    }
    void setDoubleVal(double val)
    {
        mVal = val;
    }

    // Symboltableeintrag lesen und setzen
    HlSymEntry *getsym()
    {
        return mSym;
    }
    void setsym(HlSymEntry *sym)
    {
        mSym = sym;
    }

private:
    HlSymEntry *mSym; // Verweis auf Symboltabelleneintrag
    double mVal; // Wert bei Konstanten
};

#endif // __BASETYPE_H__
