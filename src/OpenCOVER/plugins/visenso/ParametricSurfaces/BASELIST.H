/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __BASELIST_H__
#define __BASELIST_H__

#include <vector>
#include <list>
#include <algorithm>
using std::vector;
using std::list;
using std::find;

#include "BASETYPE.H"

/*
	baselist.h
	Die Klasse baselist ist die Basisliste, mit Konstruktoren, Destruktoren,
	Typpruefungsmethoden, Methoden zum Anhaengen und Loeschen von Argumenten,
	Kopiermethoden.
*/

class HlBaseList;
typedef vector<HlBaseList *> BaselistVector;

class HlBaseList : protected HlBaseType
{

private:
    // Zeiger auf Array von Argumenten vom Typ HlBaseList
    BaselistVector mArg;

public:
    // Konstruktoren
    HlBaseList();
    HlBaseList(types);
    HlBaseList(double);
    HlBaseList(HlSymEntry *);

    static void setSymTab(HlSymbolTable *symtab);

    // Destruktor
    virtual ~HlBaseList();

    // Argumentanzahl lesen und setzen
    int Length();

    // Typ des Listenkopfs (IDENT, NUMBER, SIN, ...)
    types typeOfHead();

    // Typpruefung
    bool Is(types t);
    bool Is(double v);

    // Zugriff auf Listenelement i
    HlBaseList *arg(unsigned int i);
    void setarg(unsigned int i, HlBaseList *e);

    // Argumente i und k vertauschen
    void changearg(unsigned int i, unsigned int k);

    // Argumente an this anhaengen
    // Alle Argumente von *e an this anhaengen
    HlBaseList *appargs(HlBaseList *e);

    // Ein Argument an this anhaengen
    HlBaseList *apparg(HlBaseList *);
    HlBaseList *apparg(types);
    HlBaseList *apparg(double);

    // Falls *e in der Argumentliste von this vorhanden -> *e loeschen
    HlBaseList *delarg(HlBaseList *e);

    // onlyfirst loescht this und gibt eine Kopie des 1. Arguments zurueck
    HlBaseList *onlyFirstArg();

    // this und seine Argumente werden rekursiv kopiert
    HlBaseList *copy();
};

/*------------------------------------------------------------*/
/*                Konstruktoren fuer HlBaseList                    */
/*------------------------------------------------------------*/

// Standardkonstruktor
inline HlBaseList::HlBaseList()
{
}

// Konstruktor fuer Baselist vom Typ NUMBER
inline HlBaseList::HlBaseList(double v)
{
    extern HlSymbolTable *SYMTABPTR;
    setDoubleVal(v);
    setsym(SYMTABPTR->lookup(NUMBER));
}

inline HlBaseList::HlBaseList(types t)
{
    extern HlSymbolTable *SYMTABPTR;
    setsym(SYMTABPTR->lookup(t));
    setDoubleVal(getsym()->getVal());
}

inline void HlBaseList::setSymTab(HlSymbolTable *symtab)
{
    extern HlSymbolTable *SYMTABPTR;
    SYMTABPTR = symtab;
}

// Konstruktor fuer Baselist mit Symboltabelleneintrag *se
inline HlBaseList::HlBaseList(HlSymEntry *se)
{
    setDoubleVal(se->getVal());
    setsym(se);
}

/*---------------------------------------------------------------*/
/*    Zugriffsmethoden                                           */
/*---------------------------------------------------------------*/

// Lesen auf Argument i
inline HlBaseList *HlBaseList::arg(unsigned int i)
{
    return mArg[i];
}

// Argument i mit *e belegen
inline void HlBaseList::setarg(unsigned int i, HlBaseList *e)
{
    mArg[i] = e;
}

// Argumente i und k vertauschen
inline void HlBaseList::changearg(unsigned int i, unsigned int k)
{
    HlBaseList *h = mArg[i];
    mArg[i] = mArg[k];
    mArg[k] = h;
}

// Typ des Listenkopfes zurueckgeben
inline types HlBaseList::typeOfHead()
{
    return getsym()->getToken();
}

// apparg: Haengt *e als Argument an die Argumenteliste an
inline HlBaseList *HlBaseList::apparg(HlBaseList *e)
{
    mArg.push_back(e);
    return e;
}

// apparg(types t): haengt t an die Argumentliste an
inline HlBaseList *HlBaseList::apparg(types t)
{
    return apparg(new HlBaseList(t));
}

// apparg(double v): haengt v an die Argumentliste an
inline HlBaseList *HlBaseList::apparg(double v)
{
    return apparg(new HlBaseList(v));
}

/*---------------------------------------------------------------*/

// Typ des Listenkopfes mit t vergleichen
inline bool HlBaseList::Is(types t)
{
    return typeOfHead() == t;
}

// Wert des Listenkopfes mit v vergleichen
inline bool HlBaseList::Is(double v)
{
    return Is(NUMBER) ? (getDoubleVal() == v) ? true : false : false;
}

// Argumentanzahl zurueckgeben
inline int HlBaseList::Length()
{
    return mArg.size();
}

// ~HlBaseList: Rekursiver Destruktor fuer HlBaseList
inline HlBaseList::~HlBaseList()
{
    for (unsigned int i = 0, n = Length(); i < n; i++)
        delete mArg[i];

    mArg.clear();
}

// onlyFirstArg: Loescht this und gibt das vorher kopierte 1. Argument zurueck
inline HlBaseList *HlBaseList::onlyFirstArg()
{
    if (Length())
    {
        HlBaseList *h = arg(0)->copy();
        delete this;
        return h;
    }
    return NULL;
}

// appargs: Haengt alle Argumente von *e an die Argumentliste an
inline HlBaseList *HlBaseList::appargs(HlBaseList *e)
{
    for (unsigned int i = 0, n = e->Length(); i < n; i++)
        apparg(e->mArg[i]);

    return this;
}

// delarg: loescht Element *e aus der Argumentliste
inline HlBaseList *HlBaseList::delarg(HlBaseList *e)
{
    BaselistVector::iterator where = find(mArg.begin(), mArg.end(), e);
    if (where != mArg.end())
        mArg.erase(where);

    delete e;

    return this;
}

// copy: erstellt eine rekursive Kopie der Struktur this und gibt diese zurück
inline HlBaseList *HlBaseList::copy()
{
    HlBaseList *h = new HlBaseList(getsym());
    h->setDoubleVal(getDoubleVal());

    for (unsigned int i = 0, n = Length(); i < n; i++)
        h->apparg(mArg[i]->copy());

    return h;
}

#endif
