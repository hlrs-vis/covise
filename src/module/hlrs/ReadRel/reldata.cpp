/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "reldata.hpp"
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
//# include <time.h>
#ifndef _WIN32
#include "usefullroutines.h"
#endif
#include "Globals.hpp"

extern const char *formatstring[];
// Deklaration von WriteText()
int WriteText(VERBOSE, const char *, ...);

// =======================================================
// Konstanten
// =======================================================
enum
{
    PN_IGNORE = 0,
    PN_X = 1,
    PN_Y = 2,
    PN_Z = 3,
    PN_NODEIDX = 4,
    PN_EDGEIDX = 5,
    PN_FACEIDX = 6,
    PN_CELLIDX = 7,
    PN_RED = 8,
    PN_GREEN = 9,
    PN_BLUE = 10,
    PN_DENSITY = 11,
    PN_GROUP = 12,
    PN_INDEX = 13,
    PN_ELEM = 14,
    ANZ_PN
};
enum
{
    PT_FP,
    PT_INDEX,
    PT_INT,
    ANZ_PT
};

const int prop_tab[ANZ_PN] = {
    PT_INDEX, PT_FP, PT_FP, PT_FP, PT_INDEX, PT_INDEX, PT_INDEX,
    PT_INDEX, PT_INT, PT_INT, PT_INT, PT_FP, PT_INT,
    PT_INDEX, PT_INT
};
const char *property_typenames[ANZ_PT] = {
    "fp", "index", "int",
};
const char *elemname[ANZ_ELEM] = { "Knoten", "Kante", "Flaeche", "Zelle", "kein Element", "alle Elemente" };

// =======================================================
// Konstruktor
// =======================================================
RELDATA::RELDATA()
{
    anz_eknoten = -1; // alles erst mal auf "undefiniert = -1
    anz_ekanten = -1;
    anz_eflaechen = -1;
    anz_ezellen = -1;
    eknoten = NULL;
    ekante = NULL;
    eflaeche = NULL;
    ezelle = NULL;
}

// =======================================================
// Destruktor
// =======================================================
RELDATA::~RELDATA()
{
    int i, j;
    int anz;
    ELEMENT *acttab;
    ECHAIN *ecact;
    // alle Daten loeschen

    for (i = 0; i < 4; ++i)
    {
        switch (i)
        {
        case ELEM_NODE:
            acttab = eknoten;
            anz = anz_eknoten;
            break;

        case ELEM_EDGE:
            acttab = ekante;
            anz = anz_ekanten;
            break;

        case ELEM_FACE:
            acttab = eflaeche;
            anz = anz_eflaechen;
            break;

        case ELEM_CELL:
            acttab = ezelle;
            anz = anz_ezellen;
            break;
        }
        for (j = 0; j < anz; ++j)
        {
            // Schleife ueber alle Elemente
            if (acttab[j].d_anz != 0)
                delete acttab[j].data;
            if (acttab[j].e_anz != 0)
                delete acttab[j].element;

            if (acttab[j].up != NULL)
            {
                while (acttab[j].up != NULL)
                {
                    ecact = acttab[j].up->next;
                    delete acttab[j].up;
                    acttab[j].up = ecact;
                }
            }
        }
        if (acttab != NULL)
            delete acttab;
    }
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Service Routinen
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// =======================================================
// DeleteUnused
// =======================================================
// Lösch alle Elemente, die nicht als "FLAG_USED" markiert sind
int RELDATA::DeleteUnused(ELEMENTTYPE etype)
{
    const char *funcname = "RELDATA::DeleteUnused";

    //Generates C4189
    //int anz=0;
    int anz_elem;

    int i, delcount = 0, savecount = 0, cnt;
    ELEMENT *acttab, *newtab;
    clock_t start, end;

    start = clock();
    switch (etype)
    {
    case ELEM_NODE:
        anz_elem = anz_eknoten;
        acttab = eknoten;
        break;

    case ELEM_EDGE:
        anz_elem = anz_ekanten;
        acttab = ekante;
        break;

    case ELEM_FACE:
        anz_elem = anz_eflaechen;
        acttab = eflaeche;
        break;

    case ELEM_CELL:
        anz_elem = anz_ezellen;
        acttab = ezelle;
        break;

    case ELEM_ALL:
        DeleteUnused(ELEM_NODE);
        DeleteUnused(ELEM_EDGE);
        DeleteUnused(ELEM_FACE);
        //DeleteUnused(ELEM_CELL);
        return (0);

    default:
        WriteText(VER_DEBUG, "%s meldet: nicht implementiert fuer Elementtyp %d\n", funcname, etype);
        return (0);
    }
    // Vorgehen:
    // alle Elemente ohne FLAG_USED werden mit dem ID MAXID+1 belegt, nach hinten sortiert
    // und die Variable der Elementzahl verringert
    savecount = GetFlagUsage(FLAG_USED, etype);
    if (savecount == 0)
    {
        WriteText(VER_NORMAL, "ACHTUNG: es werden alle Elemente des Typs %d geloescht!\n", etype);
        //    return(0);
    }
    newtab = new ELEMENT[savecount];
    memset(newtab, 0, sizeof(ELEMENT) * savecount);
    cnt = 0;
    delcount = 0;
    savecount = 0;
    for (i = 0; i < anz_elem; ++i)
    {
        if (!(acttab[i].flags | FLAG_DISCON))
        {
            WriteText(VER_NORMAL, "%s meldet: Element %d nicht disconnected!\n", funcname, acttab[i].id);
        }
        if (acttab[i].flags & FLAG_USED)
        {
            CopyElement(&newtab[cnt++], &acttab[i]);
            savecount++;
        }
        else
        {
            // Teile des Feldes Löschen
            DeleteEChain(acttab[i].up, FALSE);
            delete acttab[i].data;
            delete acttab[i].element;
            delcount++;
        }
    }

    delete acttab;

    // Tabelle wieder einsortieren
    switch (etype)
    {
    case ELEM_NODE:
        eknoten = newtab;
        anz_eknoten = savecount;
        break;

    case ELEM_EDGE:
        ekante = newtab;
        anz_ekanten = savecount;
        break;

    case ELEM_FACE:
        eflaeche = newtab;
        anz_eflaechen = savecount;
        break;

    case ELEM_CELL:
        ezelle = newtab;
        anz_ezellen = savecount;
        break;
    default:
        break;
    }
    end = clock();
    WriteText(VER_MAX, "%d Elemente vom Typ %d geloescht, %d Elemente gehalten [%8.4f sec]\n", delcount, etype, savecount, (double)(end - start) / CLOCKS_PER_SEC);
    return (delcount);
}

// =======================================================
// BubbleSort (integer Version)
// =======================================================
// Sortiert ein bestehendes Elementfeld (ohne Neuerstellung)
// liefert die Anzahl der benötigten Durchläuft zurück
// Kann das VZ der Elemente ignorieren, ohne es zu löschen
int RELDATA::BubbleSort(int *feld, int size, BOOL ignorevz)
{
    //Generates C4189
    //const char *funcname="RELDATA::BubbleSort(integer)";
    int i, runs = 0, tmp;
    BOOL changed = TRUE;

    if (ignorevz)
    {
        while (changed)
        {
            changed = FALSE;
            for (i = 0; i < size - 1 - runs; ++i)
            {
                if (abs(feld[i]) > abs(feld[i + 1]))
                {
                    tmp = feld[i];
                    feld[i] = feld[i + 1];
                    feld[i + 1] = tmp;
                    changed = TRUE;
                }
            }
            runs++;
        }
    }
    else
    {
        while (changed)
        {
            changed = FALSE;
            for (i = 0; i < size - 1 - runs; ++i)
            {
                if (feld[i] > feld[i + 1])
                {
                    tmp = feld[i];
                    feld[i] = feld[i + 1];
                    feld[i + 1] = tmp;
                    changed = TRUE;
                }
            }
            runs++;
        }
    }

    return (runs);
}

// =======================================================
// BubbleSort
// =======================================================
// Sortiert ein bestehendes Elementfeld (ohne Neuerstellung)
// liefert die Anzahl der benötigten Durchläuft zurück
int RELDATA::BubbleSort(ELEMENTTYPE etype)
{
    const char *funcname = "RELDATA::BubbleSort";
    int runs = 0, anz_elem, cnt;
    BOOL changed = TRUE;
    ELEMENT dummy, *acttab;

    switch (etype)
    {
    case ELEM_NODE:
        anz_elem = anz_eknoten;
        acttab = eknoten;
        break;

    case ELEM_EDGE:
        anz_elem = anz_ekanten;
        acttab = ekante;
        break;

    case ELEM_FACE:
        anz_elem = anz_eflaechen;
        acttab = eflaeche;
        break;

    case ELEM_CELL:
        anz_elem = anz_ezellen;
        acttab = ezelle;
        break;

    default:
        WriteText(VER_NORMAL, "%s meldet: nicht implementiert fuer Elementtyp %d\n", funcname, etype);
        return (0);
    }

    runs = 0;
    while (changed)
    {
        //    printf("%d ",runs);
        changed = FALSE;
        cnt = 0;
        // Das größte Element wird nach hinten durchgereicht.
        // Die kleinen wandern langsam nach vorne
        runs++;
        for (cnt = 0; cnt < anz_elem - 1; ++cnt)
        {
            if (abs(acttab[cnt].id) > abs(acttab[cnt + 1].id))
            {
                memset(&dummy, 0, sizeof(ELEMENT));
                CopyElement(&dummy, &acttab[cnt]);
                CopyElement(&acttab[cnt], &acttab[cnt + 1]);
                CopyElement(&acttab[cnt + 1], &dummy);
                changed = TRUE;
            }
        }
        // die letzten "runs" Elemente sind auf alle Fälle sortiert
    }
    //  printf("\n");
    return (runs);
}

// =======================================================
// MergeSort
// =======================================================
// Sortiert zwei Felder und
// Fügt zwei vorsortierte Teilfelder in ein erstelltes Zielfeld
ELEMENT *RELDATA::MergeSort(ELEMENT *source, int size)
{
    //  const char *funcname="RELDATA::MergeSort";
    int scount1 = 0, scount2 = 0;
    int size1, size2;
    ELEMENT *dest;
    ELEMENT *source1, *source2;

    /* 
     if (source==NULL || size ==0)
     {
       if (Globals.verbose!=VER_NONE)
       {
         printf("%s meldet: mindestens einer der Quelloperanten ist null!\n",funcname);
       }
       return(NULL);
     }
   */

    // Sort-Teil
    if (size == 1)
    {
        // Abbruchkriterium, aktuelles Feld besteht nur aus einem Element
        return (CopyElement(NULL, source));
    }

    size1 = size / 2;
    size2 = size - size1;
    source1 = MergeSort(&source[0], size1);
    source2 = MergeSort(&source[size1], size2);

    // Nicht letzte Element:
    dest = new ELEMENT[size];
    memset(dest, 0, sizeof(ELEMENT) * size);
    // Merge-Teil
    // Jetzt werden die beiden Ergebniss-Felder zusammengewürfelt
    scount1 = scount2 = 0;
    while (scount1 + scount2 < size)
    {
        if (abs(source1[scount1].id) < abs(source2[scount2].id))
        {
            CopyElement(&dest[scount1 + scount2], &source1[scount1]);
            scount1++;
        }
        else
        {
            CopyElement(&dest[scount1 + scount2], &source2[scount2]);
            scount2++;
        }
        // Schlußtreffer
        if (scount1 == size1)
        {
            // restliche Elemente von source2 kopieren
            while (scount2 < size2)
            {
                CopyElement(&dest[scount1 + scount2], &source2[scount2]);
                scount2++;
            }
        }
        else if (scount2 == size2)
        {
            // restliche Elemente von source2 kopieren
            while (scount1 < size1)
            {
                CopyElement(&dest[scount1 + scount2], &source1[scount1]);
                scount1++;
            }
        }
    }
    // die beiden Teilfelder können jetzt gelöscht werden
    delete source1;
    delete source2;
    // fertig!
    return (dest);
}

// ====================================
// MergeSort (NODELIST Version)
// ====================================

// NODELIST-Version
NODELIST *RELDATA::MergeSort(NODELIST *source, int size, int typ)
{
    int scount1 = 0, scount2 = 0;
    int size1, size2;
    NODELIST *dest;
    NODELIST *source1, *source2;

    // Sort-Teil
    if (size == 1)
    {
        // Abbruchkriterium, aktuelles Feld besteht nur aus einem Element
        dest = new NODELIST;
        memcpy(dest, source, sizeof(NODELIST));
        return (dest);
    }

    size1 = size / 2;
    size2 = size - size1;
    source1 = MergeSort(&source[0], size1, typ);
    source2 = MergeSort(&source[size1], size2, typ);

    // Nicht letzte Element:
    dest = new NODELIST[size];

    // Merge-Teil
    // Jetzt werden die beiden Ergebniss-Felder zusammengewürfelt
    scount1 = scount2 = 0;

    while (scount1 + scount2 < size)
    {
        switch (typ)
        {
        case X:
            if (source1[scount1].x < source2[scount2].x)
            {
                memcpy(&dest[scount1 + scount2], &source1[scount1], sizeof(NODELIST));
                scount1++;
            }
            else
            {
                memcpy(&dest[scount1 + scount2], &source2[scount2], sizeof(NODELIST));
                scount2++;
            }
            break;

        case Y:
            if (source1[scount1].y < source2[scount2].y)
            {
                memcpy(&dest[scount1 + scount2], &source1[scount1], sizeof(NODELIST));
                scount1++;
            }
            else
            {
                memcpy(&dest[scount1 + scount2], &source2[scount2], sizeof(NODELIST));
                scount2++;
            }
            break;

        case Z:
            if (source1[scount1].z < source2[scount2].z)
            {
                memcpy(&dest[scount1 + scount2], &source1[scount1], sizeof(NODELIST));
                scount1++;
            }
            else
            {
                memcpy(&dest[scount1 + scount2], &source2[scount2], sizeof(NODELIST));
                scount2++;
            }
            break;
        }
        // Schlußtreffer
        if (scount1 == size1)
        {
            // restliche Elemente von source2 kopieren
            while (scount2 < size2)
            {
                memcpy(&dest[scount1 + scount2], &source2[scount2], sizeof(NODELIST));
                scount2++;
            }
        }
        else if (scount2 == size2)
        {
            // restliche Elemente von source2 kopieren
            while (scount1 < size1)
            {
                memcpy(&dest[scount1 + scount2], &source1[scount1], sizeof(NODELIST));
                scount1++;
            }
        }
    }

    // die beiden Teilfelder können jetzt gelöscht werden
    delete source1;
    delete source2;
    // fertig!
    return (dest);
}

// Integer Version
// bei ignoresgn=TRUE werden alle Vorzeichen ignoriert, jedoch im sortierten
// Feld dargestellt
int *RELDATA::MergeSort(int *source, int size, BOOL ignoresgn)
{
    int scount1 = 0, scount2 = 0;
    int size1, size2;
    int *dest;
    int *source1, *source2;

    // Sort-Teil
    if (size == 1)
    {
        // Abbruchkriterium, aktuelles Feld besteht nur aus einem Element
        dest = new int;
        dest[0] = source[0];
        //    delete source;
        return (dest);
    }

    /*
     size1 = size/2;
     tmp1 = new int[size1];
     memcpy(tmp1,source,sizeof(int)*size1);
     size2 = size-size1;
     tmp2 = new int[size2];
     memcpy(tmp2,&source[size1],sizeof(int)*size2);
     delete source;
     source1 = MergeSort(tmp1,size1,ignoresgn);
     source2 = MergeSort(tmp2,size2,ignoresgn);
   */
    size1 = size / 2;
    size2 = size - size1;
    source1 = MergeSort(&source[0], size1, ignoresgn);
    source2 = MergeSort(&source[size1], size2, ignoresgn);

    // Nicht letzte Element:
    dest = new int[size];
    memset(dest, 0, sizeof(int) * size);
    // Merge-Teil
    // Jetzt werden die beiden Ergebniss-Felder zusammengewürfelt
    scount1 = scount2 = 0;

    while (scount1 + scount2 < size)
    {
        if (ignoresgn)
        {
            if (abs(source1[scount1]) < abs(source2[scount2]))
            {
                dest[scount1 + scount2] = source1[scount1];
                scount1++;
            }
            else
            {
                dest[scount1 + scount2] = source2[scount2];
                scount2++;
            }
        }
        else
        {
            if (source1[scount1] < source2[scount2])
            {
                dest[scount1 + scount2] = source1[scount1];
                scount1++;
            }
            else
            {
                dest[scount1 + scount2] = source2[scount2];
                scount2++;
            }
        }
        // Schlußtreffer
        if (scount1 == size1)
        {
            // restliche Elemente von source2 kopieren
            while (scount2 < size2)
            {
                dest[scount1 + scount2] = source2[scount2];
                scount2++;
            }
        }
        else if (scount2 == size2)
        {
            // restliche Elemente von source2 kopieren
            while (scount1 < size1)
            {
                dest[scount1 + scount2] = source1[scount1];
                scount1++;
            }
        }
    }

    // die beiden Teilfelder können jetzt gelöscht werden
    delete source1;
    delete source2;
    // fertig!
    return (dest);
}

// =======================================================
// SortEdges
// =======================================================
// Sortiert eine Liste mit IDs von Kanten
// Dabei wird die Liste stets in die Richtung sortiert,
// in die das erste Element zeigt (CLOCKWISE) oder in dessen
// Gegenrichtung.
// Bei einem Feld [-3,6,8,9], CLOCKWISE sollte das selbe
// Eregebnis heraus kommen wie bei [3,6,8,9] COUNTERCLOCKWISE
// liefert die Sortierzeit zurück
// Über den newedge-Pointer können Daten von frisch erzeugten
// Kanten mitgeliefert werden, in denen ebenfalls nach
// dem ID gesucht wird.
int RELDATA::SortEdges(int *idx, int size, SORTTYPE stype, ECHAIN *newedge)
{
    const char *funcname = "RELDATA::SortEdges";
    int *p, i, k, d[2], dummy, *help;
    int index, ret = 0;
    ECHAIN *actedge;
    ELEMENT *kante;
    BOOL found;

    switch (stype)
    {
    case SORT_CLOCKWISE:
        p = new int[size * 2];
        help = new int[size * 2];
        for (i = 0; i < size; ++i)
        {
            kante = GetElementPtr(idx[i], ELEM_EDGE);
            if (kante == NULL)
            {
                actedge = newedge; // wenn newedge NULL ist, springt das hier drüber
                while (kante == NULL && actedge != NULL)
                {
                    if (actedge->id == abs(idx[i]))
                    {
                        // Kante in neuer Liste gefunden!
                        // ACHTUNG: Diese Daten sind noch nicht Connected!!
                        kante = actedge->element;
                        WriteText(VER_DEBUG, "Kante %d in Liste der neuen Kanten gefunden!\n", kante->id);
                    }
                    actedge = actedge->next;
                }
            }
            // Wenn die kante auch da nicht gefunden wurde: Fehler
            if (kante == NULL)
            {
                WriteText(VER_DEBUG, "%s meldet: Kante %d nicht gefunden!\n", funcname, idx[i]);
                delete p;
                delete help;
                return (1);
            }
            if (idx[i] > 0)
            {
                p[i * 2] = kante->element[VON].id;
                p[i * 2 + 1] = kante->element[NACH].id;
            }
            else
            {
                p[i * 2] = -1 * kante->element[NACH].id;
                p[i * 2 + 1] = -1 * kante->element[VON].id;
            }
            help[i * 2] = abs(p[i * 2]);
            help[i * 2 + 1] = abs(p[i * 2 + 1]);
        }
        // Nach Start/Endelement suchen
        // Alle doppelten komplett löschen, start/ziel-Punkte bleiben übrig
        found = TRUE;
        for (i = 0; i < (size - 1) * 2 && found == TRUE; ++i)
        {
            found = FALSE;
            // Ganzen Array durchsuchen!
            for (k = 0; k < size * 2 && found == FALSE; ++k)
            {
                // sich selber sollte er nicht finden ;-)
                if (i != k && help[i] == help[k])
                {
                    found = TRUE;
                }
            }
            if (found == FALSE)
            {
                index = i / 2;
                // Einzelgänger gefunden, den nach vorne packen
                WriteText(VER_DEBUG, "%s meldet: Ein Kantenzug ist nicht geschlossen.\n", funcname);
                ret = 2;
                if (index != 0)
                {
                    d[0] = p[index * 2];
                    d[1] = p[index * 2 + 1];
                    p[index * 2] = p[0];
                    p[index * 2 + 1] = p[1];
                    p[0] = d[0];
                    p[1] = d[1];
                    // Index umsortieren
                    dummy = idx[0];
                    idx[0] = idx[index];
                    idx[index] = dummy;
                    WriteText(VER_DEBUG, "%s: Vertausche ID %d und ID %d\n", funcname, idx[0], idx[i]);
                }
                if (i % 2 == 1)
                {
                    // Das Element zeigt zusätzlich in die falsche Richtung
                    idx[0] *= -1;
                    dummy = p[0];
                    p[0] = -p[1];
                    p[1] = -dummy;
                }
            }
        }
        delete help;

        // Reihenfolge der Elemente korrigieren
        for (i = 0; i < size - 1; ++i)
        {
            // wir suchen nach einem Anschluß an das Element p[2*i+1]
            found = FALSE;
            for (k = i + 1; k < size && !found; ++k)
            {
                if (p[2 * k] == -p[2 * i + 1]) // von -> NACH
                {
                    if (k != i + 1) // Ist sowieso schon in richtiger Reihenfolge
                    {
                        // Gefunden! dieses Element tauschen mit dem Element i+1;
                        d[0] = p[2 * (i + 1)];
                        d[1] = p[2 * (i + 1) + 1];
                        p[2 * (i + 1)] = p[2 * k];
                        p[2 * (i + 1) + 1] = p[2 * k + 1];
                        p[2 * k] = d[0];
                        p[2 * k + 1] = d[1];
                        // jetzt fuer indices
                        dummy = idx[i + 1];
                        idx[i + 1] = idx[k];
                        idx[k] = dummy;
                    }
                    found = TRUE;
                }
                else if (p[2 * k + 1] == p[2 * i + 1]) // VON->VON, Element umdrehen
                {
                    if (k == i + 1) // richtige Reihenfolge, aber falsche von->nach Richtung
                    {
                        dummy = p[2 * k];
                        p[2 * k] = -p[2 * k + 1];
                        p[2 * k + 1] = -dummy;
                        // Index negieren
                        idx[k] *= -1;
                    }
                    else // Falsche Reihenfolge und falsche von->nach Richtung
                    {
                        d[0] = p[2 * (i + 1)];
                        d[1] = p[2 * (i + 1) + 1];
                        p[2 * (i + 1)] = -p[2 * k + 1];
                        p[2 * (i + 1) + 1] = -p[2 * k];
                        p[2 * k] = d[0];
                        p[2 * k + 1] = d[1];
                        // jetzt fuer indices
                        dummy = idx[i + 1];
                        idx[i + 1] = -idx[k];
                        idx[k] = dummy;
                    }
                    found = TRUE;
                }
            }
            if (!found)
            {
                WriteText(VER_DEBUG, "%s meldet: Fehler, Kante kann nicht eingeordnet werden: %d\n", funcname, idx[i]);
                delete p;
                return (3);
            }
        }
        delete p;
        break;

    case SORT_COUNTERCLOCKWISE:
        SortEdges(idx, size, SORT_CLOCKWISE, newedge);
        // jetzt noch umdrehen und Richtungen ändern
        p = new int[size];
        for (i = 0; i < size; ++i)
            p[i] = -idx[size - i - 1];
        // Jetzt zurück kopieren
        memcpy(idx, p, sizeof(int) * size);
        delete p;
        break;

    default:
        WriteText(VER_DEBUG, "%s meldet: nicht implementiert fuer Sorttype: %d\n", funcname, stype);
    }
    return (ret);
}

// =======================================================
// SortElement
// =======================================================
// Sortiert ein Elementfeld nach dem ID mit einem bestimmten Verfahren
// Liefert die Laufzeit der Sortierung als Cloclticks zurück
// ACHTUNG:: Feld muß diconnected sein!
clock_t RELDATA::SortElement(ELEMENTTYPE etype, SORTTYPE stype)
{
    const char *funcname = "RELDATA::SortElement";
    ELEMENT *acttab, *newtab;
    int anz_elem;
    clock_t start, end;

    switch (etype)
    {
    case ELEM_NODE:
        anz_elem = anz_eknoten;
        acttab = eknoten;
        break;

    case ELEM_EDGE:
        anz_elem = anz_ekanten;
        acttab = ekante;
        break;

    case ELEM_FACE:
        anz_elem = anz_eflaechen;
        acttab = eflaeche;
        break;

    case ELEM_CELL:
        anz_elem = anz_ezellen;
        acttab = ezelle;
        break;

    default:
        if (Globals.verbose != VER_NONE)
            printf("%s meldet: nicht implementiert fuer Elementtyp %d\n", funcname, etype);
        return (0);
    }

    switch (stype)
    {
    case SORT_MERGE:
        start = clock();
        newtab = MergeSort(acttab, anz_elem);
        end = clock();

        if (newtab != NULL)
            delete acttab;
        else
        {
            if (Globals.verbose != VER_NONE)
                printf("%s meldet: Sortierung mit Sorttype %d fehlgeschlagen\n", funcname, stype);
            newtab = acttab;
        }
        break;

    case SORT_BUBBLE:
        start = clock();
        BubbleSort(etype);
        end = clock();
        return (0);

    default:
        if (Globals.verbose != VER_NONE)
            printf("%s meldet: unbekannter Sortiertyp %d\n", funcname, stype);
        return (0);
    }

    // Tabelle wieder einsortieren
    switch (etype)
    {
    case ELEM_NODE:
        eknoten = newtab;
        break;

    case ELEM_EDGE:
        ekante = newtab;
        break;

    case ELEM_FACE:
        eflaeche = newtab;
        break;

    case ELEM_CELL:
        ezelle = newtab;
        break;
    default:
        break;
    }
    return (end - start);
}

// =======================================================
// GetFlagUsage
// =======================================================
// Liefert die Anzahl der Elemente zurück, die die
// angegebene FLAG-Kombination benutzen
// mit GetFlagUsage(FLAG_USED,ELEM_FACE) würde man
// z.B. die Anzahl der benutzen Flaechen bekommen
int RELDATA::GetFlagUsage(int flags, ELEMENTTYPE etype)
{
    const char *funcname = "RELDATA::GetFlagUsage";
    int sum = 0, i;
    int anz_elem;
    ELEMENT *acttab;

    switch (etype)
    {
    case ELEM_NODE:
        anz_elem = anz_eknoten;
        acttab = eknoten;
        break;

    case ELEM_EDGE:
        anz_elem = anz_ekanten;
        acttab = ekante;
        break;

    case ELEM_FACE:
        anz_elem = anz_eflaechen;
        acttab = eflaeche;
        break;

    case ELEM_CELL:
        anz_elem = anz_ezellen;
        acttab = ezelle;
        break;

    default:
        if (Globals.verbose != VER_NONE)
            printf("%s meldet: nicht implementiert fuer Elementtyp %d\n", funcname, etype);
        return (0);
    }

    for (i = 0; i < anz_elem; ++i)
    {
        if (acttab[i].flags & flags)
            sum++;
    }

    return (sum);
}

// =======================================================
// GetBalancePoint
// =======================================================
// liefert den Ortsvektor zum Schwerpunkt einer Fläche zurück
// im Betrag steht der Flächeninhalt der Fläche
double *RELDATA::GetBalancePoint(ELEMENT *elem)
{
    const char *funcname = "RELDATA::GetBalancePoint";
    double *mp;
    double *ret;
    ELEMENT *knoten;
    double *avec, *bvec, x, y, z;
    int *nodes, size, i;

    switch (elem->typ)
    {
    case ELEM_FACE:
        break;

    default:
        WriteText(VER_DEBUG, "%s meldet: Nicht implementiert fuer Elementtyp %d!\n", funcname, elem->typ);
        return (NULL);
    }

    ret = new double[4];
    mp = GetCenterPoint(elem);
    size = elem->e_anz;
    // zerlegung der Fläche in Dreiecke
    nodes = GetNodes(elem);
    // jetzt die erste Kante in dieser Liste mit dem ersten Punkt überschreiben, wegen der
    // folgenden Schleife (ist einfacher)
    nodes[size] = nodes[0];
    ret[X] = mp[X]; // Ortsvektor des Mittelpunkts ist Referenzpunkt
    ret[Y] = mp[Y];
    ret[Z] = mp[Z];
    ret[B] = 0; // Gibt den Gesamtflächeninhalt an (für Gewichtungen)
    // bvec klar machen
    knoten = GetElementPtr(nodes[0], ELEM_NODE);
    bvec = new double[3];
    bvec[X] = mp[X] - knoten->data[X].value;
    bvec[Y] = mp[Y] - knoten->data[Y].value;
    bvec[Z] = mp[Z] - knoten->data[Z].value;
    for (i = 0; i < size; ++i)
    {
        // Vektoren weiter reichen
        avec = bvec;
        // Neuen Vektor erstellen
        knoten = GetElementPtr(nodes[i + 1], ELEM_NODE);
        // Berechnen
        bvec = new double[3];
        bvec[X] = mp[X] - knoten->data[X].value;
        bvec[Y] = mp[Y] - knoten->data[Y].value;
        bvec[Z] = mp[Z] - knoten->data[Z].value;
        // Schwerpunkt liegt jetzt bei 1/3(avec+bvec)
        ret[X] += (avec[X] + bvec[X]) / 3;
        ret[Y] += (avec[Y] + bvec[Y]) / 3;
        ret[Z] += (avec[Z] + bvec[Z]) / 3;
        // Flächeninhalt ist 1/2 Kreuzprodukt
        x = (avec[Y] * bvec[Z] - avec[Z] * bvec[Y]);
        y = (avec[Z] * bvec[X] - avec[X] * bvec[Z]);
        z = (avec[X] * bvec[Y] - avec[Y] * bvec[X]);
        ret[B] += sqrt(x * x + y * y + z * z) / 2;
        // avec kann jetzt gelöscht werden
        delete avec;
    }
    // Bvec ist auch alteisen
    delete bvec;
    // Mittelpunkt wird uninteressant...
    delete mp;
    // that's all Folks!
    return (ret);
}

// =======================================================
// GetCenterPoint
// =======================================================
double *RELDATA::GetCenterPoint(ELEMENT *elem)
{
    const char *funcname = "RELDATA::GetCenterPoint";
    double *ret;
    ELEMENT *knoten;
    int i, *node;

    ret = new double[3];

    switch (elem->typ)
    {
    case ELEM_EDGE:
        ret[X] = elem->element[NACH].ptr->data[X].value - elem->element[VON].ptr->data[X].value;
        ret[Y] = elem->element[NACH].ptr->data[X].value - elem->element[VON].ptr->data[X].value;
        ret[Z] = elem->element[NACH].ptr->data[X].value - elem->element[VON].ptr->data[X].value;
        break;

    case ELEM_FACE:
        node = GetNodes(elem);
        ret[X] = 0;
        ret[Y] = 0;
        ret[Z] = 0;
        for (i = 0; i < elem->e_anz; ++i)
        {
            knoten = GetElementPtr(node[i], ELEM_NODE);
            ret[X] += knoten->data[X].value;
            ret[Y] += knoten->data[Y].value;
            ret[Z] += knoten->data[Z].value;
        }
        delete node;

        ret[X] /= elem->e_anz;
        ret[Y] /= elem->e_anz;
        ret[Z] /= elem->e_anz;
        break;

    default:
        WriteText(VER_DEBUG, "%s meldet: Nicht implementiert fuer Elementtyp %d.\n", funcname, elem->typ);
        return (NULL);
    }
    return (ret);
}

// =======================================================
// GetFaceNormal3
// =======================================================
// Version für nichtebene Flächen
// Gilt für beliebige Polygone
double *RELDATA::GetFaceNormal(ELEMENT *elem)
{
    //Generates C4189
    //const char* funcname="RELDATA::GetFaceNormal";
    double *ret = NULL;
    int i, size;
    double avec[3], bvec[3], *mp, *von, *nach, *tmp;

    ret = new double[4];

    ret[X] = 0;
    ret[Y] = 0;
    ret[Z] = 0;
    ret[B] = 0;

    size = elem->e_anz;
    // Schritt 1: Mittelpunkt besorgen
    mp = GetCenterPoint(elem);

    if (size == 3)
    {
        // einfache Methode verwenden: Kante1 x Kante2
        nach = GetVector(elem->element[0].ptr);
        von = GetVector(elem->element[1].ptr);
        if (elem->element[0].id > 0) // dieser Vektor muss gegen den Drehsinn zeigen
        {
            nach[X] *= -1;
            nach[Y] *= -1;
            nach[Z] *= -1;
        }
        if (elem->element[1].id < 0) // dieser Vector muss in den Drehsinn zeigen
        {
            von[X] *= -1;
            von[Y] *= -1;
            von[Z] *= -1;
        }
        ret[X] = (nach[Y] * von[Z] - nach[Z] * von[Y]);
        ret[Y] = (nach[Z] * von[X] - nach[X] * von[Z]);
        ret[Z] = (nach[X] * von[Y] - nach[Y] * von[X]);
        delete von;
        delete nach;
    }
    else
    {
        for (i = 0; i < size; ++i)
        {
            // VON-Punkt der aktuellen Kante
            von = GetVector(elem->element[i].ptr->element[VON].ptr);
            // NACH-Punkt der aktuellen Kante
            nach = GetVector(elem->element[i].ptr->element[NACH].ptr);
            if (elem->element[i].id < 0) // im positiven Fall von und nach vertauschen
            {
                tmp = von;
                von = nach;
                nach = tmp;
            }

            // Vektor vom Mittelpunkt zum Startpunkt der Kante
            avec[X] = nach[X] - mp[X];
            avec[Y] = nach[Y] - mp[Y];
            avec[Z] = nach[Z] - mp[Z];

            // Vektor vom Mittelpunkt zum Endpunkt der Kante
            bvec[X] = von[X] - mp[X];
            bvec[Y] = von[Y] - mp[Y];
            bvec[Z] = von[Z] - mp[Z];

            // jetzt a x b oder nach x von rechnen
            ret[X] += (avec[Y] * bvec[Z] - avec[Z] * bvec[Y]);
            ret[Y] += (avec[Z] * bvec[X] - avec[X] * bvec[Z]);
            ret[Z] += (avec[X] * bvec[Y] - avec[Y] * bvec[X]);

            // alten Startpunkt löschen
            delete von;
            delete nach;
        }
        delete mp;
        ret[X] /= size;
        ret[Y] /= size;
        ret[Z] /= size;
    }
    ret[B] = sqrt(ret[X] * ret[X] + ret[Y] * ret[Y] + ret[Z] * ret[Z]);

    return (ret);
}

/*
// =======================================================
// GetFaceNormale2
// =======================================================
// Version für nichtebene Flächen
// Es wird eine Normale an jedem Eckpunkt der Fläche erzeugt
// Diese Normalen werden addiert und durch die Anzahl
// der Ecken geteilt. Für Drei- und Vierecke ist der Betrag
// des Vektors (an 4. Stelle) auch der doppelte Flächeninhalt.
double *RELDATA::GetFaceNormal(ELEMENT* elem)
{
const char* funcname="RELDATA::GetFaceNormale";
double *ret=NULL;
int i,size;
double *avec,*bvec;

ret = new double[4];

ret[X]=0;
ret[Y]=0;
ret[Z]=0;
ret[B]=0;

size = elem->e_anz;
// erste mal den unteren (b) Vektor besorgen
bvec = GetVector(elem->element[size-1].ptr);
if (elem->element[size-1].id <0)
{
bvec[X]*=-1;
bvec[Y]*=-1;
bvec[Z]*=-1;
}
for (i=0;i<size;++i)
{
avec = bvec;
// alten A-Vektor umdrehen
avec[X]*=-1;
avec[Y]*=-1;
avec[Z]*=-1;
bvec = GetVector(elem->element[i].ptr);
// Vektor in richtige Richtung drehen
if (elem->element[i].id <0)
{
bvec[X]*=-1;
bvec[Y]*=-1;
bvec[Z]*=-1;
}
// jetzt a x b rechnen
ret[X] += (avec[Y]*bvec[Z]-avec[Z]*bvec[Y]);
ret[Y] += (avec[Z]*bvec[X]-avec[X]*bvec[Z]);
ret[Z] += (avec[X]*bvec[Y]-avec[Y]*bvec[X]);
// alten avec löschen
delete avec;
}
delete bvec;
ret[X]/=size; // Durch die Anzahl der verwendeten Kanten teilen
ret[Y]/=size;
ret[Z]/=size;
ret[B] = sqrt(ret[X]*ret[X]+ret[Y]*ret[Y]+ret[Z]*ret[Z]);

return(ret);
}
*/

/*
double *RELDATA::GetFaceNormal(ELEMENT *actelem)
{
  const char* funcname="RELDATA::GetFaceNormal";
  double *ret = new double[4];
  double *vec1,*vec2;

  if (actelem->typ!=ELEM_FACE)
  {
    if (Globals.verbose!=VER_NONE)
    {
printf("%s meldet: Element (Typs %d, id %d) iste keine Flaeche!\n",funcname,actelem->typ, actelem->id);
return(NULL);
}
}

// Orientierung beachten:
// Kante muß vom Punkt "weg" führen, negative ID also umdrehen
vec1 = GetVector(actelem->element[0].ptr);
if (actelem->element[0].id <0 )
{
vec1[X]*=-1;
vec1[Y]*=-1;
vec1[Z]*=-1;
}
vec2 = GetVector(actelem->element[1].ptr);
if (actelem->element[1].id < 0 )
{
vec2[X]*=-1;
vec2[Y]*=-1;
vec2[Z]*=-1;
}

// Normale erstellen
ret[X] =   vec1[Y]*vec2[Z]-vec1[Z]*vec2[Y];
ret[Y] =   vec1[Z]*vec2[X]-vec1[X]*vec2[Z];
ret[Z] =   vec1[X]*vec2[Y]-vec1[Y]*vec2[X];
ret[B] =   sqrt(ret[X]*ret[X]+ret[Y]*ret[Y]+ret[Z]*ret[Z]);
return(ret);
}
*/

int RELDATA::CalculatePhongNormals()
{
    //Generates C4189
    //const char *funcname="RELDATA::CalculatePhongNormals";
    int i;
    ECHAIN *root, *act;
    int numfaces;
    double vector[3] = { 0, 0, 0 },
           *vec, norm;
    PTR_DATA *data;

    // Phong-Normal an jedem Punkt errechnen
    for (i = 0; i < anz_eknoten; ++i)
    {
        if (eknoten[i].flags & FLAG_USED)
        {
            // erst mal alle angrenzenden Flächen ermitteln
            root = GetFaces(&eknoten[i]);
            act = root;
            numfaces = 0;
            memset(vector, 0, sizeof(double) * 3);
            while (act != NULL)
            {
                numfaces++;
                vec = GetFaceNormal(act->element);
                // zu bisherigem Vectoren addieren.
                vector[X] += (vec[X] / vec[B]);
                vector[Y] += (vec[Y] / vec[B]);
                vector[Z] += (vec[Z] / vec[B]);
                delete vec;
                act = act->next;
            }
            DeleteEChain(root, FALSE);
            // Ergebnisvector normieren
            norm = sqrt(vector[X] * vector[X] + vector[Y] * vector[Y] + vector[Z] * vector[Z]);
            vector[X] /= norm;
            vector[Y] /= norm;
            vector[Z] /= norm;
            if (!(eknoten[i].flags & FLAG_NORMAL)) // hatte noch keine Normale
            {
                // jetzt die bisherigen Daten backupen und drei vectordaten anhängen
                data = new PTR_DATA[eknoten[i].d_anz + 3];
                // alte Werte sichern
                memcpy(data, eknoten[i].data, sizeof(PTR_DATA) * eknoten[i].d_anz);
                // Neue Werte anhängen
                data[eknoten[i].d_anz].value = vector[X];
                data[eknoten[i].d_anz].typ = DATA_IGNORE;
                data[eknoten[i].d_anz + 1].value = vector[Y];
                data[eknoten[i].d_anz + 1].typ = DATA_IGNORE;
                data[eknoten[i].d_anz + 2].value = vector[Z];
                data[eknoten[i].d_anz + 2].typ = DATA_IGNORE;
                // Jetzt Daten zurück kopieren
                delete eknoten[i].data;
                eknoten[i].d_anz += 3;
                eknoten[i].data = new PTR_DATA[eknoten[i].d_anz];
                memcpy(eknoten[i].data, data, sizeof(PTR_DATA) * eknoten[i].d_anz);
                // Jetzt noch melden, dass für diesen Knoten eine Normale existiert
                ChangeFlag(&eknoten[i], FLAG_NORMAL, TRUE, FALSE);
                delete data;
            }
            else
            {
                eknoten[i].data[eknoten[i].d_anz - 3].value = vector[X];
                eknoten[i].data[eknoten[i].d_anz - 2].value = vector[Y];
                eknoten[i].data[eknoten[i].d_anz - 1].value = vector[Z];
            }
        }
    }
    return (0);
}

// =======================================================
// OptimizeFaces
// =======================================================
// Fasst nebeneinander liegende Dreiecke zu Vierecken zusammen
int RELDATA::OptimizeFaces()
{
    const char *funcname = "RELDATA::OptimizeFaces";
    int i, j, k, cutedge, *p1, *p2, *kanten, *ret;
    ECHAIN *tmp, *nachbar;
    ECHAIN *root = NULL, *act;
    ELEMENT *face, *newface, *edge;
    int deledges = 0, delfaces = 0;
    clock_t start, end;

    start = clock();
    ChangeFlag(ELEM_FACE, FLAG_OPT, FALSE, FALSE);
    for (i = 0; i < anz_eflaechen; ++i)
    {
        // Es wird nach Dreiecken gesucht, die benutzt werden und noch nicht optimiert sind
        // (dann sind es hoffentlich auch keine Dreecke mehr ;-)
        if (eflaeche[i].e_anz == 3 && !(eflaeche[i].flags & FLAG_OPT) && eflaeche[i].flags & FLAG_USED)
        {
            nachbar = GetNeighbours(&eflaeche[i], 1);
            while (nachbar != NULL && !(eflaeche[i].flags & FLAG_OPT))
            {
                face = nachbar->element;
                if (face->e_anz == 3 && (face->flags & FLAG_USED) && !(face->flags & FLAG_OPT))
                {
                    // Jetzt mal nach den Kanten sehen
                    // Welches ist die Berührungskante?
                    p1 = GetNodes(&eflaeche[i]);
                    p2 = GetNodes(face);
                    kanten = new int[6];
                    memcpy(&kanten[0], &p1[3], sizeof(int) * 3);
                    memcpy(&kanten[3], &p2[3], sizeof(int) * 3);

                    // fixme: printout der KNoten
                    //          printf("Flaeche1: %d %d %d Kanten %d %d %d\n",p1[0],p1[1],p1[2],p1[3],p1[4],p1[5]);
                    //          printf("Flaeche2: %d %d %d Kanten %d %d %d\n",p2[0],p2[1],p2[2],p2[3],p2[4],p2[5]);

                    delete p1;
                    delete p2;
                    ret = MergeSort(kanten, 6, TRUE); // sortiere und ignoriere VZ, doppelte Kanten liegen jetzt hintereinander
                    delete kanten;
                    kanten = ret;

                    cutedge = 0;
                    for (k = 0; k < 6; ++k)
                    {
                        if (kanten[k] == -kanten[k + 1])
                        {
                            cutedge = abs(kanten[k]);
                            if (k != 4) // letzten beiden sind die doppelten
                            {
                                // sonst umsortieren
                                for (j = k; j < 4; ++j)
                                    kanten[j] = kanten[j + 2];
                            }
                        }
                    }
                    if (cutedge == 0)
                    {
                        WriteText(VER_DEBUG, "%s meldet: Fehler in Kantenstruktur bei Flaeche %d und %d.\n", funcname, eflaeche[i].id, face->id);
                    }
                    else
                    {
                        // Kanten in Uhrzeigersinn sortieren
                        SortEdges(kanten, 4, SORT_CLOCKWISE);
                        // Jetzt ein neues Element erstellen
                        if (root == NULL)
                        {
                            root = new ECHAIN;
                            act = root;
                        }
                        else
                        {
                            act->next = new ECHAIN;
                            act = act->next;
                        }
                        act->next = NULL;
                        act->element = CreateElement(ELEM_FACE);
                        newface = act->element;
                        act->id = newface->id;
                        newface->e_anz = 4;
                        newface->element = new PTR_ELEM[4];
                        // Jetzt die Kanten zuweisen
                        for (j = 0; j < 4; ++j)
                        {
                            newface->element[j].id = kanten[j];
                            newface->element[j].ptr = NULL;
                        }
                        // Diese Kante zum löschen freigeben
                        edge = GetElementPtr(cutedge, ELEM_EDGE);
                        //            WriteText(VER_DEBUG,"%d meldet: Kante %d freigegeben.\n",funcname,edge->id);
                        ChangeFlag(edge, FLAG_USED, FALSE);
                        deledges++;
                        // alte Flächen jetzt freigeben
                        ChangeFlag(&eflaeche[i], FLAG_USED, FALSE);
                        ChangeFlag(&eflaeche[i], FLAG_OPT, TRUE);
                        ChangeFlag(face, FLAG_USED, FALSE);
                        ChangeFlag(face, FLAG_OPT, TRUE);
                        delfaces += 2;
                    }
                    delete kanten;
                }
                // Aus nächste Element und dieses fachgerecht löschen
                tmp = nachbar->next;
                nachbar->next = NULL;
                DeleteEChain(nachbar, FALSE);
                nachbar = tmp;
            }
            if (nachbar != NULL)
                DeleteEChain(nachbar, FALSE);
        }
    }

    Disconnect(ELEM_ALL);
    DeleteUnused(ELEM_ALL);
    Merge(root, ELEM_FACE);
    Reconnect(ELEM_ALL);
    end = clock();

    WriteText(VER_MAX, "Optimierung der Kanten fertig [%8.4f sec]. %d Kanten und %d Flaechen geloescht, %d Flaechen erstellt.\n", (double)(end - start) / CLOCKS_PER_SEC, deledges, delfaces, delfaces / 2);

    return (deledges + delfaces);
}

/*
// =======================================================
// OptimizeFaces
// =======================================================
// Fasst nebeneinander liegende Dreiecke zu Vierecken zusammen
int RELDATA::OptimizeFaces()
{
  const char* funcname="RELDATA::OptimizeFaces";
  int i,k,upcount,j,optcount=0;
  clock_t start,end;
  ELEMENT *face,*edge;
//  ELEMENT *actedge;
ECHAIN *root=NULL,*actup,*act,*nachbar,*tmp;
BOOL fertig;
int cutedge;
int ring1[3],ring2[3],*kantenliste;

// Versuch1:
// Dreiecke, deren Basen aneinander liegen zu Vierecken zusammenfassen
// Ersparnis: 0 Knoten, 1 Kante, 1 Fläche
ChangeFlag(ELEM_FACE,FLAG_OPT,FALSE,FALSE);
// Alle Flächen druchsuchen
start =clock();
for (i=0;i<anz_eflaechen;++i)
{
// Benutzte Dreiecke finden, die noch nicht optimiert wurden
if ((eflaeche[i].flags & FLAG_USED) && !(eflaeche[i].flags & FLAG_OPT) && eflaeche[i].e_anz==3)
{
fertig=FALSE;
// hier lohnt sich ein Blick
//      for (k=0;k<eflaeche[i].e_anz && !fertig;++k)
//      {
// GetNeighbour liefert in erster Rekursion die Quelle nicht mit
nachbar = GetNeighbours(&eflaeche[i],1);
while(nachbar!=0)
{
face = nachbar->element;
if (face->e_anz==3 && face->flags & FLAG_USED && !(face->flags & FLAG_OPT))
{

// Welche Flächen gehören noch zu meinen Kanten?
//          actup = eflaeche[i].element[k].ptr->up;
//          upcount=0;
//          while (actup!=NULL && !fertig)
//          {
//            upcount++;

// Gelistete Fläche ist nicht die Originalfläche
//            if (actup->id!=eflaeche[i].id && !(actup->element->flags & FLAG_OPT) && actup->element->flags & FLAG_USED)
//            {
// Fehlerüberprüfung:
//              if (upcount>2)
//                WriteText(VER_DEBUG,"Optimierung meldet: Kante gefunden [ID = %d] die mit mehr als zwei Flaechen verbunden ist!\n",funcname,eflaeche[i].element[k].ptr->id);
// Version1: einfach mit anderem Dreieck zusammenfassen
// Prüfe, ob Fläche gemeinsame Kante mit Startfläche hat
//              face = actup->element;
// gemeinsame Kante löschen
// Aussenring der beiden Dreiecke erzeugen
// Ring1: aktuelles Element
kantenliste = new int[6];
for (j=0;j<3;++j)
{
ring1[j]=eflaeche[i].element[j].id;
kantenliste[j]=abs(ring1[j]);
}
// Ring2: gefundener Nachbar
for (j=0;j<3;++j)
{
ring2[j]=face->element[j].id;
kantenliste[j+3]=abs(ring2[j]);
}
kantenliste = MergeSort(kantenliste,6);
// Jetzt nach doppelten Elementen suchen
cutedge=0;
for (j=0;j<5;++j)
{
if (kantenliste[j]==kantenliste[j+1])
{
cutedge = kantenliste[j];
WriteText(VER_DEBUG,"Die Dreiecke %d und %d teilen sich die Kante %d\n",eflaeche[i].id, face->id, cutedge);
}
}
if (cutedge!=0) //Diese Flächen haben eine gemeinsame Kante
{
// Die Ringe jetzt so hindrehen, dass die zu löschende Kante an erster Position ist
memset(kantenliste,0,sizeof(int)*6);
memcpy(kantenliste,ring1,sizeof(int)*3);  // ACHTUNG: jetzt hat die Liste wieder Vorzeichen!!
j =0;
while(abs(kantenliste[j])!=cutedge)
{
kantenliste[j+3]=kantenliste[j];
j++;
}
memset(ring1,0,sizeof(int)*3);
memcpy(ring1,&kantenliste[j],sizeof(int)*3);
// fertig Ring1

memset(kantenliste,0,sizeof(int)*6);
memcpy(kantenliste,ring2,sizeof(int)*3);
j =0;
while(abs(kantenliste[j])!=cutedge)
{
kantenliste[j+3]=kantenliste[j];
j++;
}
memset(ring2,0,sizeof(int)*3);
memcpy(ring2,&kantenliste[j],sizeof(int)*3);
// fertig ring2

// FIXME: Nur mal neue Vierecke erstellen
// Neues Viereck erstellen
if (root==NULL)
{
root = new ECHAIN;
act  = root;
}
else
{
act->next = new ECHAIN;
act=act->next;
}
act->next=NULL;
act->element = CreateElement(ELEM_FACE);  // liefert ID,TYP und FLAG_USED
act->id = act->element->id;
act->element->e_anz = 4;
act->element->element = new PTR_ELEM[4];
// Drehsinn ist gleichläufig, deshalb jetzt:
act->element->element[0].id = ring1[1];
act->element->element[0].ptr = NULL;
act->element->element[1].id = ring1[2];
act->element->element[1].ptr = NULL;
act->element->element[2].id = ring2[1];
act->element->element[2].ptr = NULL;
act->element->element[3].id = ring2[2];
act->element->element[3].ptr = NULL;

delete kantenliste;
kantenliste=NULL;
fertig=TRUE;
// Kante freigeben
edge = GetElementPtr(cutedge,ELEM_EDGE);
ChangeFlag(edge,FLAG_USED,FALSE,FALSE);
// neue Flags für beide Flächen setzen:
ChangeFlag(face,FLAG_USED,FALSE,FALSE);
ChangeFlag(&eflaeche[i],FLAG_USED,FALSE,FALSE);
// zur Kontrolle
optcount++;
WriteText(VER_DEBUG,"Ein neues Viereck mit den Kanten %d %d %d %d wurde erstellt\n",ring1[1],ring1[2],ring2[1],ring2[2]);
} // Keine gemeinsame Kante
//            } // gleiche oder gesperrte Fläche
//            actup=actup->next;
//          } // Fertig mit der UP-Liste dieser Kante
//          if (upcount>2)
//            WriteText(VER_NORMAL,"Die Kante %d der Flaeche %d wird von %d Elementen benutzt!\n",eflaeche[i].id,eflaeche[i].element[k].id,upcount);
} //Ende Nachbar ist ein Dreieck
tmp = nachbar;
nachbar=nachbar->next;
tmp->next=NULL;
DeleteEChain(tmp,FALSE);
} // Ende Schleife über alle nachbarn
} // Diese Fläche ist ein viereck oder gesperrt
} // Ende Schleife über alle Flächen
// Alle Optimize-Flags löschen
ChangeFlag(ELEM_FACE,FLAG_OPT,FALSE,FALSE);
Disconnect(ELEM_ALL);
Merge(root,ELEM_FACE);
DeleteUnused(ELEM_ALL);
Reconnect(ELEM_ALL);
end=clock();
WriteText(VER_MAX,"Es wurden %d Dreieckspaare gefunden, Optimierung liegt bei %4.1f Proz in %8.4f sec\n",optcount,(float)(100*optcount)/anz_eflaechen,(double)(end-start)/CLOCKS_PER_SEC);
return(0);
}
*/
// =======================================================
// GetVector
// =======================================================
// liefert den Vector einer Kante, von->nach, im vierten Feld steht der
// Betrag des Vectors
double *RELDATA::GetVector(ELEMENT *elem)
{
    const char *funcname = "RELDATA::GetVector";
    double *ret = new double[4];

    switch (elem->typ)
    {
    case ELEM_NODE:
        ret[X] = elem->data[X].value;
        ret[Y] = elem->data[Y].value;
        ret[Z] = elem->data[Z].value;
        break;

    case ELEM_EDGE:
        // Vector erstellen
        ret[X] = elem->element[NACH].ptr->data[X].value - elem->element[VON].ptr->data[X].value;
        ret[Y] = elem->element[NACH].ptr->data[Y].value - elem->element[VON].ptr->data[Y].value;
        ret[Z] = elem->element[NACH].ptr->data[Z].value - elem->element[VON].ptr->data[Z].value;
        break;

    case ELEM_FACE:
    case ELEM_CELL:
    default:
        WriteText(VER_DEBUG, "%s meldet: Nicht implementiert fuer Elelemntty %d\n", funcname, elem->typ);
        return (NULL);
    }
    // Länge ausrechnen
    ret[B] = sqrt(ret[X] * ret[X] + ret[Y] * ret[Y] + ret[Z] * ret[Z]);
    return (ret);
}

/*
// =======================================================
// GetFaceAngle
// =======================================================
// Liefert den Winkel zwischen den Normalen zweier *EBENER*
// Flächen und somit den Winkel an der Kante zwischen den Flächen
// Die Flächen müssen aber nicht nebeneinander liegen
// liefert den Winkel in Grad
double RELDATA::GetFaceAngle(ELEMENT* elem1, ELEMENT* elem2)
{
  const char* funcname="RELDATA::GetFaceAngle";
double ret=0.0;
double *vec1,*vec2;

vec1 = GetFaceNormal(elem1);
vec2 = GetFaceNormal(elem2);

// Jetzt winkel rausbekommen
ret = 360*acos((vec1[X]*vec2[X]+vec1[Y]*vec2[Y]+vec1[Z]*vec2[Z])/(vec1[B]*vec2[B]))/(2*PI);

delete vec1;
delete vec2;
return(ret);
}
*/

// =======================================================
// GetFaceAngle2
// =======================================================
// Version für nichtebene Flächen
double RELDATA::GetFaceAngle(ELEMENT *elem1, ELEMENT *elem2)
{
    //Generates C4189
    //const char* funcname="RELDATA::GetFaceAngle2";
    double ret = 0.0;
    int i, k, size;
    double *avec, *bvec;
    double sum[2][4];
    ELEMENT *elem[2];

    elem[0] = elem1;
    elem[1] = elem2;

    // hier kommen alle Normalen rein
    for (k = 0; k < 2; ++k)
    {
        sum[k][X] = 0;
        sum[k][Y] = 0;
        sum[k][Z] = 0;
        sum[k][B] = 0;

        size = elem[k]->e_anz;
        // erste mal den unteren (b) Vektor besorgen
        bvec = GetVector(elem[k]->element[size - 1].ptr);
        if (elem[k]->element[size - 1].id < 0)
        {
            bvec[X] *= -1;
            bvec[Y] *= -1;
            bvec[Z] *= -1;
        }
        for (i = 0; i < size; ++i)
        {
            avec = bvec;
            // alten A-Vektor umdrehen
            avec[X] *= -1;
            avec[Y] *= -1;
            avec[Z] *= -1;
            bvec = GetVector(elem[k]->element[i].ptr);
            // Vektor in richtige Richtung drehen
            if (elem[k]->element[i].id < 0)
            {
                bvec[X] *= -1;
                bvec[Y] *= -1;
                bvec[Z] *= -1;
            }
            // jetzt a x b rechnen
            sum[k][X] += (avec[Y] * bvec[Z] - avec[Z] * bvec[Y]);
            sum[k][Y] += (avec[Z] * bvec[X] - avec[X] * bvec[Z]);
            sum[k][Z] += (avec[X] * bvec[Y] - avec[Y] * bvec[X]);
            // alten avec löschen
            delete avec;
        }
        delete bvec;
        sum[k][X] /= size;
        sum[k][Y] /= size;
        sum[k][Z] /= size;
        sum[k][B] = sqrt(sum[k][X] * sum[k][X] + sum[k][Y] * sum[k][Y] + sum[k][Z] * sum[k][Z]);
    }
    // Jetzt den Winkel zwischen sum[0] und sum[1] bestimmen und zurück liefern
    // Jetzt winkel rausbekommen
    ret = 360 * acos((sum[0][X] * sum[1][X] + sum[0][Y] * sum[1][Y] + sum[0][Z] * sum[1][Z]) / (sum[0][B] * sum[1][B])) / (2 * PI);
    return (ret);
}

// =======================================================
// GetAngles
// =======================================================
// Liefert die Innenwinkel einer Fläche in der Reihenfolge
// der Kanten. Der Winkel ist immer am "von"-Punkt der
// Kante.
double *RELDATA::GetAngles(ELEMENT *actelem)
{
    const char *funcname = "RELDATA::GetAngles";
    double *ret = NULL, *vec1, *vec2;
    int i;

    if (actelem->typ != ELEM_FACE)
    {
        if (Globals.verbose != VER_NONE)
        {
            printf("%s meldet: Element ( Typ: %d, ID: %d) ist keine Flaeche!\n", funcname, actelem->typ, actelem->id);
            return (NULL);
        }
    }

    // Zielliste erstellen
    ret = new double[actelem->e_anz];
    // Jetzt rundrum, Kanten sind logisch sortiert
    // Die "linke" Kante muß immer die Orientierung "+" haben
    // Die "rechte" Kante muß immer die Orientierung "-" haben
    // letzte Kante
    vec2 = GetVector(actelem->element[actelem->e_anz - 1].ptr);
    // hindrehen, als ob er gerade "linker" Vector gewesen wäre
    if (actelem->element[actelem->e_anz - 1].id < 0) // Kante umdrehen
    {
        vec2[X] *= -1;
        vec2[Y] *= -1;
        vec2[Z] *= -1;
    }
    for (i = 0; i < actelem->e_anz; ++i)
    {
        // Aktuellen Kantenvector besorgen
        vec1 = GetVector(actelem->element[i].ptr);
        // Bei Bedarf "umdrehen"
        if (actelem->element[i].id < 0)
        {
            vec1[X] *= -1;
            vec1[Y] *= -1;
            vec1[Z] *= -1;
        }
        // Der "rechte" Vector ist aus der vorigen Berechnung bekannt, er muß nur noch umgedreht werden
        vec2[X] *= -1;
        vec2[Y] *= -1;
        vec2[Z] *= -1;
        // Winkel zwischen den Vectoren errechnen
        ret[i] = acos((vec1[X] * vec2[X] + vec1[Y] * vec2[Y] + vec1[Z] * vec2[Z]) / (vec1[B] * vec2[B]));
        // Vectoren tauschen
        delete vec2;
        vec2 = vec1;
    }
    delete vec1;
    return (ret);
}

// =======================================================
// GetFaceID
// =======================================================
// ACHTUNG: Diese Version der Methode braucht EWIG!!
int RELDATA::GetFaceID(int *punkte, int size, ECHAIN *neuflaechen, ECHAIN *neukanten)
{
    //Generates C4189
    //const char* funcname="RELDATA::GetFaceID";
    ECHAIN *act;
    //Generates C4189
    //int id=0,

    int *op = NULL;
    {
        int *tmp = new int[size];
        memcpy(tmp, punkte, sizeof(int) * size);
        op = MergeSort(tmp, size);
        delete[] tmp;
    }

    // Erst die neue Kette durchsuchen
    if (neuflaechen != NULL)
    {
        act = neuflaechen;
        while (act != NULL)
        {
            if (act->element->e_anz == size)
            {
                int *tmp = GetNodes(act->element);
                int *p = MergeSort(tmp, size);
                delete[] tmp;
                // Vergleiche p und op
                BOOL equal = TRUE;
                for (int k = 0; k < size && equal == TRUE; ++k)
                {
                    if (p[k] != op[k])
                        equal = FALSE;
                }
                delete[] p;
                if (equal)
                {
                    delete[] op;
                    return (act->id);
                }
            }
            act = act->next;
        }
    }

    // Jetzt den Bestand
    for (int i = 0; i < anz_eflaechen; ++i)
    {
        if (eflaeche[i].e_anz == size)
        {
            int *tmp = GetNodes(&eflaeche[i], neukanten);
            int *p = MergeSort(tmp, size);
            delete[] tmp;
            // Vergleiche p und op
            BOOL equal = TRUE;
            for (int k = 0; k < size && equal == TRUE; ++k)
            {
                if (p[k] != op[k])
                    equal = FALSE;
            }
            delete[] p;
            if (equal)
            {
                delete[] op;
                return (eflaeche[i].id);
            }
        }
    }

    // Nicht gefunden
    delete[] op;
    return (0);
}

// =======================================================
// GetEdgeID
// =======================================================
int RELDATA::GetEdgeID(int von, int nach, ECHAIN *neukanten)
{
    //Generates C4189
    //const char* funcname="RELDATA::GetEdgeID";
    //int id=0;
    int i;
    ECHAIN *act;

    for (i = 0; i < anz_ekanten; ++i)
    {
        if (ekante[i].element[VON].id == von && ekante[i].element[NACH].id == -nach)
            return (ekante[i].id);
        else if (ekante[i].element[VON].id == nach && ekante[i].element[NACH].id == -von)
            return (-ekante[i].id);
    }
    // in der Liste der Neuen Kanten nachsehen, wenn diese nicht Null ist
    if (neukanten != NULL)
    {
        act = neukanten;
        while (act != NULL)
        {
            if (act->element->element[VON].id == von && act->element->element[NACH].id == -nach)
                return (act->id);
            if (act->element->element[VON].id == nach && act->element->element[NACH].id == -von)
                return (-act->id);
            act = act->next;
        }
    }

    return (0);
}

// =======================================================
// GetMaxID
// =======================================================
int RELDATA::GetMaxID(ELEMENTTYPE etype)
{
    const char *funcname = "RELDATA::GetMaxID";
    ELEMENT *acttab;
    int anz_elem;
    int max = 0, i;

    switch (etype)
    {
    case ELEM_NODE:
        acttab = eknoten;
        anz_elem = anz_eknoten;
        break;

    case ELEM_EDGE:
        acttab = ekante;
        anz_elem = anz_ekanten;
        break;

    case ELEM_FACE:
        acttab = eflaeche;
        anz_elem = anz_eflaechen;
        break;

    case ELEM_CELL:
        acttab = ezelle;
        anz_elem = anz_ezellen;
        break;

    default:
        if (Globals.verbose != VER_NONE)
        {
            printf("%s meldet: nicht implementiert fuer Elementtyp %d!\n", funcname, etype);
        }
        return (-1);
    }

    if (anz_elem == -1) // es gibt bisher noch kein Element dieses Typs und es wird ein ID angefordert
        return (1);

    for (i = 0; i < anz_elem; ++i)
        max = MAX(max, abs(acttab[i].id));
    return (max);
}

// =======================================================
// Merge
// =======================================================
// Fuegt die angegebenen Elemente zu den Stammdaten hinzu
// die neuen Elemente werden in einer ECHAIN geliefert
// dabei koennte in ID der neue id stehen...
int RELDATA::Merge(ECHAIN *chain, ELEMENTTYPE etype)
{
    const char *funcname = "RELDATA::Merge";

    int elemcount;
    int anz_elem, i;
    ELEMENT *acttab, *neutab;
    ECHAIN *act, *tmp;
    clock_t start, end;

    if (chain == NULL)
        return (0);

    WriteText(VER_DEBUG, "%s: Merge Elemente vom Typ %d\n", funcname, etype);
    start = clock();
    switch (etype)
    {
    case ELEM_NODE:
        acttab = eknoten;
        anz_elem = anz_eknoten;
        break;

    case ELEM_EDGE:
        acttab = ekante;
        anz_elem = anz_ekanten;
        break;

    case ELEM_FACE:
        acttab = eflaeche;
        anz_elem = anz_eflaechen;
        break;

    case ELEM_CELL:
        acttab = ezelle;
        anz_elem = anz_ezellen;
        break;

    default:
        WriteText(VER_NORMAL, "%s meldet: noch nicht implementiert fuer Elementtyp %d\n", funcname, etype);
        return (0);
    }
    // Wenn eine Tabelle erstellt werden soll, ist ihr zähler bisher noch -1
    if (anz_elem == -1)
        anz_elem = 0; // Damit er sich beim Addieren nicht verzählt ;-)

    // erst mal die gelieferten Elemente zaehlen
    elemcount = 0;
    act = chain;
    while (act != NULL)
    {
        elemcount++;
        act = act->next;
    }
    // Kette sortieren (wegen Einfügen);
    //  chain = SortEChain(chain);
    // Jetzt kann das neue Feld erstellt werden
    // Die aktuelle Ebene wird von den anderen Getrennt
    // Jetzt ein neues Feld erstellen
    //  maxcount = anz_elem;
    //  anz_elem +=elemcount;
    //  neutab = new ELEMENT[anz_elem];
    neutab = new ELEMENT[anz_elem + elemcount];
    // Wichtig! Sonst kotzt CopyElement
    memset(neutab, 0, sizeof(ELEMENT) * (anz_elem + elemcount));
    //  memset(neutab,0,sizeof(ELEMENT)*(anz_elem));  // Wichtig! Sonst kotzt CopyElement

    /*
     // Mergesort letzter Schritt für dieses Problem gut anwendbar
     count=0;
     elemcount=0;
     for (i=0;i<anz_elem;++i)
     {
       if (chain!=NULL && (acttab[count].id > chain->id || count==maxcount))
       {
         elemcount++;
         CopyElement(&neutab[i],chain->element);
         act=chain->next;
   chain->next=NULL;
   DeleteEChain(chain,TRUE);
   chain = act;
   }
   else
   {
   elemcount++;
   CopyElement(&neutab[i],&acttab[count]);
   delete acttab[count].data;
   delete acttab[count].element;
   DeleteEChain(acttab[count].up,FALSE);
   count++;
   }
   }
   */

    // die alten Daten rein kopieren
    for (i = 0; i < anz_elem; ++i)
    {
        CopyElement(&neutab[i], &acttab[i]);
        // Element für Löschung vorbereiten
        delete acttab[i].data;
        delete acttab[i].element;
        DeleteEChain(acttab[i].up, FALSE);
    }
    // jetzt die neuen Elemente reinkopieren
    act = chain;
    elemcount = 0;
    while (act != NULL)
    {
        CopyElement(&neutab[anz_elem + elemcount++], act->element);
        // Element der Kette löschen, wurde ja kopiert
        DeleteElement(act->element);
        // Kette gleich loeschen
        tmp = act->next;
        delete act;
        act = tmp;
    }
    chain = NULL;
    // alte Tabelle löschen (wenn es eine gab!);
    if (acttab != NULL)
        delete acttab;

    anz_elem += elemcount;
    switch (etype)
    {
    case ELEM_NODE:
        eknoten = neutab;
        anz_eknoten = anz_elem;
        break;

    case ELEM_EDGE:
        ekante = neutab;
        anz_ekanten = anz_elem;
        break;

    case ELEM_FACE:
        eflaeche = neutab;
        anz_eflaechen = anz_elem;
        break;

    case ELEM_CELL:
        ezelle = neutab;
        anz_ezellen = anz_elem;
        break;

    default:
        if (Globals.verbose != VER_NONE)
            printf("%s meldet: noch nicht implementiert fuer Elementtyp %d\n", funcname, etype);
        return (0);
    }
    end = clock();

    WriteText(VER_DEBUG, "Fertig mit mergen von %d Elementen [%8.4f sec]\n", anz_elem, (double)(end - start) / CLOCKS_PER_SEC);
    // Jetzt Daten wieder Reconnecten
    return (anz_elem);
}

// =======================================================
// ClearElement
// =======================================================
// Loescht die Verbindungen Von allen anderen Elementen auf diese Element.
// die eigenen Pointer bleiben bestehen. Es werden also alle up-pointerungen gelöscht
// (in beiden Richtungen)
// das Flag FLAG_DISCON wird gesetzt
int RELDATA::ClearElement(ELEMENT *elem)
{
    //Generates C4189
    //const char *funcname ="RELDATA::ClearElement";
    int i;
    ECHAIN *eact;
    //ELEMENT *subelem,*superelem;
    ELEMENT *superelem;
    ECHAIN *tmp;
    int actid;
    //BOOL fertig;

    // Das Element wird aus dem Verbund entkoppelt,
    // niemand zeigt mehr auf das Objekt
    // schritt 1: Pointer von anderen auf sich selber löschen
    eact = elem->up;
    actid = abs(elem->id);
    while (eact != NULL)
    {
        superelem = eact->element;
        // jetzt die eigene ID in der Liste der gepointerten Elemente
        // des Superelements suchen und Pointer löschen (nicht Listeneintrag!!)
        for (i = 0; i < superelem->e_anz; ++i)
        {
            if (abs(superelem->element[i].id) == actid)
            {
                superelem->element[i].ptr = NULL;
            }
        }
        // jetzt gleich eigene Liste löschen
        tmp = eact->next;
        delete eact;
        eact = tmp;
    }
    elem->up = NULL;

    /*

     // Das hier scheint Fehler zu produzieren!
     // schritt 2:
     // Up-einträge aller tiefer gelegenen Elemente (eigenen Element-Liste) löschen
     for (i=0;i<elem->e_anz;++i)
     {
       subelem = elem->element[i].ptr;
       if (subelem!=NULL)  // ganz legal: Kann bereits disconnected sein
       {
         // UP-Liste des Sub-Elements durchsuchen
   fertig = FALSE;
   eact = subelem->up;
   while(eact!=NULL && !fertig)
   {
   if (abs(eact->id)==actid)
   {
   // Diesen UP-Eintrag im subelement löschen
   if (eact == subelem->up)
   {
   subelem->up = subelem->up->next;
   delete eact;
   fertig=TRUE;
   }
   else
   {
   tmp = subelem->up;
   while(tmp->next!=eact)
   tmp=tmp->next;
   tmp->next = eact->next;
   delete eact;
   fertig=TRUE;
   }
   }
   else
   eact=eact->next;
   }
   if (!fertig)
   {
   WriteText(VER_DEBUG,"%s meldet: eigenen ID [%d] nicht in UP-Liste des SUB-Elements [%d] gefunden!\n",funcname,actid,subelem->id);
   }
   }
   }
   */
    // Jetzt zeigt niemand mehr auf dieses Element.
    // seine eigene UP-Liste ist leer
    // seine .ptr-Pointer sind aber noch aktiv!!!
    return (0);
}

// =======================================================
// BreakDownFaces
// =======================================================
// Mittelpunktmethode, macht immer Dreiecke
int RELDATA::BreakDownFaces(int num)
{
    const char *funcname = "RELDATA::BreakDownFaces";
    int i, j, *p, anz_nodes, restecken, count;
    int kanten[4];
    int kant1[3], kant2[3];
    int divisor;
    //  double mp[3]; // Mittelpunkt
    double *mp;
    double ecke[4][3]; // Ecken eines Vieecks
    double dist1, dist2, vec1[4], vec2[4], dmp[4], x, y, z;
    ELEMENT *knoten, *kante;
    ECHAIN *nodes = NULL, *actnode;
    ECHAIN *edges = NULL, *actedge = NULL, *startedge;
    ECHAIN *faces = NULL, *actface;
    ECHAIN *startsearch;

    clock_t start, end;

    int newnodes = 0, newedges = 0, newfaces = 0;

    int oldedgeid; // zu FIXME

    if (num < 3)
        return (0);

    start = clock();
    // Ausgabepuffer auf NULL bei VER_MAX
    //  setvbuf(stdout,NULL,_IONBF,6);
    divisor = MAX(anz_eflaechen / 786, 50); // ab 38400 Elemente vergröbert er die Ausgabe auf >50
    for (i = 0; i < anz_eflaechen; ++i)
    {
        // Zähler, alle "divisor" Zahlen eine Ausgabe
        if (i % divisor == 0)
            printf("%6.2f%%\r", (double)(100 * i) / anz_eflaechen);

        if ((eflaeche[i].flags & FLAG_USED) && eflaeche[i].e_anz > num)
        {
            startsearch = actedge; // actedge==NULL sollte funktionieren
            ChangeFlag(&eflaeche[i], FLAG_USED, FALSE, FALSE);
            if (eflaeche[i].e_anz == 4) // Viereckmethode
            {
                // Theorie: Man ermittelt die beiden "Diagonalen" durch das Viereck
                // und nimmt die kleinere Diagonale.
                // Achtung: schneiden sich die Diagonalen nicht (in ihrer Projektion in
                // Normalenrichtung!), muß die längere gewählt werden. Das Viereck ist dann nicht konvex.
                // Diese Methode ist inhaltlich fehlerhaft!!

                // Es muss ein Sicherer Weg gefunden werden, Vierecke in zwei Dreiecke zu teilen
                // Versuch1:
                // Der Abstand von der Mitte der gewählten Diagonale zu den beiden nicht beteiligten Ecken
                // sollte kleiner sein als die andere Diagonale

                // Optimierung: neue Kanten der zwei Dreiecke werden erst nach der letzten Kante eingetragen

                p = GetNodes(&eflaeche[i]);
                for (j = 0; j < 4; ++j)
                {
                    knoten = GetElementPtr(p[j], ELEM_NODE);
                    ecke[j][X] = knoten->data[X].value;
                    ecke[j][Y] = knoten->data[Y].value;
                    ecke[j][Z] = knoten->data[Z].value;
                }
                // Erste Diagonale 0->2 berechnen
                vec1[X] = ecke[2][X] - ecke[0][X];
                vec1[Y] = ecke[2][Y] - ecke[0][Y];
                vec1[Z] = ecke[2][Z] - ecke[0][Z];
                vec1[B] = sqrt(vec1[X] * vec1[X] + vec1[Y] * vec1[Y] + vec1[Z] * vec1[Z]);

                // Dann 1->3
                vec2[X] = ecke[3][X] - ecke[1][X];
                vec2[Y] = ecke[3][Y] - ecke[1][Y];
                vec2[Z] = ecke[3][Z] - ecke[1][Z];
                vec2[B] = sqrt(vec2[X] * vec2[X] + vec2[Y] * vec2[Y] + vec2[Z] * vec2[Z]);

                // Welche soll es denn sein?
                if (vec1[B] > vec2[B])
                {
                    // nimm diag2 als neue Kante

                    // siehe oben: Abstandsprobe
                    // Schritt1: Mittelpunkt der gewählten Diagonale bestimmen
                    dmp[X] = vec2[X] / 2 + ecke[1][X];
                    dmp[Y] = vec2[Y] / 2 + ecke[1][Y];
                    dmp[Z] = vec2[Z] / 2 + ecke[1][Z];
                    dmp[B] = vec2[B] / 2;

                    // Schritt2: Abstand dieses Mittelpunktes zu den beiden nicht beteiligten Ecken berechnen
                    x = dmp[X] - ecke[0][X];
                    y = dmp[Y] - ecke[0][Y];
                    z = dmp[Z] - ecke[0][Z];
                    dist1 = sqrt(x * x + y * y + z * z);

                    x = dmp[X] - ecke[2][X];
                    y = dmp[Y] - ecke[2][Y];
                    z = dmp[Z] - ecke[2][Z];
                    dist2 = sqrt(x * x + y * y + z * z);

                    // nimm die andere Diagonale
                    if (dist1 > vec1[B] || dist2 > vec1[B])
                    {
                        // nimm doch Diagonale1
                        if (edges == NULL)
                        {
                            edges = new ECHAIN;
                            actedge = edges;
                        }
                        else
                        {
                            actedge->next = new ECHAIN;
                            actedge = actedge->next;
                        }
                        actedge->next = NULL;

                        // FIXME: Gibt es diese Kante schon? sollte eigentlich nicht so sein...
                        oldedgeid = GetEdgeID(p[0], p[2]);
                        if (oldedgeid != 0)
                        {
                            WriteText(VER_NORMAL, "%s meldet: falsche Kante gefunden %d. Diese Kante sollte geloescht sein.\n", funcname, oldedgeid);
                        }
                        // ende FIXME

                        actedge->element = CreateElement(ELEM_EDGE);
                        actedge->id = actedge->element->id;
                        kante = actedge->element;
                        kante->e_anz = 2;
                        kante->element = new PTR_ELEM[2];
                        kante->element[VON].id = p[0];
                        kante->element[VON].ptr = NULL;
                        kante->element[NACH].id = -p[2];
                        kante->element[NACH].ptr = NULL;

                        // Daraus ergeben sich die Kanten für Fläche 1
                        kant1[0] = p[4];
                        kant1[1] = p[5];
                        kant1[2] = -actedge->id;
                        //.. und Fläche 2
                        kant2[0] = p[6];
                        kant2[1] = p[7];
                        kant2[2] = actedge->id;
                    }
                    else
                    {
                        // erste Wahl war gute Wahl!
                        if (edges == NULL)
                        {
                            edges = new ECHAIN;
                            actedge = edges;
                        }
                        else
                        {
                            actedge->next = new ECHAIN;
                            actedge = actedge->next;
                        }
                        actedge->next = NULL;

                        // FIXME: Gibt es diese Kante schon? sollte eigentlich nicht so eins...
                        oldedgeid = GetEdgeID(p[1], p[3]);
                        if (oldedgeid != 0)
                        {
                            WriteText(VER_NORMAL, "%s meldet: falsche Kante gefunden %d. Diese Kante sollte geloescht sein.\n", funcname, oldedgeid);
                        }
                        // ende FIXME

                        actedge->element = CreateElement(ELEM_EDGE);
                        actedge->id = actedge->element->id;
                        kante = actedge->element;
                        kante->e_anz = 2;
                        kante->element = new PTR_ELEM[2];
                        kante->element[VON].id = p[1];
                        kante->element[VON].ptr = NULL;
                        kante->element[NACH].id = -p[3];
                        kante->element[NACH].ptr = NULL;

                        // Daraus ergeben sich die Kanten für Fläche 1
                        kant1[0] = p[4];
                        kant1[1] = actedge->id;
                        kant1[2] = p[7];
                        //.. und Fläche 2
                        kant2[0] = p[5];
                        kant2[1] = p[6];
                        kant2[2] = -actedge->id;
                    }
                }
                else
                {
                    // Das selbe jetzt mit den anderen Diagonalen
                    // Schritt1: Mittelpunkt der gewählten Diagonale bestimmen
                    dmp[X] = vec1[X] / 2 + ecke[0][X];
                    dmp[Y] = vec1[Y] / 2 + ecke[0][Y];
                    dmp[Z] = vec1[Z] / 2 + ecke[0][Z];
                    dmp[B] = vec1[B] / 2;

                    // Schritt2: Abstand dieses Mittelpunktes zu den beiden nicht beteiligten Ecken berechnen
                    x = dmp[X] - ecke[1][X];
                    y = dmp[Y] - ecke[1][Y];
                    z = dmp[Z] - ecke[1][Z];
                    dist1 = sqrt(x * x + y * y + z * z);

                    x = dmp[X] - ecke[3][X];
                    y = dmp[Y] - ecke[3][Y];
                    z = dmp[Z] - ecke[3][Z];
                    dist2 = sqrt(x * x + y * y + z * z);

                    // nimm die andere Diagonale
                    if (dist1 > vec2[B] || dist2 > vec2[B])
                    {
                        // nimm doch Diagonale2
                        if (edges == NULL)
                        {
                            edges = new ECHAIN;
                            actedge = edges;
                        }
                        else
                        {
                            actedge->next = new ECHAIN;
                            actedge = actedge->next;
                        }
                        actedge->next = NULL;

                        // FIXME: Gibt es diese Kante schon? sollte eigentlich nicht so eins...
                        oldedgeid = GetEdgeID(p[1], p[3]);
                        if (oldedgeid != 0)
                        {
                            WriteText(VER_NORMAL, "%s meldet: falsche Kante gefunden %d. Diese Kante sollte geloescht sein.\n", funcname, oldedgeid);
                        }
                        // ende FIXME

                        actedge->element = CreateElement(ELEM_EDGE);
                        actedge->id = actedge->element->id;
                        kante = actedge->element;
                        kante->e_anz = 2;
                        kante->element = new PTR_ELEM[2];
                        kante->element[VON].id = p[1];
                        kante->element[VON].ptr = NULL;
                        kante->element[NACH].id = -p[3];
                        kante->element[NACH].ptr = NULL;
                        // Daraus ergeben sich die Kanten für Fläche 1
                        kant1[0] = p[4];
                        kant1[1] = actedge->id;
                        kant1[2] = p[7];
                        //.. und Fläche 2
                        kant2[0] = p[5];
                        kant2[1] = p[6];
                        kant2[2] = -actedge->id;
                    }
                    else
                    {
                        // erste Wahl war gute Wahl!
                        if (edges == NULL)
                        {
                            edges = new ECHAIN;
                            actedge = edges;
                        }
                        else
                        {
                            actedge->next = new ECHAIN;
                            actedge = actedge->next;
                        }
                        actedge->next = NULL;

                        // FIXME: Gibt es diese Kante schon? sollte eigentlich nicht so eins...
                        oldedgeid = GetEdgeID(p[0], p[2]);
                        if (oldedgeid != 0)
                        {
                            WriteText(VER_NORMAL, "%s meldet: falsche Kante gefunden %d. Diese Kante sollte geloescht sein.\n", funcname, oldedgeid);
                        }
                        // ende FIXME

                        actedge->element = CreateElement(ELEM_EDGE);
                        actedge->id = actedge->element->id;
                        kante = actedge->element;
                        kante->e_anz = 2;
                        kante->element = new PTR_ELEM[2];
                        kante->element[VON].id = p[0];
                        kante->element[VON].ptr = NULL;
                        kante->element[NACH].id = -p[2];
                        kante->element[NACH].ptr = NULL;
                        // Daraus ergeben sich die Kanten für Fläche 1
                        kant1[0] = p[4];
                        kant1[1] = p[5];
                        kant1[2] = -actedge->id;
                        //.. und Fläche 2
                        kant2[0] = p[6];
                        kant2[1] = p[7];
                        kant2[2] = actedge->id;
                    }
                }
                // Beide Flächen jetzt erstellen

                //        SortEdges(kant1,3,SORT_CLOCKWISE,edges); // in richtige Reihenfolge bringen
                //        SortEdges(kant2,3,SORT_CLOCKWISE,edges);
                // in richtige Reihenfolge bringen
                SortEdges(kant1, 3, SORT_CLOCKWISE, startsearch);
                SortEdges(kant2, 3, SORT_CLOCKWISE, startsearch);
                // zwei Flächen erstellen
                if (faces == NULL)
                {
                    faces = new ECHAIN;
                    actface = faces;
                }
                else
                {
                    actface->next = new ECHAIN;
                    actface = actface->next;
                }
                actface->next = NULL;
                actface->element = CreateElement(ELEM_FACE);
                actface->id = actface->element->id;
                // Dreieck erstellen
                actface->element->e_anz = 3;
                actface->element->element = new PTR_ELEM[3];
                actface->element->element[0].id = kant1[0];
                actface->element->element[0].ptr = NULL;
                actface->element->element[1].id = kant1[1];
                actface->element->element[1].ptr = NULL;
                actface->element->element[2].id = kant1[2];
                actface->element->element[2].ptr = NULL;

                // zweite Fläche
                actface->next = new ECHAIN;
                actface = actface->next;
                actface->next = NULL;
                actface->element = CreateElement(ELEM_FACE);
                actface->id = actface->element->id;
                // Dreieck erstellen
                actface->element->e_anz = 3;
                actface->element->element = new PTR_ELEM[3];
                actface->element->element[0].id = kant2[0];
                actface->element->element[0].ptr = NULL;
                actface->element->element[1].id = kant2[1];
                actface->element->element[1].ptr = NULL;
                actface->element->element[2].id = kant2[2];
                actface->element->element[2].ptr = NULL;

                /*
                    else
                    {
                      // nimm diag2 als neue Kante
                      if (edges==NULL)
                      {
                        edges= new ECHAIN;
                        actedge=edges;
                      }
                      else
                      {
            actedge->next = new ECHAIN;
            actedge = actedge->next;
            }
            actedge->next=NULL;
            actedge->element = CreateElement(ELEM_EDGE);
            actedge->id = actedge->element->id;
            kante = actedge->element;
            kante->e_anz = 2;
            kante->element = new PTR_ELEM[2];
            kante->element[VON].id = p[0];
            kante->element[VON].ptr = NULL;
            kante->element[NACH].id = -p[2];
            kante->element[NACH].ptr = NULL;
            // zwei Flächen erstellen
            if (faces==NULL)
            {
            faces= new ECHAIN;
            actface=faces;
            }
            else
            {
            actface->next = new ECHAIN;
            actface = actface->next;
            }
            actface->next = NULL;
            actface->element = CreateElement(ELEM_FACE);
            actface->id = actface->element->id;
            // Dreieck erstellen
            actface->element->e_anz = 3;
            actface->element->element = new PTR_ELEM[3];
            actface->element->element[0].id  = p[4];
            actface->element->element[0].ptr = NULL;
            actface->element->element[1].id  = p[5];
            actface->element->element[1].ptr = NULL;
            actface->element->element[2].id  = -actedge->id;
            actface->element->element[2].ptr = NULL;

            // zweite Fläche
            actface->next = new ECHAIN;
            actface = actface->next;
            actface->next = NULL;
            actface->element = CreateElement(ELEM_FACE);
            actface->id = actface->element->id;
            // Dreieck erstellen
            actface->element->e_anz = 3;
            actface->element->element = new PTR_ELEM[3];
            actface->element->element[0].id  = p[6];
            actface->element->element[0].ptr = NULL;
            actface->element->element[1].id  = p[7];
            actface->element->element[1].ptr = NULL;
            actface->element->element[2].id  = actedge->id;
            actface->element->element[2].ptr = NULL;
            }
            */
                delete p;
            }
            else // Mehreck-Methode
            {
                // Knotenliste besorgen
                p = GetNodes(&eflaeche[i]); // REMEMBER: Liefert die Kantenliste gleich mit!
                anz_nodes = eflaeche[i].e_anz; // Hat soviele Knoten wie Kanten

                // Neuen Punkt in den Schwerpunkt legen
                mp = GetBalancePoint(&eflaeche[i]);

                // Jetzt Knoten erstellen
                newnodes++;
                if (nodes == NULL)
                {
                    nodes = new ECHAIN;
                    actnode = nodes;
                }
                else
                {
                    actnode->next = new ECHAIN;
                    actnode = actnode->next;
                }
                actnode->next = NULL;
                // mit ID erzeugen, NODE erzeugt gleich x,y,z,val
                actnode->element = CreateElement(ELEM_NODE);
                actnode->id = actnode->element->id;
                for (j = 0; j < 3; ++j)
                {
                    // FIXME, er sollte schauen, wo die Daten hingehören und nicht einfach losschreiben
                    actnode->element->data[j].value = mp[j];
                }

                restecken = anz_nodes;
                count = 0;
                if (num > 3)
                {
                    // Alternative: zerlege in Vierecke
                    // Also nur jede zweite Randkante mit dem Mittelpunkt verbinden
                    for (j = 0; j < anz_nodes; j += 2)
                    {
                        // Neue Kante erstellen
                        if (edges == NULL)
                        {
                            edges = new ECHAIN;
                            actedge = edges;
                        }
                        else
                        {
                            actedge->next = new ECHAIN;
                            actedge = actedge->next;
                        }
                        if (j == 0)
                            startedge = actedge; // für später merken
                        actedge->next = NULL;
                        actedge->element = CreateElement(ELEM_EDGE);
                        kante = actedge->element;
                        actedge->id = kante->id;
                        kante->e_anz = 2;
                        kante->element = new PTR_ELEM[2];
                        kante->element[VON].id = abs(p[j]);
                        kante->element[NACH].id = -actnode->id;
                        newedges++;
                    }
                    // Für diesen Durchgang auf die erste erzeugte Kante stellen
                    actedge = startedge;
                    // Viereck-Methode
                    while (restecken > 1)
                    {
                        // Neue Fläche erstellen
                        if (faces == NULL)
                        {
                            faces = new ECHAIN;
                            actface = faces;
                        }
                        else
                        {
                            actface->next = new ECHAIN;
                            actface = actface->next;
                        }
                        actface->next = NULL;
                        actface->element = CreateElement(ELEM_FACE);
                        actface->id = actface->element->id;
                        // Viereck erstellen
                        actface->element->e_anz = 4;
                        actface->element->element = new PTR_ELEM[4];
                        // Kanten in einen Array und sortieren
                        kanten[0] = p[anz_nodes + count];
                        kanten[1] = p[anz_nodes + count + 1];
                        if (actedge->next != NULL)
                            kanten[2] = actedge->next->id;
                        else
                            kanten[2] = startedge->id;
                        kanten[3] = -actedge->id;

                        //            SortEdges(kanten,4,SORT_CLOCKWISE,edges);
                        SortEdges(kanten, 4, SORT_CLOCKWISE, startsearch);

                        for (j = 0; j < 4; ++j)
                        {
                            actface->element->element[j].id = kanten[j];
                            actface->element->element[j].ptr = NULL;
                        }
                        newfaces++;
                        restecken -= 2;
                        count += 2;
                        // nächste Kante
                        if (restecken != 0)
                            actedge = actedge->next;
                    }
                    if (restecken != 0) // noch eine Randecke ist übrig
                    {
                        actface->next = new ECHAIN;
                        actface = actface->next;
                        actface->next = NULL;
                        actface->element = CreateElement(ELEM_FACE);
                        actface->id = actface->element->id;
                        // Dreieck erstellen
                        actface->element->e_anz = 3;
                        actface->element->element = new PTR_ELEM[3];
                        kanten[0] = p[anz_nodes + count];
                        kanten[1] = startedge->id;
                        kanten[2] = -actedge->id;
                        //            SortEdges(kanten,3,SORT_CLOCKWISE,edges);
                        SortEdges(kanten, 3, SORT_CLOCKWISE, startsearch);

                        for (j = 0; j < 3; ++j)
                        {
                            actface->element->element[j].id = kanten[j];
                            actface->element->element[j].ptr = NULL;
                        }
                        newfaces++;
                    }
                }
                else
                {
                    // Dreieckmethode
                    for (j = 0; j < anz_nodes; ++j)
                    {
                        // Neue Kante erstellen
                        if (edges == NULL)
                        {
                            edges = new ECHAIN;
                            actedge = edges;
                        }
                        else
                        {
                            actedge->next = new ECHAIN;
                            actedge = actedge->next;
                        }
                        if (j == 0)
                            startedge = actedge; // für später merken
                        actedge->next = NULL;
                        actedge->element = CreateElement(ELEM_EDGE);
                        actedge->id = actedge->element->id;
                        kante = actedge->element;
                        kante->e_anz = 2;
                        kante->element = new PTR_ELEM[2];
                        kante->element[VON].id = abs(p[j]);
                        kante->element[NACH].id = -actnode->id;
                        newedges++;
                    }
                    // Flächen erstellen
                    actedge = startedge;
                    for (j = 0; j < anz_nodes; ++j)
                    {
                        // Neue Kante erstellen
                        if (faces == NULL)
                        {
                            faces = new ECHAIN;
                            actface = faces;
                        }
                        else
                        {
                            actface->next = new ECHAIN;
                            actface = actface->next;
                        }
                        actface->next = NULL;
                        actface->element = CreateElement(ELEM_FACE);
                        actface->id = actface->element->id;
                        // Dreieck erstellen
                        actface->element->e_anz = 3;
                        actface->element->element = new PTR_ELEM[3];
                        actface->element->element[0].id = p[anz_nodes + j];
                        actface->element->element[0].ptr = NULL;
                        actface->element->element[2].id = -actedge->id;
                        actface->element->element[2].ptr = NULL;

                        actface->element->element[1].ptr = NULL;
                        if (actedge->next == NULL)
                            actface->element->element[1].id = startedge->id;
                        else
                        {
                            actface->element->element[1].id = actedge->next->id;
                            actedge = actedge->next;
                        }
                        // nächste Kante
                    }
                    newfaces += anz_nodes;
                }
                delete p;
                delete mp;
            } // Ende Mehrknotenmethode
        }
    }
    //  setvbuf(stdout,NULL,_IONBF,BUFSIZ);

    Disconnect(ELEM_ALL);
    DeleteUnused(ELEM_ALL);
    Merge(nodes, ELEM_NODE);
    Merge(edges, ELEM_EDGE);
    Merge(faces, ELEM_FACE);
    Reconnect(ELEM_ALL);

    end = clock();
    WriteText(VER_MAX, "%s erstellte %d neue Flaechen [%6.4f sec]\n", funcname, newfaces, (double)(end - start) / CLOCKS_PER_SEC);

    return (0);
}

/*
// =======================================================
// BreakDownFaces
// =======================================================
int RELDATA::BreakDownFaces(int num)
{
  const char *funcname ="RELDATA::BreakDownFaces";
  ECHAIN *edgeroot=NULL,*edgeact;
  ECHAIN *faceroot=NULL,*faceact;
  int newedges=0,newfaces=0,delfaces=0;
  int *idx,k;
int target;
int count,anz_elem=0;
clock_t start,end;

int i,*p,restecken;

// Wenn nicht zerlegt werden soll: Abbruch
if (Globals.breakdownto==MAXINT)
return (0);

start = clock();
WriteText(VER_NORMAL,"Beginne mit dem Zerlegen von grossen Flaechen...\n");
// Schleife ueber alle (alten) Flaechen
for (i=0;i<anz_eflaechen;++i)
{
if (eflaeche[i].e_anz>num)
{
count=0;
// Alle hoehereckigen Flaeche zerlegen
// Punkte besorgen
p = GetNodes(&eflaeche[i]);
anz_elem = eflaeche[i].e_anz;
idx = new int[anz_elem];
restecken = anz_elem;
idx[2] = -1*p[anz_elem]; // linkeste Kante (Dreieck), fuer Schleife. VZ umdrehen, wird zurueckgedreht
while (restecken>=num)
{
// FIXME Erster Versuch: Dreiecke abschneiden
// schritt 1: Kante erstellen (wenn noetig!)

// Index setzen
if (restecken == num)
{
// keine neue Kante erstellen (letzte Flaeche)
// ist ein num-Eck!
target = num;
idx[0] = edgeact->id;         // zuletzt erstellte Kante
for (k=1;k<target;++k)
{
idx[k] = p[anz_elem+count+k];
}
}
else
{
target = 3; // das gibt ein Dreieck
// Kante erstellen
newedges++;
if (edgeroot==NULL)
{
edgeroot=new ECHAIN;
edgeact =edgeroot;
}
else
{
edgeact->next= new ECHAIN;
edgeact=edgeact->next;
}
memset(edgeact,0,sizeof(ECHAIN));
// Kontainer vorbereitet
edgeact->element = CreateElement(ELEM_EDGE);
edgeact->id =edgeact->element->id;
edgeact->element->e_anz = 2;
edgeact->element->element=new PTR_ELEM[2];
// Jetzt die neue Kante erschaffen. Die Kante geht von p[0] nach p[count+2]
edgeact->element->element[0].id = abs(p[0]);
edgeact->element->element[0].ptr = NULL;      // wird spaeter reconnected
edgeact->element->element[1].id = abs(p[count+2])*-1; // nach
edgeact->element->element[1].ptr = NULL;      // wird spaeter reconnected
edgeact->element->id = edgeact->id;

// Indices fuer neue Flaeche
idx[0] = -1*idx[2];
idx[1] = p[anz_elem+count+1];
idx[2] = -1*abs(edgeact->id);
}

// Schritt 2: Flaeche erstellen
newfaces++;
if (faceroot==NULL)
{
faceroot=new ECHAIN;
faceact =faceroot;
}
else
{
faceact->next= new ECHAIN;
faceact=faceact->next;
}
memset(faceact,0,sizeof(ECHAIN));
// Kontainer vorbereitet
faceact->element = CreateElement(ELEM_FACE);  // liefert z.B. ID,TYP und FLAG_USED
faceact->id = faceact->element->id;
faceact->element->e_anz = target;
faceact->element->element=new PTR_ELEM[target];
// Jetzt die neue Flaeche erschaffen
// Verwendet die Kanten p[0] nach p[1], p[1] nach p[2] und die aktelle Kante
for (k=0;k<target;++k)
{
faceact->element->element[k].id = idx[k];
faceact->element->element[k].ptr = NULL;      // wird spaeter reconnected
}
faceact->element->id = faceact->id;

count++;  // Durchlaufzaehler fuer ein Vieleck
restecken--;  // restliche Ecken
}
// Das alte Rechteck ist jetzt ueberfluessig
ChangeFlag(&eflaeche[i],FLAG_USED,FALSE,FALSE);
delfaces++;
delete p;
delete idx;
}
}
end=clock();
// so!
// jetzt haben wir zwei riesen Ketten!
// alle neuen Elemente sind nicht Connected
Disconnect(ELEM_ALL);
Merge(edgeroot,ELEM_EDGE);
Merge(faceroot,ELEM_FACE);
DeleteUnused(ELEM_ALL);
Reconnect(ELEM_ALL);
// die Ketten werden in Merge geloescht
WriteText(VER_NORMAL,"%d neue Kanten und %d neue Flaechen erzeugt, %d Flaechen freigegeben! [%8.4f]\n",newedges,newfaces,delfaces,(double)(end-start)/CLOCKS_PER_SEC);
//  PrintStatement();
return(0);
}
*/

// =======================================================
// CreateUplinks
// =======================================================
// Erstellt die up-Links in allen Elementen, die von diesem
// Element benützt werden
//int RELDATA::CreateUplinks(ELEMENTTYPE etype)
int RELDATA::CreateUplinks(ELEMENT *actelem)
{
    const char *funcname = "RELDATA::CreateUplinks";
    int i;
    ECHAIN *echain;
    ELEMENT *subelem;
    BOOL found;

    /*
     if (!(actelem->flags & FLAG_USED))  // Verursacht Fehler!
     {
       // Sich aus der UP-Link Liste der eigenen Elemente Löschen
       ClearElement(actelem);
       return(0);
     }
   */

    for (i = 0; i < actelem->e_anz; ++i)
    {
        // erst mal suchen, ob der eigene ID nicht schon eingetragen ist
        found = FALSE;
        subelem = actelem->element[i].ptr;
        if (subelem != NULL)
        {
            echain = subelem->up;
            while (!found && echain != NULL)
            {
                if (abs(echain->id) == abs(actelem->id))
                {
                    found = TRUE;
                    WriteText(VER_DEBUG, "%s meldet: Sub-Element %d von Element %d (Typ %d) hat bereits den zu erstellenden up-Eintrag!\n", funcname, subelem->id, actelem->id, actelem->typ);
                }
                echain = echain->next;
            }
            if (!found)
            {
                // meinen ID jetzt im subelement eintragen
                if (subelem->up == NULL)
                {
                    subelem->up = new ECHAIN;
                    echain = subelem->up;
                }
                else
                {
                    // vorspulen bis zum letzten Element
                    echain = subelem->up;
                    while (echain->next != NULL)
                    {
                        echain = echain->next;
                    }
                    echain->next = new ECHAIN;
                    echain = echain->next;
                }
                echain->next = NULL;
                echain->id = actelem->id;
                echain->element = actelem;
            }
        }
    }
    // das wars!
    return (0);
}

// =======================================================
// GetNeighbours
// =======================================================
// Liefert eine Liste mit Nachbarelementen
// Die Ergebniskette sieht folgendermassen aus:
// Jedes Element hat einen (Das letzte Element: keinen) pointer auf ein weiteres
// Element. Die Kette kann spaeter geloescht werden.
// Die Methode liefert *NUR* die Nachbarn eines Elements, nicht
// das Element selber mit! actelem ist also nicht Teil
// des Ergebnisses. Selbst bei höheren Rekursionen (range>1)
// wird das Startelement automatisch gelöscht
// Zur löschung der ECHAIN Kette: Bitte Methode DeleteEChain(...,FALSE) verwenden,
// die element-Pointer zeigen auf benutzte Elemente!!
ECHAIN *RELDATA::GetNeighbours(ELEMENT *actelem, int range)
{
    const char *funcname = "RELDATA::GetNeighbours";
    ECHAIN *ret = NULL, *act, *retact, *addchain;
    ELEMENT *first, *last;
    int i, edges;
    BOOL search;

    if (range == 0)
        return (NULL);

    switch (actelem->typ)
    {
    case ELEM_NODE:
        act = actelem->up;
        // hat das Teil ueberhaupt Verkettete Objekte?
        if (act == NULL)
            return (NULL);
        while (act != NULL)
        {
            if (ret == NULL)
            {
                ret = new ECHAIN;
                retact = ret;
            }
            else
            {
                retact->next = new ECHAIN;
                retact = retact->next;
            }
            retact->next = NULL;
            retact->element = NULL;
            // Diese Kante fuehrt zum Nachbarpunkt
            // ACHTUNG POLYLINE-KANTEN:
            // der aktuelle Punkt ist entweder der Erste oder der Letzte in der Liste
            first = act->element->element[0].ptr;
            last = act->element->element[act->element->e_anz - 1].ptr;
            if (actelem == first) // erster in der Reihe
            {
                //..ist last ein Nachbarpunkt
                retact->element = last;
                retact->id = last->id;
            }
            else
            {
                //.ist first der Nachbarpunkt
                retact->element = first;
                retact->id = first->id;
            }
            // sollte range noch groeßer sein, wird der ermittelte Nachbarpunkt auch gleich mal abgefragt
            // Die Liste enthaelt dann aber doppelte Punkte, die muessen zum Schluß noch rausgeworfen werden
            addchain = GetNeighbours(retact->element, range - 1);
            // Kette Anhaengen
            retact->next = addchain;
            // Bis zum Ende laufen
            while (retact->next != NULL)
                retact = retact->next;
            act = act->next;
        }

        // Jetzt Kette von Doppelten Elementen saeubern
        // erst mal nach ID sortieren (Bubblesort);
        search = TRUE;
        while (search == TRUE)
        {
            search = FALSE;
            // auf root zuruecksetzen
            act = ret;
            while (act->next != NULL)
            {
                if (abs(act->id) > abs(act->next->id))
                {
                    search = TRUE;
                    // vertauschen
                    if (act == ret)
                    {
                        // Pointer auf naechstes Element setzen
                        retact = act->next;
                        act->next = retact->next;
                        retact->next = act;
                        // root-Pointer wieder aufs erste Element
                        ret = retact;
                    }
                    else
                    {
                        // Pointer auf Element VOR act besorgen
                        retact = ret;
                        while (retact->next != act)
                            retact = retact->next;
                        // so, retact steht jetzt ein Element vor act;
                        // vertauschen wollen wir act und act->next;
                        retact->next = act->next;
                        act->next = act->next->next;
                        retact->next->next = act;
                    }
                }
                else
                    act = act->next;
            }
        }
        // Feld jetzt sortiert, alle doppelten IDs loeschen
        retact = ret;
        ret = SortEChain(retact);

        /* 
               while(retact!=NULL)
               {
                 if (retact->next!=NULL)
                 {
                   while (abs(retact->id) == abs(retact->next->id))
                   {
                     // dieses Element loeschen
                     act =retact->next->next;
                     // Folgelement loeschen
                     delete retact->next;
         retact->next = act;
         }
         }
         retact=retact->next;
         }
         */
        // Liste jetzt gesaeubert, jetzt noch eigenen ID loeschen
        retact = ret;
        while (retact != NULL)
        {
            if (abs(retact->id) == abs(actelem->id))
            {
                if (retact == ret)
                {
                    ret = retact->next;
                    delete retact;
                }
                else
                {
                    // pointer auf Element davor besorgen
                    act = ret;
                    while (act->next != retact)
                        act = act->next;
                    // Element loeschen
                    act->next = retact->next;
                    delete retact;
                }
            }
            retact = retact->next;
        }
        break;

    case ELEM_FACE:
        // nach Nachbarflächen suchen
        for (i = 0; i < actelem->e_anz; i++)
        {
            // alle Kanten durchlaufen, zu welchen anderen Flächen sie gehört
            edges = 0;
            act = actelem->element[i].ptr->up;
            while (act != NULL)
            {
                edges++;
                // wenn andere Fläche als eigene...
                if (abs(act->id) != abs(actelem->id))
                {
                    // Neues Element in Kette einfügen
                    if (ret == NULL)
                    {
                        ret = new ECHAIN;
                        retact = ret;
                    }
                    else
                    {
                        retact->next = new ECHAIN;
                        retact = retact->next;
                    }
                    retact->next = NULL;
                    retact->id = act->id;
                    retact->element = act->element;
                    // Weitere Ebene anhängen wenn nötig
                    addchain = GetNeighbours(retact->element, range - 1);
                    // Addchain zu aktueller Kette hinzufügen
                    if (addchain != NULL)
                    {
                        if (ret == NULL)
                        {
                            ret = addchain;
                            retact = ret;
                        }
                        else
                        {
                            retact->next = addchain;
                            while (retact->next != NULL)
                                retact = retact->next;
                        }
                        /*
                                   // Verlegt nach unten
                                   // eventuell doppelte IDs löschen
                                   ret = SortEChain(ret);
                                   retact = ret;
                                   while(retact->next!=NULL)
                                   {
                                     // Flächen-IDs sind vorerst mal immer positiv
                                     while(abs(retact->id)==abs(retact->next->id))
                                     {
                                       // nächsten ID löschen
                     addchain = retact->next;
                     retact->next = addchain->next;
                     delete addchain;
                     }
                     retact=retact->next;
                     }
                     */
                    }
                }
                act = act->next;
            }
            if (edges > 2)
                WriteText(VER_NORMAL, "Die Kante %d der Flaeche %d gehoert zu insgesamt %d Flaechen!\n", actelem->element[i].ptr->id, actelem->id, edges);
        }
        // eventuell doppelte IDs löschen
        // Kann entweder von Rekursion kommen (Nebenfläche hat Hauptfläche als
        // Nachbarn UND umgekehrt), oder von Flächen die sich mehr als eine
        // Kante teilen
        ret = SortEChain(ret);
        if (ret == NULL)
        {
            WriteText(VER_NORMAL, "%s meldet: Flaeche %d hat nur einen Nachbarn\n", funcname, actelem->id);
            // Ist OK, wenn die
            return (NULL);
        }
        retact = ret;
        while (retact->next != NULL)
        {
            // Flächen-IDs sind vorerst mal immer positiv
            if (abs(retact->id) == abs(retact->next->id))
            {
                while (retact->next != NULL && abs(retact->id) == abs(retact->next->id))
                {
                    // retact->next löschen
                    addchain = retact->next;
                    // ist jetzt eventuell NULL
                    retact->next = retact->next->next;
                    addchain->next = NULL;
                    DeleteEChain(addchain, FALSE);
                }
            }
            else
                retact = retact->next;
        }
        break;

    case ELEM_EDGE:
    case ELEM_CELL:
        if (Globals.verbose != VER_NONE)
            printf("%s meldet: Keine implementierung fuer Elementtyp %d!\n", funcname, actelem->typ);
        return (NULL);
    default:
        break;
    }

    // und zurueck
    return (ret);
}

// =======================================================
// ChangeFlag
// =======================================================
// gibt die Anzahl der geaenderten Elemente zurueck
// ACHTUNG: es koennen natuerlich auch Kombinationen von FLAGS uebergeben werden
int RELDATA::ChangeFlag(ELEMENT *actelem, unsigned int flag, BOOL set, BOOL forlower)
{
    //Generates C4189
    //const char *funcname ="RELDATA::ChangeFlag";
    int i;
    int count = 0;

    if (actelem == NULL)
        return (0); // kann bei disconnecteten Objekten vorkommen!

    if (set)
    {
        actelem->flags |= flag;
        count++;
    }
    else
    {
        actelem->flags -= (actelem->flags & flag);
        count++;
    }

    // Abbruchkriterium
    // ACHTUNG: Es werden auch Elemente hoeherer Stufe korrekt bekhandelt, die keine
    // Verweise auf niedrigere Elemente haben. Wann auch immer das vorkommen mag ;-)
    if (!forlower || actelem->e_anz == 0)
        return (count);

    // so, jetzt das ganze fuer mehrere Ebenen
    for (i = 0; i < actelem->e_anz; ++i)
    {
        count += ChangeFlag(actelem->element[i].ptr, flag, set, TRUE);
    }

    return (count);
}

int RELDATA::ChangeFlag(ELEMENTTYPE etype, unsigned int flag, BOOL set, BOOL forlower)
{
    const char *funcname = "RELDATA::ChangeFlag (ETYP)";
    int i, k, elem_anz;
    int count = 0;
    ELEMENT *acttab, *actelem;

    switch (etype)
    {
    case ELEM_NODE:
        acttab = eknoten;
        elem_anz = anz_eknoten;
        break;

    case ELEM_EDGE:
        acttab = ekante;
        elem_anz = anz_ekanten;
        break;

    case ELEM_FACE:
        acttab = eflaeche;
        elem_anz = anz_eflaechen;
        break;

    case ELEM_CELL:
        acttab = ezelle;
        elem_anz = anz_ezellen;
        break;

    case ELEM_ALL:
        ChangeFlag(ELEM_NODE, flag, set, FALSE);
        ChangeFlag(ELEM_EDGE, flag, set, FALSE);
        ChangeFlag(ELEM_FACE, flag, set, FALSE);
        ChangeFlag(ELEM_CELL, flag, set, FALSE);
        return (0);

    default:
        if (Globals.verbose != VER_NONE)
            printf("%s meldet: Nicht implementiert fuer Elementtyp %d\n", funcname, etype);
        return (0);
    }
    if (elem_anz <= 0)
        return (0);

    for (k = 0; k < elem_anz; ++k)
    {
        actelem = &acttab[k];
        if (set)
        {
            actelem->flags |= flag;
            count++;
        }
        else
        {
            actelem->flags -= (actelem->flags & flag);
            count++;
        }
        if (forlower && actelem->e_anz > 0)
        {
            for (i = 0; i < actelem->e_anz; ++i)
            {
                // Vorsicht: der übergebene Pointer kann Null sein (NODE, Disconnectete Elemente)
                count += ChangeFlag(actelem->element[i].ptr, flag, set, TRUE);
            }
        }
    }
    return (count);
}

// =======================================================
// DeleteDoubleElement
// =======================================================
// Löscht ein doppeltes Element
int RELDATA::DeleteDoubleElement(ELEMENTTYPE etype)
{
    const char *funcname = "RELDATA::DeleteDoubleElement";
    ELEMENT *elem, *node, *face, *cell, *edge;
    ECHAIN *actup, *up, *edgeup, *faceup;
    int delcount = 0;
    int i, k, j, *p, *op, sign;
    double x, y, z, erg;
    const double radius = 0.0001;
    int upcnt;
    //  int idx, actidx, act;
    BOOL equal;

    NODELIST *root = NULL, *act, *tmp, *prev;
    NODELIST *feld, *neufeld;

    clock_t start, end;

    start = clock();
    switch (etype)
    {
    case ELEM_NODE:
        // Sonderfall Knoten: Da die Koordinaten mit Rundungsfehlern behaftet sind,
        // muß der Vergleich mit einem bestimmten Radius ablaufen. Diese
        // Unschärfe muß mit Bedacht gewählt werden, weil man so evtl.
        // Unbeasichtigt Knoten zusammenfasst
        feld = new NODELIST[anz_eknoten];
        for (i = 0; i < anz_eknoten; ++i)
        {
            feld[i].idx = i;
            feld[i].x = eknoten[i].data[X].value;
            feld[i].y = eknoten[i].data[Y].value;
            feld[i].z = eknoten[i].data[Z].value;
            feld[i].next = NULL;
        }

        neufeld = MergeSort(feld, anz_eknoten, X);
        delete feld;
        feld = neufeld;
        for (i = 0; i < anz_eknoten; ++i)
        {
            if (root == 0)
            {
                root = new NODELIST;
                act = root;
            }
            else
            {
                act->next = new NODELIST;
                act = act->next;
            }
            memcpy(act, &feld[i], sizeof(NODELIST));
            //        act->next=NULL; // siehe oben!
        }

        // Sortierte Nodelist jetzt fertig
        while (root != NULL)
        {
            act = root->next;
            prev = root;
            if (act != NULL)
            {
                x = act->x - root->x;
                // hier lohnt ein Blick
                while (act != NULL && fabs(x) <= radius)
                {
                    y = act->y - root->y;
                    z = act->z - root->z;

                    erg = sqrt(x * x + y * y + z * z);
                    if (erg <= radius) // Unschärfeformel
                    {
                        // Diesen Punkt löschen
                        delcount++;
                        ChangeFlag(&eknoten[act->idx], FLAG_USED, FALSE, FALSE);
                        WriteText(VER_DEBUG, "%s loescht Knoten %d (doppelt zu Knoten %d)\n", funcname, eknoten[act->idx].id, eknoten[root->idx].id);
                        equal = TRUE;
                        actup = eknoten[act->idx].up;
                        // Umnummerierung der oberen Elemente (Kanten)
                        while (actup != NULL)
                        {
                            elem = actup->element;
                            if (elem->element[VON].id == eknoten[act->idx].id)
                            {
                                elem->element[VON].id = eknoten[root->idx].id;
                                elem->element[VON].ptr = &eknoten[root->idx];
                            }
                            else if (elem->element[NACH].id == -eknoten[act->idx].id)
                            {
                                elem->element[NACH].id = -eknoten[root->idx].id;
                                elem->element[NACH].ptr = &eknoten[root->idx];
                            }
                            actup = actup->next;
                        }
                        // Element act löschen
                        if (prev == root)
                            root->next = act->next;
                        else
                            prev->next = act->next;
                        delete act;
                        act = prev->next;
                        if (act != NULL)
                        {
                            x = act->x - root->x;
                        }
                    }
                    else
                    {
                        prev = act;
                        act = act->next;
                        if (act != NULL)
                        {
                            x = act->x - root->x;
                        }
                    }
                }
            }
            // Root Element löschen
            tmp = root->next;
            delete root;
            root = tmp;
        }

        /*
               // version 30s
               // Teil: ab MergeSort
               s = clock();
               for (i=0;i<anz_eknoten-1;++i)
               {
                 if (feld[i].idx!=0) // schon bearbeitete Felder
                 {
                   act=i+1;
                   x = abs(feld[i].x-feld[act].x);
                   while(x<radius && act < anz_ekanten)  // Feld ist sortiert!
         {
         y = feld[i].y-feld[act].y;
         z = feld[i].z-feld[act].z;
         erg = x*x+y*y+z*z;
         if (erg<radius)
         {
         idx = feld[i].idx;
         actidx = feld[act].idx;
         // Diesen Punkt löschen
         delcount++;
         ChangeFlag(&eknoten[actidx],FLAG_USED,FALSE,FALSE);
         equal=TRUE;
         actup = eknoten[actidx].up;
         // Umnummerierung der oberen Elemente (Kanten)
         while(actup!=NULL)
         {
         elem = actup->element;
         if (elem->element[VON].id == eknoten[actidx].id)
         {
         elem->element[VON].id  = eknoten[idx].id;
         elem->element[VON].ptr = &eknoten[idx];
         }
         else if (elem->element[NACH].id == -eknoten[actidx].id)
         {
         elem->element[NACH].id  = -eknoten[idx].id;
         elem->element[NACH].ptr = &eknoten[idx];
         }
         actup=actup->next;
         }
         // Element act löschen
         feld[act].idx=0;
         }
         act++;
         if (act<anz_ekanten)
         x = feld[i].x-feld[act].x;
         } // ende While radius
         }
         }
         delete feld;
         e = clock();
         printf("%8.4f sec\n",(double)(e-s)/CLOCKS_PER_SEC);
         */

        /*
               // das hier funzt gut: 45,9 sec
               s=clock();
               for (i=0;i<anz_eknoten;++i)
               {
                 tmp = new NODELIST;
                 tmp->idx = i;
                 tmp->x = eknoten[i].data[X].value;
                 tmp->y = eknoten[i].data[Y].value;
                 tmp->z = eknoten[i].data[Z].value;
                 tmp->next=NULL;
         if (root==NULL)
         {
         root=tmp;
         }
         else
         {
         act=root;
         prev=act;
         while(act!=NULL && act->x < tmp->x)
         {
         prev=act;
         act=act->next;
         }
         // einfügen bei act;
         if (act==root)
         {
         tmp->next=root;
         root=tmp;
         }
         else
         {
         if (act==NULL)
         {
         prev->next=tmp;
         }
         else
         {
         prev->next = tmp;
         tmp->next=act;
         }
         }
         }
         }
         e=clock();
         printf("%8.4f sec\n",(double)(e-s)/CLOCKS_PER_SEC);

         s=clock();
         // Sortierte Nodelist jetzt fertig
         while(root!=NULL)
         {
         act=root->next;
         prev=root;
         if (act!=NULL)
         {
         x = act->x-root->x;
         while (act!=NULL && abs(x)<=radius) // hier lohnt ein Blick
         {
         y = act->y-root->y;
         z = act->z-root->z;

         erg = sqrt(x*x+y*y+z*z);
         if (erg<=radius)  // Unschärfeformel
         {
         // Diesen Punkt löschen
         delcount++;
         ChangeFlag(&eknoten[act->idx],FLAG_USED,FALSE,FALSE);
         equal=TRUE;
         actup = eknoten[act->idx].up;
         // Umnummerierung der oberen Elemente (Kanten)
         while(actup!=NULL)
         {
         elem = actup->element;
         if (elem->element[VON].id == eknoten[act->idx].id)
         {
         elem->element[VON].id  = eknoten[root->idx].id;
         elem->element[VON].ptr = &eknoten[root->idx];
         }
         else if (elem->element[NACH].id == -eknoten[act->idx].id)
         {
         elem->element[NACH].id  = -eknoten[root->idx].id;
         elem->element[NACH].ptr = &eknoten[root->idx];
         }
         actup=actup->next;
         }
         // Element act löschen
         if (prev==root)
         root->next = act->next;
         else
         prev->next = act->next;
         delete act;
         act=prev->next;
         if (act!=NULL)
         {
         x = act->x-root->x;
         }
         }
         else
         {
         prev=act;
         act=act->next;
         if (act!=NULL)
         {
         x = act->x-root->x;
         }
         }
         }
         }
         // Root Element löschen
         tmp = root->next;
         delete root;
         root=tmp;
         }
         e=clock();
         printf("%8.4f sec\n",(double)(e-s)/CLOCKS_PER_SEC);
         */

        /*
               // das hier ist lahm: 97,66 sec
               for (i=0;i<anz_eknoten-1;++i)
               {
                 if (eknoten[i].flags & FLAG_USED)
                 {
                   equal=FALSE;
                   for (k=i+1;k<anz_eknoten;++k)
                   {
                     if (eknoten[k].flags & FLAG_USED)
                     {
         x = eknoten[i].data[X].value-eknoten[k].data[X].value;
         if (abs(x)<radius)
         {
         y = eknoten[i].data[Y].value-eknoten[k].data[Y].value;
         z = eknoten[i].data[Z].value-eknoten[k].data[Z].value;

         erg = sqrt(x*x+y*y+z*z);

         if (erg<=radius)  // Unschärfeformel
         {
         // Diesen Punkt löschen
         delcount++;
         ChangeFlag(&eknoten[k],FLAG_USED,FALSE,FALSE);
         equal=TRUE;
         actup = eknoten[k].up;
         while(actup!=NULL)
         {
         elem = actup->element;
         if (elem->element[VON].id == eknoten[k].id)
         {
         elem->element[VON].id  = eknoten[i].id;
         elem->element[VON].ptr = &eknoten[i];
         }
         else if (elem->element[NACH].id == -eknoten[k].id)
         {
         elem->element[NACH].id  = -eknoten[i].id;
         elem->element[NACH].ptr = &eknoten[i];
         }
         actup=actup->next;
         }
         }
         }
         }
         }
         }
         }
         */
        break;

    case ELEM_EDGE:
        for (i = 0; i < anz_eknoten; ++i)
        {
            // An jedem Knoten nachschauen, ob eine Kante doppelt ist
            if (eknoten[i].flags & FLAG_USED)
            {
                up = eknoten[i].up;
                upcnt = 0;
                while (up != NULL)
                {
                    elem = up->element;
                    // von hier aus alle Restlichen UP-Mitglieder auf Gleichheit testen
                    actup = up->next;
                    while (actup != NULL)
                    {
                        equal = FALSE;
                        edge = actup->element;
                        if (edge->flags & FLAG_USED)
                        {
                            if ((edge->element[VON].id == elem->element[VON].id) && (edge->element[NACH].id == elem->element[NACH].id))
                            {
                                equal = TRUE;
                                sign = 1;
                            }
                            else if ((edge->element[VON].id == -elem->element[NACH].id) && (edge->element[NACH].id == -elem->element[VON].id))
                            {
                                equal = TRUE;
                                sign = -1; // Kante relativ umdrehen
                            }
                            if (equal)
                            {
                                // edge und elem sind gleich, edge löschen
                                delcount++;
                                ChangeFlag(edge, FLAG_USED, FALSE, FALSE);
                                // Jetzt alle Elemente, die auf edge zeigen auf elem verbiegen
                                edgeup = edge->up;
                                while (edgeup != NULL)
                                {
                                    face = edgeup->element;
                                    if (face->flags & FLAG_USED)
                                    {
                                        for (k = 0; k < face->e_anz; ++k)
                                        {
                                            if (abs(face->element[k].id) == edge->id)
                                            {
                                                face->element[k].id = SIGN(face->element[k].id) * sign * elem->id;
                                                face->element[k].ptr = elem;
                                            }
                                        }
                                    }
                                    edgeup = edgeup->next;
                                }
                                // das wars
                            }
                        }
                        actup = actup->next;
                    }
                    // Zähler weiter
                    up = up->next;
                    upcnt++;
                }
            }
        }
        break;

    /*
             case ELEM_EDGE:
               for (i=0;i<anz_ekanten;++i)
               {
                 if (ekante[i].flags & FLAG_USED)
                 {
                   actup=ekante[i].element[VON].ptr->up; // Startpunkt sollte reichen
                   upcnt=0;
                   while(actup!=NULL)
                   {
                     // welche Kante ist noch auf diesen Punkt eingetragen?
         elem = actup->element;
         if ((actup->id != ekante[i].id) && (elem->flags & FLAG_USED))
         {
         equal=FALSE;
         if ((elem->element[VON].id ==  ekante[i].element[VON].id)  && (elem->element[NACH].id ==  ekante[i].element[NACH].id))
         {
         equal=TRUE;
         sign=1;
         }
         else if ((elem->element[VON].id == -ekante[i].element[NACH].id) && (elem->element[NACH].id == -ekante[i].element[VON].id))
         {
         equal=TRUE;
         sign=-1;
         }
         if (equal)
         {
         // doppelte Kante gefunden
         ChangeFlag(elem,FLAG_USED,FALSE,FALSE);
         WriteText(VER_DEBUG,"%s loescht Kante %d (doppelt zu Kante %d), Sign: %d\n",funcname,elem->id,ekante[i].id,sign);
         if (elem->id==135)
         printf("FIXME at %s\n",funcname);
         delcount++;
         // UPLINKS dieser Kante verfolgen
         up = elem->up;
         while(up!=NULL)
         {
         for (k=0;k<up->element->e_anz;++k)
         {
         // zeigt dieses Element auf die zu ersetzende Kante?
         if (up->element->element[k].id==elem->id)
         {
         // zeige auf mich!
         up->element->element[k].id = sign * ekante[i].id;
         up->element->element[k].ptr = &ekante[i];
         }
         else if (up->element->element[k].id==-elem->id)
         {
         // zeige auf mich!
         up->element->element[k].id = sign * -1 * ekante[i].id;
         up->element->element[k].ptr = &ekante[i];
         }
         }
         up = up->next;
         }
         }
         }
         actup=actup->next;
         upcnt++;
         }
         WriteText(VER_DEBUG,"%s meldet: Der Punkt %d hat %d Kanten.\n",funcname,ekante[i].element[VON].ptr->id,upcnt);
         }
         }
         break;
         */
    case ELEM_FACE:
        for (i = 0; i < anz_eflaechen; ++i)
        {
            if (eflaeche[i].flags & FLAG_USED)
            {
                // tiefer gelegene Elemente suchen. ACHTUNG: doppelte Kanten müssen noch nicht gelöscht sein!!
                // es reicht ein Knoten für den Test!
                op = GetNodes(&eflaeche[i]);
                BubbleSort(op, eflaeche[i].e_anz);

                node = eflaeche[i].element[0].ptr->element[VON].ptr;
                actup = node->up;
                while (actup != NULL)
                {
                    edgeup = actup->element->up;
                    while (edgeup != NULL)
                    {
                        // diese Flächen hängen an dem Knoten
                        face = edgeup->element;
                        if (face->flags & FLAG_USED && face->id != eflaeche[i].id && face->e_anz == eflaeche[i].e_anz)
                        {
                            // Diese Fläche testen
                            p = GetNodes(face);
                            BubbleSort(p, face->e_anz);
                            equal = TRUE;
                            for (k = 0; k < face->e_anz; ++k)
                            {
                                if (p[k] != op[k])
                                    equal = FALSE;
                            }
                            if (equal)
                            {
                                // diese Fläche löschen
                                delcount++;
                                ChangeFlag(face, FLAG_USED, FALSE, FALSE);
                                WriteText(VER_DEBUG, "%s loescht Flaeche %d (doppelt zu Flaeche %d)\n", funcname, face->id, eflaeche[i].id);
                                // Schauen, ob jemand dieses Element benutzt
                                faceup = face->up;
                                while (faceup != NULL)
                                {
                                    // Diese Elemente umbiegen
                                    cell = faceup->element;
                                    for (j = 0; j < cell->e_anz; ++j)
                                    {
                                        if (cell->element[j].id == face->id)
                                        {
                                            cell->element[j].id = eflaeche[i].id;
                                            cell->element[j].ptr = &eflaeche[i];
                                        }
                                    }
                                    faceup = faceup->next;
                                }
                            }
                            delete p;
                        }
                        edgeup = edgeup->next;
                    }
                    actup = actup->next;
                }
                delete op;
            }
        }
        break;

    case ELEM_ALL:
        delcount += DeleteDoubleElement(ELEM_NODE);
        delcount += DeleteDoubleElement(ELEM_EDGE);
        delcount += DeleteDoubleElement(ELEM_FACE);
        //      delcount += DeleteDoubleElement(ELEM_CELL);
        return (delcount);

    case ELEM_CELL:
    default:
        WriteText(VER_NORMAL, "%s meldet: nicht implementiert fuer Elementtyp %d\n", funcname, etype);
        return (0);
    }
    end = clock();

    WriteText(VER_MAX, "%s meldet: %d Elemente des Typs %d freigegeben! [%8.4f sec]\n", funcname, delcount, etype, (double)(end - start) / CLOCKS_PER_SEC);
    return (delcount);
}

// =======================================================
// KillDoubleElement
// =======================================================
// Löscht zwei Elemente, wenn sie doppelt sind
int RELDATA::KillDoubleElement(ELEMENTTYPE etype)
{
    const char *funcname = "RELDATA::KillDoubleElement";
    ELEMENT *elem, *node, *face;
    ECHAIN *actup, *edgeup;
    int delcount = 0;
    int i, k, *p, *op, sign;
    double x, y, z, erg;
    const double radius = 0.0001;
    BOOL equal, gotcha;

    NODELIST *root = NULL, *act, *tmp, *prev;
    NODELIST *feld, *neufeld;

    clock_t start, end;

    start = clock();
    switch (etype)
    {
    case ELEM_NODE:
        // Sonderfall Knoten: Da die Koordinaten mit Rundungsfehlern behaftet sind,
        // muß der Vergleich mit einem bestimmten Radius ablaufen. Diese
        // Unschärfe muß mit Bedacht gewählt werden, weil man so evtl.
        // Unbeasichtigt Knoten zusammenfasst
        feld = new NODELIST[anz_eknoten];
        for (i = 0; i < anz_eknoten; ++i)
        {
            feld[i].idx = i;
            feld[i].x = eknoten[i].data[X].value;
            feld[i].y = eknoten[i].data[Y].value;
            feld[i].z = eknoten[i].data[Z].value;
            feld[i].next = NULL;
        }

        neufeld = MergeSort(feld, anz_eknoten, X);
        delete feld;
        feld = neufeld;
        for (i = 0; i < anz_eknoten; ++i)
        {
            if (root == 0)
            {
                root = new NODELIST;
                act = root;
            }
            else
            {
                act->next = new NODELIST;
                act = act->next;
            }
            memcpy(act, &feld[i], sizeof(NODELIST));
        }

        // Sortierte Nodelist jetzt fertig
        while (root != NULL)
        {
            act = root->next;
            prev = root;
            gotcha = FALSE; // wird TRUE, wenn mindestens ein passendes Element gelöscht wurde
            if (act != NULL)
            {
                x = act->x - root->x;
                // hier lohnt ein Blick
                while (act != NULL && fabs(x) <= radius)
                {
                    y = act->y - root->y;
                    z = act->z - root->z;

                    erg = sqrt(x * x + y * y + z * z);
                    if (erg <= radius) // Unschärfeformel
                    {
                        // Diesen Punkt löschen
                        if (eknoten[act->idx].up != NULL)
                        {
                            WriteText(VER_DEBUG, "%s meldet: Der Knoten %d wird nicht geloescht, weil er verwendet wird.\n", funcname, eknoten[act->idx].id);
                            prev = act;
                            act = act->next;
                            if (act != NULL)
                            {
                                x = act->x - root->x;
                            }
                        }
                        else
                        {
                            // Element act löschen
                            gotcha = TRUE;
                            delcount++;
                            ChangeFlag(&eknoten[act->idx], FLAG_USED, FALSE, FALSE);
                            //                equal=TRUE;
                            if (prev == root)
                                root->next = act->next;
                            else
                                prev->next = act->next;
                            delete act;
                            act = prev->next;
                        }
                        if (act != NULL)
                        {
                            x = act->x - root->x;
                        }
                    }
                    else
                    {
                        prev = act;
                        act = act->next;
                        if (act != NULL)
                        {
                            x = act->x - root->x;
                        }
                    }
                }
            }
            if (gotcha)
            {
                // Inhalt von ROOT löschen, eventuell Warung ausgeben
                if (eknoten[root->idx].up != NULL)
                {
                    WriteText(VER_DEBUG, "%s meldet: Der Knoten %d wird nicht geloescht, weil er verwendet wird.\n", funcname, eknoten[root->idx].id);
                }
                else
                {
                    delcount++;
                    ChangeFlag(&eknoten[root->idx], FLAG_USED, FALSE, FALSE);
                }
            }
            // Root Element löschen
            tmp = root->next;
            delete root;
            root = tmp;
        }
        break;

    case ELEM_EDGE:
        for (i = 0; i < anz_ekanten; ++i)
        {
            if (ekante[i].flags & FLAG_USED)
            {
                gotcha = FALSE;
                actup = ekante[i].element[VON].ptr->up;
                while (actup != NULL)
                {
                    // welche Kante ist noch auf diesen Punkt eingetragen?
                    elem = actup->element;
                    if (actup->id != ekante[i].id && elem->flags & FLAG_USED)
                    {
                        equal = FALSE;
                        if ((elem->element[VON].id == ekante[i].element[VON].id && elem->element[NACH].id == ekante[i].element[NACH].id))
                        {
                            equal = TRUE;
                            sign = 1;
                        }
                        else if (elem->element[VON].id == -ekante[i].element[NACH].id && elem->element[NACH].id == -ekante[i].element[VON].id)
                        {
                            equal = TRUE;
                            sign = -1;
                        }
                        if (equal)
                        {
                            // doppelte Kante gefunden
                            gotcha = TRUE; // Markieren, da eventuell der Quelldatensatz frei sein könnte!
                            if (ekante[i].up != NULL)
                            {
                                WriteText(VER_DEBUG, "%s meldet: Die Kante %d wird nicht geloescht, weil sie verwendet wird.\n", funcname, elem->id);
                            }
                            else
                            {
                                ChangeFlag(elem, FLAG_USED, FALSE, FALSE);
                                delcount++;
                                // UP-Eintrag der unteren Elemente Löschen
                                ClearElement(elem);
                            }
                        }
                    }
                    actup = actup->next;
                }
                if (gotcha)
                {
                    if (ekante[i].up != NULL)
                    {
                        WriteText(VER_DEBUG, "%s meldet: Die Kante %d wird nicht geloescht, weil sie verwendet wird.\n", funcname, ekante[i].id);
                    }
                    else
                    {
                        ChangeFlag(&ekante[i], FLAG_USED, FALSE, FALSE);
                        ClearElement(&ekante[i]);
                        delcount++;
                    }
                }
            }
        }
        break;

    case ELEM_FACE:
        for (i = 0; i < anz_eflaechen; ++i)
        {
            if (eflaeche[i].flags & FLAG_USED)
            {
                // tiefer gelegene Elemente suchen. ACHTUNG: doppelte Kanten müssen noch nicht gelöscht sein!!
                // es reicht ein Knoten für den Test!
                op = GetNodes(&eflaeche[i]);
                BubbleSort(op, eflaeche[i].e_anz);

                node = eflaeche[i].element[0].ptr->element[VON].ptr;
                actup = node->up;
                gotcha = FALSE;
                while (actup != NULL)
                {
                    edgeup = actup->element->up;
                    while (edgeup != NULL)
                    {
                        // diese Flächen hängen an dem Knoten
                        face = edgeup->element;
                        if (face->flags & FLAG_USED && face->id != eflaeche[i].id && face->e_anz == eflaeche[i].e_anz)
                        {
                            // Diese Fläche testen
                            p = GetNodes(face);
                            BubbleSort(p, face->e_anz);
                            equal = TRUE;
                            for (k = 0; k < face->e_anz; ++k)
                            {
                                if (p[k] != op[k])
                                    equal = FALSE;
                            }
                            if (equal)
                            {
                                // diese Fläche löschen
                                gotcha = TRUE;
                                if (face->up != NULL)
                                {
                                    WriteText(VER_DEBUG, "%s meldet: Die Flaeche %d wird nicht geloescht, weil sie verwendet wird.\n", funcname, face->id);
                                }
                                else
                                {
                                    delcount++;
                                    ChangeFlag(face, FLAG_USED, FALSE, FALSE);
                                    ClearElement(face);
                                }
                            }
                            delete p;
                        }
                        edgeup = edgeup->next;
                    }
                    actup = actup->next;
                }
                delete op;
                // Löschen der Quellfläche
                if (gotcha)
                {
                    if (eflaeche[i].up != NULL)
                    {
                        WriteText(VER_DEBUG, "%s meldet: Die Flaeche %d wird nicht geloescht, weil sie verwendet wird.\n", funcname, eflaeche[i].id);
                    }
                    else
                    {
                        delcount++;
                        ChangeFlag(&eflaeche[i], FLAG_USED, FALSE, FALSE);
                        ClearElement(&eflaeche[i]);
                    }
                }
            }
        }
        break;

    case ELEM_ALL:
        delcount += DeleteDoubleElement(ELEM_CELL);
        delcount += DeleteDoubleElement(ELEM_FACE);
        delcount += DeleteDoubleElement(ELEM_EDGE);
        delcount += DeleteDoubleElement(ELEM_NODE);
        return (delcount);

    case ELEM_CELL:
        WriteText(VER_DEBUG, "%s meldet: nicht implementiert fuer Elementtyp %d\n", funcname, etype);
        return (0);

    default:
        WriteText(VER_NORMAL, "%s meldet: nicht implementiert fuer unbekannten Elementtyp %d\n", funcname, etype);
        return (0);
    }
    end = clock();

    WriteText(VER_MAX, "%s meldet: %d Elemente des Typs %d freigegeben! [%8.4f sec]\n", funcname, delcount, etype, (double)(end - start) / CLOCKS_PER_SEC);
    return (delcount);
}

// =======================================================
// DeleteElement
// =======================================================
// Löscht ein Element
BOOL RELDATA::DeleteElement(ELEMENT *elem)
{
    //Generates C4189
    //const char *funcname ="RELDATA::DeleteElement";

    if (elem == NULL)
        return (FALSE);

    // Datenliste löschen
    delete elem->data;
    // Elementliste löschen
    delete elem->element;
    // UP-Liste löschen ohne daran hängende Elemente zu killen (das wäre ziemlich übel!)
    DeleteEChain(elem->up, FALSE);
    // und jetzt das Element selber
    delete elem;

    return (TRUE);
}

// =======================================================
// DeleteEChain
// =======================================================
// Löscht eine ECHAIN-Kette nach allen Regeln der Kunst :-)
int RELDATA::DeleteEChain(ECHAIN *root, BOOL killelems)
{
    //Generates C4189
    //const char *funcname ="RELDATA::DeleteEChain";
    ECHAIN *tmp;
    int elemcount = 0;

    while (root != NULL)
    {
        elemcount++;
        if (killelems)
        {
            // Löscht Element und up-Liste
            DeleteElement(root->element);
        }

        tmp = root->next;
        delete root;
        root = tmp;
    }

    return (elemcount);
}

// =======================================================
// SortEChain
// =======================================================
// Liefert die Liste der Flächen
ECHAIN *RELDATA::SortEChain(ECHAIN *root)
{

    //Generates C4189
    //const char *funcname ="RELDATA::SortEChain";
    ECHAIN *start = root;
    ECHAIN *act, *tmp;
    BOOL found = TRUE;

    if (root == NULL)
        return (NULL);

    // Nach ID in aufsteigender Reihe sortieren
    while (found)
    {
        found = FALSE;
        act = start;
        while (act->next != NULL)
        {
            if (abs(act->id) > abs(act->next->id))
            {
                found = TRUE;
                if (act == start)
                {
                    start = act->next;
                    act->next = start->next;
                    start->next = act;
                }
                else
                {
                    tmp = start;
                    while (tmp->next != act)
                        tmp = tmp->next;
                    tmp->next = act->next;
                    act->next = act->next->next;
                    tmp->next->next = act;
                }
            }
            else
                act = act->next;
        }
    }
    return (start);
}

// =======================================================
// GetFaces
// =======================================================
// Liefert die Liste der Flächen
ECHAIN *RELDATA::GetFaces(ELEMENT *actelem)
{
    const char *funcname = "RELDATA::GetFaces";
    ECHAIN *ret = NULL, *actret, *recret, *tmp;
    ECHAIN *actup;

    if (!(actelem->flags & FLAG_USED))
    {
        WriteText(VER_DEBUG, "%s meldet: Element [Typ:%d ID:%d] als ungenutzt markiert!\n", funcname, actelem->typ, actelem->id);
        return (NULL);
    }

    actup = actelem->up;
    switch (actelem->typ)
    {
    case ELEM_NODE:
        while (actup != NULL)
        {
            recret = GetFaces(actup->element);
            if (ret == NULL)
            {
                ret = recret;
            }
            else
            {
                actret = ret;
                while (actret->next != NULL)
                    actret = actret->next;
                actret->next = recret;
            }
            actup = actup->next;
        }
        // Liste nach doppelten Elementen durchsuchen
        // Die Ergebnisse wurden von mehereren Durchläufen gesammelt!!
        if (ret == NULL)
            return (NULL); // ist schon etwas komisch: FLAG_USED aber niemand brauchts...

        // Liste sortieren
        actret = SortEChain(ret);
        ret = actret;
        // Doppelte Elemente löschen
        while (actret->next != NULL)
        {
            if (abs(actret->id) == abs(actret->next->id))
            {
                // actret->next löschen
                tmp = actret->next->next;
                delete actret->next;
                actret->next = tmp;
            }
            else
                actret = actret->next;
        }

        break;

    case ELEM_EDGE:
        while (actup != NULL)
        {
            // neues Element für die Kette erstellen
            if (ret == NULL)
            {
                ret = new ECHAIN;
                actret = ret;
            }
            else
            {
                actret->next = new ECHAIN;
                actret = actret->next;
            }
            // Elementdaten speichern
            memcpy(actret, actup, sizeof(ECHAIN));
            actret->next = NULL;
            // weiter
            actup = actup->next;
        }
        break;

    case ELEM_FACE:
    case ELEM_CELL:
    default:
        if (Globals.verbose != VER_NONE)
            printf("%s meldet: Nicht implementiert fuer Elementtyp %d!\n", funcname, actelem->typ);
        return (NULL);
    }

    return (ret);
}

// =======================================================
// GetNodes
// =======================================================
// Liefert die Liste der Knoten in logischer Reihenfolge
// bei ELEM_FACE liefert es zusaetzlich die Reihenfolge der
// Kanten-Indices hinter den Punktdaten
// Bei ELEM_CELL liefert es Knoten und Flächen in der Form:
// [Punkte Bodenfläche][Punkte Dachfläche][Bodenfläche,Dachfläche,Seitenflächen]
int *RELDATA::GetNodes(ELEMENT *actelem, ECHAIN *neukanten)
{
    const char *funcname = "RELDATA::GetNodes";
    ELEMENT *face;
    ECHAIN *act;
    int *ret, *p, d[2], *idx, dummy;
    int i, k;
    switch (actelem->typ)
    {
    case ELEM_NODE:
        // nur ID zurueck liefern
        ret = new int;
        *ret = actelem->id;
        return (ret);

    case ELEM_EDGE:
        ret = new int[actelem->e_anz]; // FIXME: ist nicht fuer Polylines
        // Polylines koennte man so machen:
        // entweder hat dar erste Knoten ein Minus oder der letzte
        // Auf alle Faelle: die Dinger sind schon von->nach sortiert (beim Einlesen!)
        for (i = 0; i < actelem->e_anz; ++i)
            ret[i] = actelem->element[i].ptr->id;
        return (ret);

    case ELEM_FACE:
        // erst mal alle Punkte besorgen
        p = new int[actelem->e_anz * 2];
        idx = new int[actelem->e_anz]; // Array fuer indices, wird parallel mitsortiert
        for (i = 0; i < actelem->e_anz; ++i)
        {
            idx[i] = actelem->element[i].id;
            // FIXME: Funktioniert nur fuer EDGES mit zwei Punkten!!
            if (actelem->element[i].ptr != NULL)
            {
                if (actelem->element[i].id < 0)
                {
                    p[2 * i] = actelem->element[i].ptr->element[1].ptr->id;
                    p[2 * i + 1] = actelem->element[i].ptr->element[0].ptr->id;
                }
                else
                {
                    p[2 * i] = actelem->element[i].ptr->element[0].ptr->id;
                    p[2 * i + 1] = actelem->element[i].ptr->element[1].ptr->id;
                }
            }
            else if (neukanten != NULL)
            {
                // in mitgelieferter Liste suchen
                act = neukanten;
                while (act != NULL && act->id != abs(actelem->element[i].id))
                    act = act->next;
                if (act != NULL) // Kante in Liste gefunden
                {
                    if (actelem->element[i].id < 0)
                    {
                        p[2 * i] = act->element->element[NACH].ptr->id;
                        p[2 * i + 1] = act->element->element[VON].ptr->id;
                    }
                    else
                    {
                        p[2 * i] = act->element->element[VON].ptr->id;
                        p[2 * i + 1] = act->element->element[NACH].ptr->id;
                    }
                }
                else
                {
                    // Funktion gescheitert
                    delete p;
                    delete idx;
                    WriteText(VER_NORMAL, "%s meldet: Kante %d des Elements %d nicht gefunden.\n", funcname, abs(actelem->element[i].id), actelem->id);
                    return (NULL);
                }
            }
        }
        // Reihenfolge der Elemente korrigieren
        for (i = 0; i < actelem->e_anz - 1; ++i)
        {
            // wir suchen nach einem Anschluß an das Element p[2*i+1]
            for (k = i + 1; k < actelem->e_anz; ++k)
            {
                if (p[2 * k] == p[2 * i + 1])
                {
                    // Gefunden! dieses Element tauschen mit dem Element i+1;
                    d[0] = p[2 * (i + 1)];
                    d[1] = p[2 * (i + 1) + 1];
                    p[2 * (i + 1)] = p[2 * k];
                    p[2 * (i + 1) + 1] = p[2 * k + 1];
                    p[2 * k] = d[0];
                    p[2 * k + 1] = d[1];
                    // jetzt fuer indices
                    dummy = idx[i + 1];
                    idx[i + 1] = idx[k];
                    idx[k] = dummy;
                }
            }
        }
        // Jetzt Kanten Indices hinten anhaengen
        ret = new int[2 * actelem->e_anz];
        for (i = 0; i < actelem->e_anz; ++i)
            ret[i] = p[i * 2];
        for (i = 0; i < actelem->e_anz; ++i)
            ret[actelem->e_anz + i] = idx[i];
        delete p;
        delete idx;
        return (ret);

    case ELEM_CELL:
        // REMEMBER: Bei den Zellen ist die erste die Bodenfläche und die
        // zweite die Dachfläche, alle folgenden sind Seitenflächen
        // Es haben als fläche1 und Fläche2 keine gemeinsamen Punkte
        face = actelem->element[0].ptr;
        if (face == NULL)
        {
            // Element nicht connected, Fehler
            WriteText(VER_DEBUG, "%s meldet: Pointer auf Flaeche %d der Zelle %d ist NULL.\n", funcname, actelem->element[0].id, actelem->id);
            return (NULL);
        }
        // Punkte + Flächen indices
        ret = new int[2 * face->e_anz + actelem->e_anz];
        p = GetNodes(face);
        memcpy(ret, p, sizeof(int) * face->e_anz); // Kanten des Ergebnis werden nicht mitkopiert
        delete p;
        face = actelem->element[1].ptr;
        p = GetNodes(face);
        // Beide Flächen haben ja identische Eckzahlen
        memcpy(ret + face->e_anz, p, sizeof(int) * face->e_anz);
        delete p;
        // jetzt noch die Indices der Flächen hintenan
        for (i = 0; i < actelem->e_anz; ++i)
        {
            ret[i + 2 * face->e_anz] = abs(actelem->element[i].id);
        }
        return (ret);

    default:
        if (Globals.verbose != VER_NONE)
            printf("%s meldet: Methode nicht definiert fuer Elementtyp %d\n", funcname, actelem->typ);
        return (NULL);
    }
}

// =======================================================
// GetPrevElementType
// =======================================================
ELEMENTTYPE RELDATA::GetPrevElementType(ELEMENTTYPE etype)
{
    switch (etype)
    {
    case ELEM_EDGE:
        return (ELEM_NODE);

    case ELEM_FACE:
        return (ELEM_EDGE);

    case ELEM_CELL:
        return (ELEM_FACE);

    case ELEM_NODE:
    default:
        return (ELEM_NONE);
    }
}

// =======================================================
// GetNextElementType
// =======================================================
ELEMENTTYPE RELDATA::GetNextElementType(ELEMENTTYPE etype)
{
    switch (etype)
    {
    case ELEM_NODE:
        return (ELEM_EDGE);

    case ELEM_EDGE:
        return (ELEM_FACE);

    case ELEM_FACE:
        return (ELEM_CELL);

    case ELEM_CELL:
    default:
        return (ELEM_NONE);
    }
}

// =======================================================
// GetDataCount
// =======================================================
// Liefert die Anzahl der DATA-Elemente und der ELEMENT-Elemente
// in der Datenzeile
int RELDATA::GetDataCount(char *line, int *tabelle, int *datacount, int *elemcount)
{
    const char *funcname = "RELDATA::GetDataCount";
    int i = 0;
    //Generates C4189
    //int ende=0;
    int count = 1; // wir beginnen mit der 2.Zahl (meist index)

    *datacount = 0; // alles auf Null
    *elemcount = 0;

    while (1)
    {
        // Leerzeichen loeschen
        while (line[i] == ' ')
            ++i;
        // ist das Ende des Strings erreicht?
        if ((line[i] == '\n') || (line[i] == '\r'))
            return (0);
        // Nein, also stehen wir auf einer Zahl
        switch (tabelle[count])
        {
        // diese hier sind Daten:
        case PN_X:
        case PN_Y:
        case PN_Z:
        case PN_DENSITY:
        case PN_GROUP:
            (*datacount)++;
            break;

        // Diese hier sind Elemente
        case PN_ELEM:
            (*elemcount)++;
            break;

        // diese hier sind weder Daten noch Elemente: nix machen
        case PN_IGNORE: //ignoriert ;-)
        case PN_INDEX:
        case PN_NODEIDX:
        case PN_FACEIDX:
        case PN_EDGEIDX:
        case PN_CELLIDX:
        case PN_RED:
        case PN_GREEN:
        case PN_BLUE:
            break;

        default:
            if (Globals.verbose != VER_NONE)
                printf("Methode %s meldet: PROPERTY_NAME %d unbekannt Position %d.\n", funcname, tabelle[count], count);
            return (-1);
        }
        count++; // eine Zahl abgehandelt
        while (isfloat(line[i]))
            ++i; // Zahl ueberspringen
    }
    return (0);
}

// =======================================================
// CopyElement
// =======================================================
// Kopiert eine Struktur und liefert einen Pointer auf die Ziel-Struktur
// Ist dest==NULL, wird ein neues Objekt erstellt und zurueck geliefert
// bei Fehler NULL
ELEMENT *RELDATA::CopyElement(ELEMENT *dest, ELEMENT *source)
{
    const char *funcname = "RELDATA::CopyElement";
    ELEMENT *ret;
    ECHAIN *act, *esource, *edest;

    if (source == NULL)
    {
        if (Globals.verbose == VER_DEBUG)
            printf("Methode %s meldet: Quell-Pointer ist null!\n", funcname);
        return (NULL);
    }

    if (dest == NULL)
    {
        // neues Element erstellen
        // Hier wird zwar ein ID verschwendet, aber was solls
        ret = CreateElement(source->typ, FALSE);
        dest = ret;
    }
    else
        ret = dest;

    // Falls im Zielobjekt noch Daten vorhanden sind...
    if (dest->element != NULL)
        delete dest->element;

    if (dest->data != NULL)
        delete dest->data;

    // Kette befreien
    while (dest->up != NULL)
    {
        act = dest->up->next;
        delete dest->up;
        dest->up = act;
    }

    // mal alles kopieren...
    memcpy(dest, source, sizeof(ELEMENT));
    // .. aber up-Pointer loeschen. Wird spaeter kopiert.
    dest->up = NULL;

    if (dest->e_anz > 0)
    {
        dest->element = new PTR_ELEM[dest->e_anz];
        memcpy(dest->element, source->element, sizeof(PTR_ELEM) * source->e_anz);
    }
    if (dest->d_anz > 0)
    {
        dest->data = new PTR_DATA[dest->d_anz];
        memcpy(dest->data, source->data, sizeof(PTR_DATA) * source->d_anz);
    }
    // UP-Kette kopieren (wenn vorhanden)
    esource = source->up;
    while (esource != NULL)
    {
        if (dest->up == NULL)
        {
            dest->up = new ECHAIN;
            edest = dest->up;
        }
        else
        {
            edest->next = new ECHAIN;
            edest = edest->next;
        }
        memcpy(edest, esource, sizeof(ECHAIN));
        edest->next = NULL;
        esource = esource->next;
    }
    return (ret);
}

// =======================================================
// ElementBinSearch
// =======================================================
// Nuche nach einem ID
int RELDATA::ElementBinSearch(int low, int high, int id, ELEMENTTYPE etype)
{
    const char *funcname = "RELDATA::ElementBinSearch";
    ELEMENT *acttab;
    int m = low + (high - low) / 2;

    switch (etype)
    {
    case ELEM_NODE:
        acttab = eknoten;
        break;

    case ELEM_EDGE:
        acttab = ekante;
        break;

    case ELEM_FACE:
        acttab = eflaeche;
        break;

    case ELEM_CELL:
        if (Globals.verbose != VER_NONE)
            printf("%s ist fuer den Elementtyp %d nicht implementiert\n", funcname, etype);
        return (0);

    default:
        if (Globals.verbose == VER_DEBUG)
            printf("%s meldet: Elementtyp %d unbekannt!\n", funcname, etype);
        return (0);
    }

    if (low == high && acttab[low].id != id)
        return (-1);

    if (low + 1 == high)
    {
        if (acttab[low].id == id || acttab[high].id == id)
        {
            if (acttab[low].id == id)
                return (low);
            else
                return (high);
        }
        else
            return -1;
    }

    // Knoten gefunden
    if (acttab[m].id == id)
        return (m);

    else if (acttab[m].id > id)
        return (ElementBinSearch(low, m, id, etype));
    else
        return (ElementBinSearch(m, high, id, etype));
}

int RELDATA::BinSearch(int low, int high, int id, int *feld)
{
    //Generates C4189
    //const char *funcname ="RELDATA::BinSearch";
    int m = low + (high - low) / 2;

    if (low == high && feld[low] != id)
        return (-1);

    if (low + 1 == high)
    {
        if (feld[low] == id || feld[high] == id)
        {
            if (feld[low] == id)
                return (low);
            else
                return (high);
        }
        else
            return (-1);
    }

    // Knoten gefunden
    if (feld[m] == id)
        return (TRUE);

    else if (feld[m] > id)
        return (BinSearch(low, m, id, feld));
    else
        return (BinSearch(m, high, id, feld));
}

// =======================================================
// GetNextID
// =======================================================
// liefert einen gültigen ID für den angeforderten Elementtyp
int RELDATA::GetNextID(ELEMENTTYPE etype)
{
    //Generates C4189
    //const char *funcname="RELDATA::GetNextID";

    switch (etype)
    {
    case ELEM_NODE:
        return (nextnodeid++);
    case ELEM_EDGE:
        return (nextedgeid++);
    case ELEM_FACE:
        return (nextfaceid++);
    case ELEM_CELL:
        return (nextcellid++);
    default:
        WriteText(VER_DEBUG, "%s meldet: nicht implementiert fuer Elementtyp %d\n", etype);
        return (0);
    }
}

// =======================================================
// CreateElement
// =======================================================
// Erstellt ein neues Element-Gerippe.
// voreingestellte Daten sind:
// ID, TYP, FLAG=USED. ID wird nur geliefert, wenn das Flag
// newid=TRUE ist
ELEMENT *RELDATA::CreateElement(ELEMENTTYPE etype, BOOL newid)
{
    //Generates C4189
    //   const char *funcname="RELDATA::CreateElement";
    ELEMENT *elem;

    elem = new ELEMENT;
    if (elem == NULL)
    {
        WriteText(VER_NORMAL, "%s meldet: Kein Speicherplatz mehr fuer neues Element vom Typ %s\n", elemname[etype]);
        return (NULL);
    }
    memset(elem, 0, sizeof(ELEMENT));
    if (newid)
        elem->id = GetNextID(etype);
    elem->typ = etype;
    elem->flags |= FLAG_USED | FLAG_DISCON;
    // Warum der Scheiss? Weil hier später noch die Behandlung von Daten reinkommt
    // und weil die ID-Vergabe nun zentral ist
    // Deshalb!
    switch (etype)
    {
    case ELEM_NODE:
        // Daten-Value gleich mal anhängen und x-y-z-Daten erstellen

        elem->d_anz = 4;
        elem->data = new PTR_DATA[4];
        memset(elem->data, 0, sizeof(PTR_DATA) * 4);
        elem->data[X].typ = DATA_X;
        elem->data[Y].typ = DATA_Y;
        elem->data[Z].typ = DATA_Z;
        elem->data[B].typ = DATA_VAL;
        // FIXME
        elem->data[B].value = eknoten[0].data[B].value;
        break;

    case ELEM_EDGE:
        elem->e_anz = 2;
        elem->element = new PTR_ELEM[2];
        memset(elem->element, 0, sizeof(PTR_ELEM) * 2);
        break;

    case ELEM_FACE:
        break;
    default:
        break;
    }

    return (elem);
}

// =======================================================
// GetElementPtr
// =======================================================
// Liefert den Pointer aus einen durch den ID bezeichneten Knoten
ELEMENT *RELDATA::GetElementPtr(int inp_id, ELEMENTTYPE etype)
{
    int index, id;
    const char *funcname = "RELDATA::GetElementPtr";
    ELEMENT *acttab;
    int maxval;

    id = abs(inp_id); // negativer ID ist "nach" Merkmal

    // Aktion: suche den ID
    // ACHTUNG: Daten sind groß
    switch (etype)
    {
    case ELEM_NODE:
        acttab = eknoten;
        maxval = anz_eknoten;
        break;

    case ELEM_EDGE:
        acttab = ekante;
        maxval = anz_ekanten;
        break;

    case ELEM_FACE:
        acttab = eflaeche;
        maxval = anz_eflaechen;
        break;

    case ELEM_CELL:
        acttab = ezelle;
        maxval = anz_ezellen;
        return (NULL);

    default:
        WriteText(VER_DEBUG, "%s meldet: Elementtyp %d unbekannt!\n", funcname, etype);
        return (NULL);
    }

    if (maxval <= 0)
    {
        WriteText(VER_DEBUG, "%s meldet: Es gibt keine Elemente des Typs %d.\n", funcname, etype);
        return (NULL);
    }

    index = ElementBinSearch(0, maxval, id, etype);

    if (index == -1)
    {
        WriteText(VER_DEBUG, "%s meldet: Element id %d bei Elementtyp %d nicht gefunden\n", funcname, id, etype);
        return (NULL);
    }
    return (&acttab[index]);
}

// =======================================================
// GetDataOrder
// =======================================================
// Liefert die Reihenfolge der Datentypen einer Property-Zeile
// erwartet im String di property_numbers-Zeile, ab der Zahlenreihenfolge
// Beispiel:
// "! property_numbers= 1 3 5 2 4 0\n"
// erwartet " 1 3 5 2 4 0\n"
int RELDATA::GetDataNameOrder(char *line, int **tabelle)
{
    const char *funcname = "RELDATA::GetDataOrder";
    char zeile[256];
    struct PROPS
    {
        int val;
        struct PROPS *next;
    } *proot = NULL, *pact;

    int num_data = 0; // Anzahl der Properties in Zeile
    int i = 0;
    sprintf(zeile, "%s", line);
    while ((zeile[i] != '\n') && (zeile[i] != '\r'))
    {
        if (isdigit(*(zeile + i)))
        {
            num_data++;
            if (proot == NULL)
            {
                proot = new PROPS;
                pact = proot;
            }
            else
            {
                pact->next = new PROPS;
                pact = pact->next;
            }
            pact->val = atoi(zeile + i);
            pact->next = NULL;
            // Zahl vorspulen
            while (isdigit(*(zeile + i)))
                i++;
        }
        else if (zeile[i] == ' ')
        {
            ++i;
        }
        else
        {
            if (Globals.verbose == VER_DEBUG)
                printf("Methode %s meldet: seltsames Zeichen gefunden [%c]\n", funcname, zeile[i]);
            ++i;
        }
    }

    if (num_data == 0)
    {
        if (Globals.verbose == VER_DEBUG)
            printf("Methode %s meldet: Anzahl der Daten ist null [%s]\n", funcname, zeile);
        return (-1);
    }
    // jetzt ist die Anzahl der Daten bekannt
    *tabelle = new int[num_data];
    i = 0;
    // Jetzt die Pointerkette einsortieren und loeschen
    while (proot != NULL)
    {
        (*tabelle)[i++] = proot->val;
        pact = proot->next;
        delete proot;
        proot = pact;
    }
    // und zurueck
    return (num_data);
}

// =======================================================
// GetDataType
// =======================================================
// Liefert den Variablentyp des angegebenen Property-Typs zurueck
// also z.B. PT_FP oder PT_INT, etc
int RELDATA::GetDataType(int prop)
{
    const char *funcname = "RELDATA::GetDataType";
    if (prop > ANZ_PN)
    {
        if (Globals.verbose == VER_DEBUG)
            printf("Methode %s meldet: property Typ out of range [%d]\n", funcname, prop);
        return (-1);
    }
    return (prop_tab[prop]);
}

// =======================================================
// Reconnect
// =======================================================
// Verbindet eine Elementschicht mit der tiefer gelegenen
// Reconnect verbindet eine Elementebene (z.B. Faces)
// mit der naechst niedrigeren (z.B. Edges)
// ACHTUNG: Da die UP-Liste eines DICONNECTeten Elements gelöscht wurde,
// gibt es keine möglichkeit mehr, die Elementebene selbstständig wieder
// einzuköppeln. Mann muß Also noch ein Reconnect auf die Höhere
// Ebene hinzufügen (Die kennt je die Verbindungen zur nächst niedrigeren)
int RELDATA::Reconnect(ELEMENTTYPE etype)
{
    const char *funcname = "RELDATA::Reconnect";
    int i, k;
    ELEMENT *acttab;
    ELEMENTTYPE prevtyp;
    int anz_elem;
    clock_t start, end;

    //  WriteText(VER_DEBUG,"Reconnecte Elementtyp %d\n",etype);
    start = clock();
    prevtyp = GetPrevElementType(etype);

    switch (etype)
    {
    case ELEM_NODE:
        // Die Elemente wieder als connected melden
        ChangeFlag(etype, FLAG_DISCON, FALSE, FALSE);
        return (-1);

    case ELEM_EDGE:
        anz_elem = anz_ekanten;
        acttab = ekante;
        break;

    case ELEM_FACE:
        anz_elem = anz_eflaechen;
        acttab = eflaeche;
        break;

    case ELEM_CELL:
        anz_elem = anz_ezellen;
        acttab = ezelle;
        break;

    case ELEM_ALL:
        Reconnect(ELEM_EDGE);
        Reconnect(ELEM_FACE);
        Reconnect(ELEM_CELL);
        return (0);

    default:
        if (Globals.verbose == VER_DEBUG)
            printf("%s meldet: Unbekannter Elementtype %d!\n", funcname, etype);
        return (-1);
    }

    if (anz_elem <= 0)
    {
        WriteText(VER_DEBUG, "%s meldet: Es sind keine Elemente vom Typ %d vorhanden.\n", funcname, etype);
        return (0);
    }

    for (i = 0; i < anz_elem; ++i)
    {
        for (k = 0; k < acttab[i].e_anz; ++k)
        {
            acttab[i].element[k].ptr = GetElementPtr(acttab[i].element[k].id, prevtyp);
        }
    }
    // alles reconnected, jetzt uplinken
    for (i = 0; i < anz_elem; ++i)
        CreateUplinks(&acttab[i]);

    // Die Elemente wieder als connected melden
    ChangeFlag(etype, FLAG_DISCON, FALSE, FALSE);
    // Ebenfalls die eine Ebene tiefer
    ChangeFlag(prevtyp, FLAG_DISCON, FALSE, FALSE);

    end = clock();
    //  WriteText(VER_DEBUG,"Fertig mit Reconnect Typ %d [%8.4f sec]\n",etype,(double)(end-start)/CLOCKS_PER_SEC);
    return (0);
}

// =======================================================
// Disconnect
// =======================================================
// Löst die angegebene Ebene von der Höher gelegenen Ebene ab
// ACHTUNG: Anders als Reconnect wird diese Ebene komplett von allen
// anderen Ebenen getrennt. Beim wieder eingliedern muß ein Reconnect
// der nächst höheren ebene (GetNextElementType()) zusätzlich erfolgen
// Beispiel:
// Diskonnect(EDGES)
// Reconnect(EDGES)     Jetzt kennen die Edges wieder ihre NODES, UP-Liste NODES erstellt
// Reconnect(FACES)     Jetzt kennen die Faces wieder ihre Edges, UP-Liste Edges erstellt
int RELDATA::Disconnect(ELEMENTTYPE etype)
{
    const char *funcname = "RELDATA::Disconnect";
    int i;
    ELEMENT *acttab;
    int anz_elem;
    clock_t start, end;

    //  WriteText(VER_DEBUG,"Disconnect typ %d\n",etype);
    start = clock();
    switch (etype)
    {
    case ELEM_NODE:
        anz_elem = anz_eknoten;
        acttab = eknoten;
        break;

    case ELEM_EDGE:
        anz_elem = anz_ekanten;
        acttab = ekante;
        break;

    case ELEM_FACE:
        anz_elem = anz_eflaechen;
        acttab = eflaeche;
        break;

    case ELEM_CELL:
        WriteText(VER_NORMAL, "%s meldet: Nicht implementiert fuer Elementtype %d!\n", funcname, etype);
        return (-1);

    case ELEM_ALL:
        Disconnect(ELEM_NODE);
        Disconnect(ELEM_EDGE);
        Disconnect(ELEM_FACE);
        return (0);

    default:
        if (Globals.verbose == VER_DEBUG)
            printf("%s meldet: Unbekannter Elementtype %d!\n", funcname, etype);
        return (-1);
    }

    for (i = 0; i < anz_elem; ++i)
    {
        // Diese Methode loest dir ptr verpointerung und loescht die up kette
        // der unteren Elemente
        ClearElement(&acttab[i]);
    }
    // Alle aktuellen Elemente Disconnected (hat Verbindung nach unten verloren)
    ChangeFlag(etype, FLAG_DISCON, TRUE, FALSE);
    // Die Ganze Ebene darüber als Disconnected markieren (hat Verbindung nach unten verloren)
    ChangeFlag(GetNextElementType(etype), FLAG_DISCON, TRUE, FALSE);
    end = clock();
    //  WriteText(VER_DEBUG,"Fertig Disconnect typ %d [%8.4f sec]\n",etype,(double)(end-start)/CLOCKS_PER_SEC);
    return (0);
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Body Routinen (nach aussen frei)
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// =======================================================
// PrintStatement
// =======================================================
// Gibt eine kurze Übersicht aus
int RELDATA::PrintStatement(void)
{
    int i;
    int ecke[12] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    WriteText(VER_NORMAL, "Uebersicht: Gesamt Knoten: %d  Kanten: %d  Flaechen: %d  Zellen: %d\n", anz_eknoten, anz_ekanten, anz_eflaechen, anz_ezellen);
    for (i = 0; i < anz_eflaechen; ++i)
    {
        if (eflaeche[i].flags & FLAG_USED)
        {
            switch (eflaeche[i].e_anz)
            {
            case 0:
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
            case 9:
            case 10:
                ecke[eflaeche[i].e_anz]++;
                break;
            default:
                ecke[11]++;
                break;
            }
        }
    }
    WriteText(VER_NORMAL, "Die Flaechen Setzen sich zusammen aus folgenden Eckzahlen:\n");
    for (i = 0; i < 12; ++i)
    {
        if (i == 11)
            WriteText(VER_NORMAL, ">10:%d\n", ecke[i]);
        else
            WriteText(VER_NORMAL, "%d:%d ", i, ecke[i]);
    }

    return (0);
}

// =======================================================
// SurfaceSmooth
// =======================================================
// Verteilt die Aufgaben an die Schreibunterroutinen
int RELDATA::SurfaceSmooth(SMOOTHTYP typ, int runs)
{
    const char *funcname = { "RELDATA::SurfaceSmooth" };
    int count = 0, i, j, k, neighbours, cnt, numfaces;
    ECHAIN *root, *act;
    ELEMENT *kante1;
    ELEMENT *kante2;
    ELEMENT *face;
    double x, y, z, b, *avec, *bvec;
    double skalar, area;
    int smoothed = 0;
    int range = 1;
    int *kanten, *flaechen;
    int anz_kanten, tmp[2];
    BOOL found;

    struct PUNKT
    {
        double x, y, z;
    } *neu, mp;
    //Generates C4189
    //*dummy=NULL,

    if (typ == SM_NONE || runs == 0)
        return (0);

    for (count = 0; count < runs; ++count)
    {
        smoothed = 0;

        neu = new PUNKT[anz_eknoten];
        memset(neu, 0, sizeof(PUNKT) * anz_eknoten);

        // Schleife ueber alle Knoten
        switch (typ)
        {
        case SM_DOUBLE:
            range = 2;
        case SM_NORMAL:
            // Eigener Punkt bekommt dreifache Gewichtung
            for (i = 0; i < anz_eknoten; ++i)
            {
                root = GetNeighbours(&eknoten[i], range);
                if (root != NULL)
                {
                    act = root;
                    neighbours = 0;
                    // Nachbarn zählen
                    while (act != NULL)
                    {
                        neighbours++;
                        act = act->next;
                    }
                    act = root;

                    x = eknoten[i].data[X].value * neighbours;
                    y = eknoten[i].data[Y].value * neighbours;
                    z = eknoten[i].data[Z].value * neighbours;
                    // alle x,y und z-Werte addieren
                    while (act != NULL)
                    {
                        neighbours++;
                        x += act->element->data[0].value;
                        y += act->element->data[1].value;
                        z += act->element->data[2].value;
                        act = act->next;
                    }
                    // neue x,y,z-Werte setzen
                    neu[i].x = x / neighbours;
                    neu[i].y = y / neighbours;
                    neu[i].z = z / neighbours;
                    smoothed++;
                    // Kette loeschen ohne Elemente zu löschen
                    DeleteEChain(root, FALSE);
                }
            } // Schleife ueber alle Knoten
            break;

        case SM_QUALITY:
            for (i = 0; i < anz_eknoten; ++i)
            {
                root = GetNeighbours(&eknoten[i], range);
                if (root != NULL)
                {
                    act = root;
                    x = y = z = 0;
                    neighbours = 0;
                    // alle x,y und z-Werte addieren
                    while (act != NULL)
                    {
                        // Vektorlänge der Kanten mit berücksichtigen
                        neighbours++;
                        x += act->element->data[0].value;
                        y += act->element->data[1].value;
                        z += act->element->data[2].value;
                        act = act->next;
                    }
                    // neue x,y,z-Werte setzen
                    eknoten[i].data[0].value = x / neighbours;
                    eknoten[i].data[1].value = y / neighbours;
                    eknoten[i].data[2].value = z / neighbours;
                    smoothed++;
                    // Kette loeschen ohne Elemente zu löschen
                    DeleteEChain(root, FALSE);
                }
            } // Schleife ueber alle Knoten
            break;

        case SM_GRID:
            // Diese Version verschiebt den aktellen Punkt auf einer Ebene, die durch die
            // Angrenzenden Punkte bestimmt ist. Die Degenerierung sollte deutlich kleiner sein,
            // dabei aber die Äquidistanz der Punkte untereinander deutlich besser
            ChangeFlag(ELEM_NODE, FLAG_OPT, FALSE, FALSE);
            for (i = 0; i < anz_eknoten; ++i)
            {
                if (eknoten[i].flags & FLAG_USED) // && !(eknoten[i].flags & FLAG_OPT))
                {
                    ChangeFlag(&eknoten[i], FLAG_OPT, TRUE, FALSE);
                    cnt = 0; // Kanten zählen
                    act = eknoten[i].up;
                    while (act != NULL)
                    {
                        if (act->element->up != NULL)
                            cnt++;
                        act = act->next;
                    }
                    // Jetzt die Felder erstellen
                    kanten = new int[cnt + 1]; // +1 um später eine einfachere Schleife zu machen
                    flaechen = new int[cnt * 2]; // immer maximal zwei Flächen pro Kante angenommen
                    anz_kanten = cnt;
                    cnt = 0;
                    numfaces = 0;
                    act = eknoten[i].up;
                    while (act != NULL)
                    {
                        if (act->element->up != NULL)
                        {
                            kanten[cnt] = act->id;
                            flaechen[numfaces++] = act->element->up->id;
                            if (act->element->up->next != NULL)
                                flaechen[numfaces++] = act->element->up->next->id;
                            cnt++;
                        }
                        act = act->next;
                    }

                    // Randerkennung:
                    if (numfaces == cnt * 2) // kein Rand
                    {
                        // Jetzt in dem Sinn sortieren, dass die Flächen-Indices eine geschlossene Kette bilden
                        // Dabei können die Paare (Flächen einer Kante) auch vertauscht sein
                        for (j = 0; j < anz_kanten - 1; ++j)
                        {
                            // suche nach dem zweiten Element des Feldes flaechen[]
                            found = FALSE;
                            for (k = j + 1; k < anz_kanten && found == FALSE; ++k)
                            {
                                // Element gefunden, Elemente tauschen
                                if (flaechen[k * 2] == flaechen[j * 2 + 1])
                                {
                                    found = TRUE;
                                    if (k != j + 1)
                                    {
                                        tmp[0] = flaechen[(j + 1) * 2];
                                        tmp[1] = flaechen[(j + 1) * 2 + 1];
                                        flaechen[(j + 1) * 2] = flaechen[k * 2];
                                        flaechen[(j + 1) * 2 + 1] = flaechen[k * 2 + 1];
                                        flaechen[k * 2] = tmp[0];
                                        flaechen[k * 2 + 1] = tmp[1];
                                        // index mit drehen
                                        tmp[0] = kanten[j];
                                        kanten[j] = kanten[k];
                                        kanten[k] = tmp[0];
                                    }
                                }
                                // Gefunden, aber vertauscht
                                else if (flaechen[k * 2 + 1] == flaechen[j * 2 + 1])
                                {
                                    found = TRUE;
                                    if (k != j + 1)
                                    {
                                        // wie oben, nur vertauschen
                                        tmp[0] = flaechen[(j + 1) * 2];
                                        tmp[1] = flaechen[(j + 1) * 2 + 1];
                                        // hier vertauscht
                                        flaechen[(j + 1) * 2] = flaechen[k * 2 + 1];
                                        // hier vertauscht
                                        flaechen[(j + 1) * 2 + 1] = flaechen[k * 2];
                                        flaechen[k * 2] = tmp[0];
                                        flaechen[k * 2 + 1] = tmp[1];
                                        // index mit drehen
                                        tmp[0] = kanten[j];
                                        kanten[j] = kanten[k];
                                        kanten[k] = tmp[0];
                                    }
                                    else
                                    {
                                        // Position stimmt, flaechen aber vertauschen
                                        tmp[0] = flaechen[k * 2];
                                        flaechen[k * 2] = flaechen[k * 2 + 1];
                                        flaechen[k * 2 + 1] = tmp[0];
                                    }
                                }
                            }
                            if (!found)
                            {
                                WriteText(VER_DEBUG, "%s meldet: Fehler in der Kantenfolge bei SM_GRID.\n", funcname);
                            }
                        }
                        // Jetzt sind die Kanten in richtiger Reihenfolge

                        // Neuer Ansatz mit der Methode GetBalancePoint und unter Berücksichtigung
                        // der angrenzenden Schwerpunkte der gesamten Flächen

                        // Liste der Beteiligten Flächen besorgen
                        cnt *= 2;
                        flaechen = DeleteDoubleIDs(flaechen, &cnt);
                        // Schleife über alle Flächen
                        mp.x = 0;
                        mp.y = 0;
                        mp.z = 0;
                        area = 0;
                        for (k = 0; k < cnt; ++k)
                        {
                            face = GetElementPtr(flaechen[k], ELEM_FACE);
                            avec = GetBalancePoint(face);
                            // Jeweils mit der Fläche gewichtet
                            mp.x += (avec[X] - eknoten[i].data[X].value) * avec[B];
                            mp.y += (avec[Y] - eknoten[i].data[X].value) * avec[B];
                            mp.z += (avec[Z] - eknoten[i].data[X].value) * avec[B];
                            area += avec[B];
                            delete avec;
                        }
                        // Den Schwerpunkt jetzt ausrechnen
                        mp.x /= area;
                        mp.y /= area;
                        mp.z /= area;
                        mp.x += eknoten[i].data[X].value;
                        mp.y += eknoten[i].data[X].value;
                        mp.z += eknoten[i].data[X].value;
                        // Fertig!

                        delete flaechen;
                        kanten[anz_kanten] = kanten[0]; // erleichtert die folgende Schleife
                        // jetzt erfolgt die Glättung
                        x = 0;
                        y = 0;
                        z = 0;
                        b = 0;
                        // Mittelpunkt/Schwerpunkt gleich mitberechnen
                        //            mp.x=0;
                        //            mp.y=0;
                        //            mp.z=0;
                        kante2 = GetElementPtr(kanten[0], ELEM_EDGE);
                        bvec = GetVector(kante2);
                        // Vektor umdrehen
                        if (kante2->element[NACH].id == -eknoten[i].id)
                        {
                            bvec[X] *= -1;
                            bvec[Y] *= -1;
                            bvec[Z] *= -1;
                            ChangeFlag(kante2->element[VON].ptr, FLAG_OPT, TRUE, TRUE);
                        }
                        for (k = 0; k < anz_kanten; ++k)
                        {
                            kante1 = kante2; // alte "rechte" Kante wird neue "linke" Kante
                            // neue "rechte" Kante
                            kante2 = GetElementPtr(kanten[k + 1], ELEM_EDGE);
                            // Bilde das Kreuzprodukt von allen Kantenpaaren und addiere es auf
                            // Die Vektoren zeigen immer vom aktuellen Punkt weg!
                            avec = bvec; // siehe oben
                            bvec = GetVector(kante2);
                            // Vektor umdrehen
                            if (kante2->element[NACH].id == -eknoten[i].id)
                            {
                                bvec[X] *= -1;
                                bvec[Y] *= -1;
                                bvec[Z] *= -1;
                                //                mp.x += kante2->element[VON].ptr->data[X].value;
                                //                mp.y += kante2->element[VON].ptr->data[Y].value;
                                //                mp.z += kante2->element[VON].ptr->data[Z].value;
                                ChangeFlag(kante2->element[VON].ptr, FLAG_OPT, TRUE, TRUE);
                            }
                            else
                            {
                                //                mp.x += kante2->element[NACH].ptr->data[X].value;
                                //                mp.y += kante2->element[NACH].ptr->data[Y].value;
                                //                mp.z += kante2->element[NACH].ptr->data[Z].value;
                            }
                            // Schwerpunkt berechnen
                            //              mp.x += (3*eknoten[i].data[X].value+avec[X]+bvec[X])/3;
                            //              mp.y += (3*eknoten[i].data[Y].value+avec[Y]+bvec[Y])/3;
                            //              mp.z += (3*eknoten[i].data[Z].value+avec[Z]+bvec[Z])/3;
                            x += (avec[Y] * bvec[Z] - avec[Z] * bvec[Y]);
                            y += (avec[Z] * bvec[X] - avec[X] * bvec[Z]);
                            z += (avec[X] * bvec[Y] - avec[Y] * bvec[X]);
                            delete (avec);
                        }
                        // Vektor halbieren (siehe Formel) und normieren
                        x /= 2;
                        y /= 2;
                        z /= 2;
                        // Betrag errechnen
                        b = sqrt(x * x + y * y + z * z);
                        // Mittelpunktsvector berichtigen
                        //            mp.x/=anz_kanten;
                        //            mp.y/=anz_kanten;
                        //            mp.z/=anz_kanten;
                        // Normalenvektor der Pseudoflöche fertig, Hessesche Normalenform fertig
                        delete bvec;
                        delete kanten;

                        // Ebenengleichung lösen: Der Punkt eknoten[i] liegt um den Faktor a (=Abstand)
                        // in Richtung des Normalenvektors "über" der Ebene oder anders gesagt:
                        // Die Parallele Ebene zu der eben betrachteten, die durch den Punkt eknoten[i] geht
                        // hat in der Hesseschen Normalform den Abstand 0 zu diesem. Gesucht ist a
                        // unser Zielpunkt ist dann die Projektion von MP auf diese Parallele Fläche durch eknoten[i]
                        // die zu lösende Gleichung ist (alles Vektoren):
                        // neu = mp + (<(eknoten[i]-mp),Flächenvec>*Flächenvec)/|flächenvec|^2
                        // das skalarprodukt <(eknoten[i]-mp),Flächenvec> rechne ich zuerst
                        skalar = x * (eknoten[i].data[X].value - mp.x) + y * (eknoten[i].data[Y].value - mp.y) + z * (eknoten[i].data[Z].value - mp.z);
                        /*
                                 neu[i].x = mp.x + (skalar * x)/(b*b);
                                 neu[i].y = mp.y + (skalar * y)/(b*b);
                                 neu[i].z = mp.z + (skalar * z)/(b*b);
                     */
                        // + (skalar * x)/(b*b);
                        eknoten[i].data[X].value = mp.x;
                        // + (skalar * y)/(b*b);
                        eknoten[i].data[Y].value = mp.y;
                        // + (skalar * z)/(b*b);
                        eknoten[i].data[Z].value = mp.z;
                        // das wars
                        smoothed++;
                    } // ende Rand
                    else
                    {
                        delete kanten;
                        delete flaechen;
                    }
                }
                /* 
                         else  // Knoten wird nicht geglättet
                         {

                           neu[i].x = eknoten[i].data[X].value;
                           neu[i].y = eknoten[i].data[Y].value;
                           neu[i].z = eknoten[i].data[Z].value;
                         }
               */
            }
            break;

        default:
            if (Globals.verbose == VER_DEBUG)
                printf("%s meldet: unbekannter Glaettungstyp %d.\n", funcname, typ);
            return (-1);

        } // Ende Case

        if (Globals.verbose <= VER_MAX)
            printf("%s meldet: beende Glaettungsvorgang %d von %d Knoten\n", funcname, count + 1, smoothed);

        // Jetzt neue xyz-Werte eintragen
        if (typ != SM_QUALITY && typ != SM_GRID)
        {
            for (i = 0; i < anz_eknoten; ++i)
            {
                eknoten[i].data[X].value = neu[i].x;
                eknoten[i].data[Y].value = neu[i].y;
                eknoten[i].data[Z].value = neu[i].z;
            }
        }
    } // Schleife ueber alle Durchlaeufe

    delete neu;
    return (smoothed);
}

/*

// Abgewandert nach writeout.cpp

// =======================================================
// WriteAs
// =======================================================
// Verteilt die Aufgaben an die Schreibunterroutinen
int RELDATA::WriteAs(char *project,FORMAT fmt)
{
  // Gitter anpassen
if (Globals.breakdownto!=MAXINT)
{
if (BreakDownFaces(Globals.breakdownto)<0)
{
if (Globals.verbose!=VER_NONE)
{
printf("Fehler beim Zerlegen der Flaechen!\n");
}
return(-1);
}
}

// Dann Glaetten
if (Globals.smoothtyp!=SM_NONE)
SurfaceSmooth((SMOOTHTYP)(Globals.smoothtyp),Globals.smoothruns);

// und rausschreiben
switch(fmt)
{
case FMT_ANSYS:
return(WriteANSYS(project));

case FMT_GAMBIT:
return(WriteGAMBIT(project));

case FMT_POVRAY:
CalculatePhongNormals();
return(WritePOVRAY(project));

case FMT_ASCII:
case FMT_STL:
case FMT_VRML:
if (Globals.verbose!=VER_NONE)
printf("Bisher nicht implementiert fuer Ausgabe nach Format %s!\n",formatstring[Globals.writeas]);
break;
}
return(0);
}
*/

// =======================================================
// ReadData()
// =======================================================
// Liest alle Daten aus den Relationsfiles aus
int RELDATA::ReadData(char *filename)
{
    const char *funcname = "RELDATA::ReadData";
    //  int nodes,edges,faces;
    char edgename[256], facename[256], nodename[256];
    ;
    int i, k, *pliste;
    clock_t start, end;

    // test
    int *used, cnt, index;

    sprintf(nodename, "%s.nodes", filename);
    sprintf(edgename, "%s.edge2node", filename);
    sprintf(facename, "%s.face2edge", filename);

    // Elemente Top->Down einlesen
    // Flaechen einlesen
    start = clock();
    startfaces = ReadElementData(facename, ELEM_FACE, NULL, 0);
    end = clock();
    //  if (startfaces==-1) // Es gibt keinen face2edge-File, mal mit den Kanten versuchen
    //    return(-1);

    used = NULL;
    cnt = 0;
    if (startfaces > 0)
    {
        WriteText(VER_NORMAL, "Methode %s meldet: %10d Flaechen gelesen [%6.3f sec]\n", funcname, startfaces, (double)(end - start) / CLOCKS_PER_SEC);

        // Jetzt einen Array erstellen, der alle benutzen IDs der Kanten beinhalten
        for (i = 0; i < anz_eflaechen; ++i)
        {
            cnt += eflaeche[i].e_anz;
        }
        WriteText(VER_DEBUG, "%d Kanten verwendet.\n", cnt);

        start = clock();
        used = new int[cnt];
        index = 0;
        for (i = 0; i < anz_eflaechen; ++i)
            for (k = 0; k < eflaeche[i].e_anz; k++)
                used[index++] = abs(eflaeche[i].element[k].id);
        end = clock();
        WriteText(VER_MAX, "Dauer der Erstellung des ID Arrays : %8.4f Sekunden\n", (double)(end - start) / CLOCKS_PER_SEC);

        start = clock();
        used = DeleteDoubleIDs(used, &cnt);
        end = clock();
        WriteText(VER_MAX, "Dauer der Sortierung des ID Arrays : %8.4f Sekunden bei %d Elementen\n", (double)(end - start) / CLOCKS_PER_SEC, cnt);
    }

    // Kanten einlesen
    start = clock();
    startedges = ReadElementData(edgename, ELEM_EDGE, used, cnt);
    end = clock();
    if (startedges == -1)
        return (-1);
    WriteText(VER_NORMAL, "Methode %s meldet: %10d Kanten gelesen   [%6.3f sec]\n", funcname, startedges, (double)(end - start) / CLOCKS_PER_SEC);
    delete used;

    cnt = 0;
    // Nochmal für Kanten
    for (i = 0; i < anz_ekanten; ++i)
    {
        cnt += ekante[i].e_anz;
    }
    WriteText(VER_DEBUG, "%d Knoten verwendet.\n", cnt);

    start = clock();
    used = new int[cnt];
    index = 0;
    for (i = 0; i < anz_ekanten; ++i)
        for (k = 0; k < ekante[i].e_anz; k++)
            used[index++] = abs(ekante[i].element[k].id);
    end = clock();
    WriteText(VER_MAX, "Dauer der Erstellung des ID Arrays : %8.4f Sekunden\n", (double)(end - start) / CLOCKS_PER_SEC);
    // Das Ganze jetzt sortieren
    start = clock();
    //  used = MergeSort(used,cnt);
    used = DeleteDoubleIDs(used, &cnt);
    end = clock();
    WriteText(VER_MAX, "Dauer der Sortierung des ID Arrays : %8.4f Sekunden bei %d Elementen\n", (double)(end - start) / CLOCKS_PER_SEC, cnt);

    // Knoten einlesen
    start = clock();
    startnodes = ReadElementData(nodename, ELEM_NODE, used, cnt);
    end = clock();
    if (startnodes == -1)
        return (-1);
    WriteText(VER_NORMAL, "Methode %s meldet: %10d Knoten gelesen   [%6.3f sec]\n", funcname, startnodes, (double)(end - start) / CLOCKS_PER_SEC);
    delete used;
    // alles drin!

    // Jetzt die maximalen IDs erkennen und fpr CreateElement speichern
    nextnodeid = GetMaxID(ELEM_NODE) + 1;
    nextedgeid = GetMaxID(ELEM_EDGE) + 1;
    nextfaceid = GetMaxID(ELEM_FACE) + 1;
    nextcellid = GetMaxID(ELEM_CELL) + 1;

    // Das Reconnecten sparen wir uns erst mal...

    // Alle Felder sortieren
    end = SortElement(ELEM_NODE, SORT_MERGE);
    WriteText(VER_DEBUG, "Methode %s meldet: Das Sortieren der Knoten dauerte %8.4f Sekunden\n", funcname, ((double)end) / CLOCKS_PER_SEC);

    end = SortElement(ELEM_EDGE, SORT_MERGE);
    WriteText(VER_DEBUG, "Methode %s meldet: Das Sortieren der Kanten dauerte %8.4f Sekunden\n", funcname, ((double)end) / CLOCKS_PER_SEC);

    if (anz_eflaechen > 0)
    {
        end = SortElement(ELEM_FACE, SORT_MERGE);
        WriteText(VER_DEBUG, "Methode %s meldet: Das Sortieren der Flaechen dauerte %8.4f Sekunden\n", funcname, ((double)end) / CLOCKS_PER_SEC);
    }

    // Alle Elemente miteinander verbinden
    Reconnect(ELEM_ALL);
    // Alle verbundenen Elemente als benutzt markieren
    // Overload Version für eine gesamte Ebene (ungenutzte Flächen gibt es zu
    // diesem Zeitpunkt noch nicht!
    ChangeFlag(ELEM_FACE, FLAG_USED, TRUE, TRUE);

    // Kanten der Flächen im Uhrzeigersinn eintragen
    // Merke: GetNodes liefert bei einer Fläche ebenfalls die Kanten in Reihenfolge mit!
    // Änderung ist nicht so krass, da ja nur die Reihenfolge vertauscht wird
    for (i = 0; i < anz_eflaechen; ++i)
    {
        pliste = GetNodes(&eflaeche[i]);
        for (k = 0; k < eflaeche[i].e_anz; ++k)
        {
            // indices vertauschen
            eflaeche[i].element[k].id = pliste[eflaeche[i].e_anz + k];
            eflaeche[i].element[k].ptr = GetElementPtr(eflaeche[i].element[k].id, ELEM_EDGE);
            if (Globals.verbose != VER_NONE && eflaeche[i].element[k].ptr == NULL)
            {
                printf("Die Methode %s meldet: Pointer nicht gesetzt fuer Kante %d, Flaeche %d!\n", funcname, eflaeche[i].element[k].id, eflaeche[i].id);
            }
        }
        // Jetzt sind alle Kanten logisch sortiert
        delete pliste;
    }

    return (0);
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// I/O-Routinen
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/*

// =======================================================
// ReadElementData()
// =======================================================
// Liest alle Knotenpunkte aus den Relationsfiles aus
// sortiert alle Knoten nach ID (kleine zuerst)
int RELDATA::ReadElementData(char *filename, ELEMENTTYPE etype)
{
  const char *funcname="RELDATA::ReadElementData";
  char line[257];
int i,linenumber=0,fertig=0,k;
int *dataorder=NULL;
int data_per_line=-1;
int index=0;
int count,ptype;
int ivalue;
//  int exitsort;
int datacount=0;
int elemcount=0;
int argcount;
int anz_elemente=-1;
double fvalue;
ELEMENT actelem;
ELEMENT *acttab;  // zeigt auf die aktuelle Tabelle (z.B. eknoten)
PTR_ELEM *ptrelem;
PTR_DATA tempdata;
FILE *fp;

clock_t start,end,sum=0;

if ((fp = fopen(filename,"r"))==0)
{
if (Globals.verbose != VER_NONE)
printf("Methode %s meldet Fehler: File nicht gefunden [%s]\n",funcname,filename);
return(-1);
}

// Jetzt mal den File parsen nach "! end of header"
fseek(fp,0,SEEK_SET);

while(fertig==0 && fgets(line,256,fp)!=NULL)
{
linenumber++;
// Zeile durchstoebern
i=0;
// Leerzeichen loeschen
while(line[i]==' ') ++i;
// gefundenes Zeichen interpretieren
switch(line[i])
{
case '!':
++i;
// es folgt ein Befehl
// Leerzeichen loeschen
while(line[i]==' ') ++i;
// Befehl "end_of_header" suchen
if (strncmp(line+i,"end_of_header",13)==0)
{
// Naechste Zeile kann Daten beinhalten
fertig=1;
}
else if (strncmp(line+i,"number_of_data=",15)==0)
{
// Schluesselwort ueberspringen
i+=15;
// leerzeilen ueberspringen
while(line[i]==' ') ++i;
// Zahl auslesen
anz_elemente = atoi(line+i);
if (Globals.verbose==VER_DEBUG)
printf("Methode %s meldet: Elementzahl ist %d\n",funcname,anz_elemente);
}
else if (strncmp(line+i,"property_numbers=",17)==0)
{
// Jetzt mal die Reihenfolge rausbekommen
data_per_line = GetDataNameOrder(line+i+17,&dataorder);
}
break;

case '#':
// es folgt ein Kommentar
break;

case '\n':
// Leerzeile
break;

case '\0':
// EOF, fertig
break;
}
}

// Sicherheitsabfragen
if (fertig==0)
{
if (Globals.verbose!=VER_NONE)
printf("Methode %s meldet Fehler: Schluesselwort end_of_header nicht gefunden [%d] Zeilen gelesen\n",funcname,linenumber);
return(-1);
}

if (anz_elemente==-1)
{
if (Globals.verbose!=VER_NONE)
printf("Methode %s meldet Fehler: Schluesselwort number_of_data nicht gefunden [%d] Zeilen gelesen\n",funcname,linenumber);
return(-1);
}

if (data_per_line==-1)
{
if (Globals.verbose!=VER_NONE)
printf("Methode %s meldet Fehler: die Datentypen (property_numbers)\n",funcname);
printf("konnten im File %s nicht identifiziert werden\n",filename);
return(-1);
}

// Jetzt nur noch Kommentare oder Daten zu erwarten
switch(etype)
{
case ELEM_NODE:
// Kotenstruktur initialisieren
eknoten = new struct ELEMENT[anz_elemente];
anz_eknoten = anz_elemente;
// alles auf Null. Id=0 bedeutet: nicht benutzt
memset(eknoten,0,sizeof (struct ELEMENT)*anz_elemente);
acttab = eknoten; // im Folgenden werden knoten bearbeitet
break;

case ELEM_EDGE:
ekante = new struct ELEMENT[anz_elemente];
anz_ekanten = anz_elemente;
// alles auf Null. Id=0 bedeutet: nicht benutzt
memset(ekante,0,sizeof (struct ELEMENT)*anz_elemente);
acttab = ekante; // im Folgenden werden knoten bearbeitet
break;

case ELEM_FACE:
eflaeche = new struct ELEMENT[anz_elemente];
anz_eflaechen = anz_elemente;
// alles auf Null. Id=0 bedeutet: nicht benutzt
memset(eflaeche,0,sizeof (struct ELEMENT)*anz_elemente);
acttab = eflaeche; // im Folgenden werden knoten bearbeitet
break;

case ELEM_CELL:
break;
}

// File parsen wieder aufnehmen
fertig=0;
index=0;
while(index < anz_elemente && fgets(line,256,fp)!=NULL)
{
linenumber++;
// Zeile durchstoebern
i=0;
// Leerzeichen loeschen
while(line[i]==' ') ++i;
// gefundenes Zeichen interpretieren
switch(line[i])
{
case '#' : // es folgt ein Kommentar
case '\n': // Leerzeile
break;

default:
switch(etype)
{
case ELEM_NODE:

// die Zeile wird jetzt interpretiert. Die Form der Zeile steht in
// der Schluesselwortzeile property_numbers= xx xx xx xx xx
// Reihenfolge von PT_X, PT_Y etc steht in dataorder, die Anzahl der Daten pro Zeile in data_per_line
// jetzt erst mal die Struktur erstellen
memset(&actelem,0,sizeof(ELEMENT));
actelem.d_anz  = data_per_line-1;   // index wird abgezogen
actelem.typ    = etype;
actelem.data   = new PTR_DATA[actelem.d_anz];

datacount=0;
for (count=0;count<data_per_line;++count)
{
ptype =GetDataType(dataorder[count]);
// aktuelle Leerzeichen loeschen
while(line[i]==' ') ++i;

switch(ptype)
{
case PT_FP:
fvalue = atof(line+i);
// Zahl ueberspringen
while(isfloat(*(line+i))) ++i;
break;

case PT_INT:
if (Globals.verbose!=VER_NONE)
printf("Methode %s meldet: Property_type integer noch nicht implementiert\n",funcname);
return(-1);

case PT_INDEX:          // FIXME: nix passiert
ivalue = atoi(line+i);
// Zahl ueberspringen
while(isdigit(*(line+i))) ++i;
break;

default:
if (Globals.verbose==VER_DEBUG)
printf("Methode %s meldet Fehler: unbekannter PROPERTY_TYPE [%d] fuer prop %d in Zeile %d\n",funcname,ptype,count,linenumber);
return(-1);
}
// jetzt die gefundene Zahl zuweisen:
switch(dataorder[count])
{
case PN_INDEX:
actelem.id = ivalue;
break;

case PN_X:
case PN_Y:
case PN_Z:
case PN_DENSITY:
case PN_GROUP:
actelem.data[datacount].typ     = dataorder[count];
actelem.data[datacount++].value = fvalue;
break;
}
}
break;

case ELEM_EDGE:
case ELEM_FACE:
// Die Datenzeile hat immer den Typ
// [anz Daten in Zeile] [Index] [Daten]
// also erst mal die Anzahl der Daten rausbekommen
argcount = atoi(line+i)+1;  // Zahl selber zaehlt mit
// Zahl ueberspringen
while(isdigit(*(line+i))) ++i;
// Zahl der Daten und der Elemente ermitteln
if (GetDataCount(line+i,dataorder,&datacount, &elemcount)==-1)
return(-1);

memset(&actelem,0,sizeof(ELEMENT));
actelem.d_anz  = datacount;
actelem.e_anz  = elemcount;
actelem.typ    = etype;

// Datacount und elemcount werden jetzt zurueckgesetzt und als Zaehler
// fuer die bereits eingesetzten Daten/Elemente verwendet
datacount=elemcount=0;

if (actelem.e_anz==0)
actelem.element = NULL;
else
actelem.element= new PTR_ELEM[actelem.e_anz];

if (actelem.d_anz==0)
actelem.data=NULL;
else
actelem.data   = new PTR_DATA[actelem.d_anz];

// count zaehlt die bearbeiteten Eintraege in einer Zeile
// faengt bei 1 an, da die erste Zahl schon bearbeitet wurde
for (count=1;count<argcount;++count)
{
ptype =GetDataType(dataorder[count]);
// aktuelle Leerzeichen loeschen
while(line[i]==' ') ++i;

switch(ptype)
{
case PT_FP:
fvalue = atof(line+i);
// Zahl ueberspringen
while(isfloat(*(line+i))) ++i;
break;

case PT_INT:
case PT_INDEX:
ivalue = atoi(line+i);
// Zahl ueberspringen
while(isfloat(*(line+i))) ++i;
break;

default:
if (Globals.verbose==VER_DEBUG)
printf("Methode %s meldet Fehler: unbekannter PROPERTY_TYPE [%d] fuer prop %d in Zeile %d\n",funcname,ptype,count,linenumber);
return(-1);
}
// jetzt die gefundene Zahl zuweisen:
// die esrte Zahl wurde ausserhalb gelesen
switch(dataorder[count])
{
case PN_INDEX:
actelem.id = ivalue;
break;

case PN_X:
case PN_Y:
case PN_Z:
case PN_DENSITY:
case PN_GROUP:
actelem.data[datacount].typ     = dataorder[count];
actelem.data[datacount++].value = fvalue;
break;

case PN_ELEM:
//                  actelem.element[elemcount].ptr  = GetElementPtr(abs(ivalue),GetPrevElementType(etype));
//                  if (actelem.element[elemcount].ptr==NULL)
//                    return(-1);

actelem.element[elemcount].id   = ivalue;
// Elemente werden nach dem einlesen reconnected
actelem.element[elemcount].ptr  = NULL;
elemcount++;
break;
}
}
break;

default:
if (Globals.verbose!=VER_NONE)
printf("Methode %s ist fuer den Elementtyp %d nicht implementiert\n",funcname,etype);
break;
}

if (elemcount<actelem.e_anz)
{
if (Globals.verbose==VER_DEBUG)
printf("%s meldet: nicht alle Elemente gelesen [%d von %d]\n",funcname,elemcount,actelem.e_anz);
return(-1);
}

if (datacount<actelem.d_anz)
{
if (Globals.verbose==VER_DEBUG)
printf("%s meldet: nicht alle Daten gelesen [%d von %d]\n",funcname,datacount,actelem.d_anz);
return(-1);
}

// Elementdaten intern sortieren
// z.B. Nodes: x,y,z
// oder Edges: von, nach
switch(etype)
{
case ELEM_NODE:
for (k=0;k<actelem.d_anz;++k)
{
if (actelem.data[k].typ==PN_X && k!=0)
{
// vertausche diesen Wert mit dem nullten
memcpy(&tempdata,&actelem.data[0],sizeof(PTR_ELEM));
memcpy(&actelem.data[0],&actelem.data[k],sizeof(PTR_ELEM));
memcpy(&actelem.data[k],&tempdata,sizeof(PTR_ELEM));
}
if (actelem.data[k].typ==PN_Y && k!=1)
{
// vertausche diesen Wert mit dem nullten
memcpy(&tempdata,&actelem.data[1],sizeof(PTR_ELEM));
memcpy(&actelem.data[1],&actelem.data[k],sizeof(PTR_ELEM));
memcpy(&actelem.data[k],&tempdata,sizeof(PTR_ELEM));
}
if (actelem.data[k].typ==PN_Z && k!=2)
{
// vertausche diesen Wert mit dem nullten
memcpy(&tempdata,&actelem.data[2],sizeof(PTR_ELEM));
memcpy(&actelem.data[2],&actelem.data[k],sizeof(PTR_ELEM));
memcpy(&actelem.data[k],&tempdata,sizeof(PTR_ELEM));
}
}
break;

case ELEM_EDGE:
if (actelem.element[0].id <0) // nach ist zuerst eingetregen!
{
ptrelem = new PTR_ELEM; // Siehe Code oben
ptrelem->id = actelem.element[0].id;
ptrelem->ptr = actelem.element[0].ptr;
actelem.element[0].id = actelem.element[1].id;
actelem.element[0].ptr = actelem.element[1].ptr;
actelem.element[1].id = ptrelem->id;
actelem.element[1].ptr  = ptrelem->ptr;
delete ptrelem;
}
// jetzt ist immer von >0 und nach <0
break;

case ELEM_FACE: // FIXME: sollte immer auf Ausgabeobjekt (z.B. auch Zelle) gehen
// Fuer Oberflaechen: alle genutzen Kanten und Knoten als "benutzt" melden
//ChangeFlag(&actelem,FLAG_USED,TRUE,TRUE); // später das!
// jetzt noch die Kanten sortieren
break;
}

if (Globals.verbose==VER_DATAOUT)
{
printf("Element %d: Daten: %d Elemente: %d  ",actelem.id, actelem.d_anz,actelem.e_anz);
printf("Elemente: ");
for (k=0;k<actelem.e_anz;++k)
printf(" %d ",actelem.element[k].ptr->id*actelem.element[k].id);
printf("  Daten   : ");
for (k=0;k<actelem.d_anz;++k)
printf(" %9.6f ",actelem.data[k].value);
printf("\n");
}

// Ersetzt durch SortElement() nach dem Einlesen
// Knoten einsortieren:
// erst mal langsamen Bubblesort
// Das hier dauert!
start = clock();
CopyElement(&acttab[index],&actelem);
end = clock();
sum +=(end-start);
// Also nur: einfügen
//        CopyElement(&acttab[index],&actelem);

// actnode loeschen
if (actelem.d_anz>0)
delete actelem.data;
if (actelem.e_anz>0)
delete actelem.element;

actelem.data=NULL;
actelem.element=NULL;

index++;
break;
}
}

if (Globals.verbose==VER_DATAOUT)
{
switch(etype)
{
case ELEM_NODE:
printf("Schreibe Knoteninformationen:\n");
acttab = eknoten;
break;

case ELEM_EDGE:
printf("Schreibe Kanteninformationen:\n");
acttab = ekante;
break;

case ELEM_FACE:
printf("Schreibe Flaecheninformationen:\n");
acttab = eflaeche;
break;

case ELEM_CELL:
printf("Schreibe Zelleninformationen:\n");
acttab = ezelle;
break;
}
for (k=0;k<anz_elemente;++k)
{

printf("Element %d: Daten: %d Elemente: %d  ",acttab[k].id, acttab[k].d_anz, acttab[k].e_anz);
printf("Elemente: ");
for (i=0;i<acttab[k].e_anz;++i)
printf(" %d ",acttab[k].element[i].ptr->id*acttab[k].element[i].id);
printf("  Daten   : ");
for (i=0;i<acttab[k].d_anz;++i)
printf(" %9.6f ",acttab[k].data[i].value);
printf("\n");

}
}

// Tabelle wieder killen
delete dataorder;
fclose(fp);
return(index);
}
*/

/*
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Ausgaberoutinen
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
int RELDATA::WriteGAMBIT(char *project)
{
  const char *funcname="RELDATA::WriteGAMBIT";
  int nodeswritten=0;
  int faceswritten=0;
  int i;
  int *p;
char filename[256];
FILE *fp;

sprintf(filename,"%s.neu",project);
if ((fp=fopen(filename,"wt"))==NULL)
{
if (Globals.verbose!=VER_NONE)
printf("%s meldet: Fehler beim Erstellen des files %s\n",funcname,filename);
return(-1);
}

// File ist offen und Bereit:
fprintf(fp,"%20s\n","CONTROL INFO");
fprintf(fp,"%20s\n","** GAMBIT NEUTRAL FILE");
fprintf(fp,"%s\n","Beispiel");
fprintf(fp,"PROGRAM:  %20s     VERSION:%f5.2\n","RELCONV",1.0);
fprintf(fp,"%s     %8s\n","01 Jul 2001","11:11:11");
fprintf(fp,"     NUMNP     NELEM     NGRPS    NBSETS     NDFCD     NDFVL\n");
fprintf(fp," %9i %9i %9i %9i %9i %9i\n",GetFlagUsage(FLAG_USED,ELEM_NODE),GetFlagUsage(FLAG_USED,ELEM_NODE),0,0,3,3);
fprintf(fp,"ENDOFSECTION\n");
fprintf(fp,"%20s\n","NODAL COORDINATES");

if (Globals.verbose<=VER_MAX)
printf("Knoten nicht geschrieben: ");

for (i=0;i<anz_eknoten;++i)
{
if (eknoten[i].flags & FLAG_USED)
{
fprintf(fp,"%10d %19.12e %19.12e %19.12e\n",eknoten[i].id,eknoten[i].data[0].value,eknoten[i].data[1].value,eknoten[i].data[2].value);
nodeswritten++;
}
else if (Globals.verbose<=VER_MAX)
printf("%d ",eknoten[i].id);
}
fprintf(fp,"ENDOFSECTION\n");
fprintf(fp,"%20s\n","ELEMENTS/CELLS");

if (Globals.verbose<=VER_MAX)
printf("\n");

else if (Globals.verbose<=VER_MAX)
printf("Nicht geschriebene Flaechen: ");
faceswritten=0;
for (i=0;i<anz_eflaechen;++i)
{
if (eflaeche[i].flags & FLAG_USED)
{
if (eflaeche[i].e_anz==3)
{
// Punkte besorgen
p = GetNodes(&eflaeche[i]);
// Daten schreiben
faceswritten++;
fprintf(fp,"%8d %2d %2d %8d%8d%8d\n",faceswritten,3,3,p[0],p[1],p[2]);
delete p;
}
else if (Globals.verbose!=VER_NONE)
{
printf("Die Methode %d meldet: Nicht zerlegte Fläche: %d\n",eflaeche[i].id);
}
}
else if (Globals.verbose<=VER_MAX)
{
printf("%d ",eflaeche[i].id);
}
}
fprintf(fp,"ENDOFSECTION\n");

if (Globals.verbose<=VER_MAX)
printf("\n");

fclose(fp);
if (Globals.verbose<=VER_NORMAL)
printf("%s meldet: %d Knoten und %d Flaechen geschrieben.\n",funcname,nodeswritten, faceswritten);
return(0);
}

int RELDATA::WriteANSYS(char *project)
{
const char *funcname="RELDATA::WriteANSYS";
int i,offset;
int faceswritten;
int nodeswritten;
int *p;
int dreiecke=0,vierecke=0;
char filename[256];
FILE *fp;

sprintf(filename,"%s.cdb",project);
if ((fp=fopen(filename,"wt"))==NULL)
{
if (Globals.verbose!=VER_NONE)
printf("%s meldet: Fehler beim Erstellen des files %s\n",funcname,filename);
return(-1);
}

// File ist offen und Bereit:

// Kommentar schreiben
fprintf(fp,"/COM ***************************************************************************\n");
fprintf(fp,"/COM File erstellt mit RELCONV, Universitaet Stuttgart\n");
fprintf(fp,"/COM Fuer ANSYS 5.6.2\n");
fprintf(fp,"/COM ***************************************************************************\n");
fprintf(fp,"/COM\n");
fprintf(fp,"/PREP7                                ! schaltet in den Praeprozessor\n");
fprintf(fp,"/NOPR                                 ! unterdruecke Fileausgabe beim Lesen\n");
fprintf(fp,"/COM Angaben, was bei Einlesen weiterer Daten geschehen soll\n");
fprintf(fp,"*IF,_CDRDOFF,EQ,1,THEN                ! fuer SOLID model\n");
fprintf(fp,"_CDRDOFF=                             ! RESET Flag, Offset schon angegeben\n");
fprintf(fp,"*ELSE                                 ! Offsets angeben\n");
fprintf(fp,"NUMOFF,NODE,%15d\n",anz_eknoten+1);   // Das hier ist noch falsch! FIXME
fprintf(fp,"NUMOFF,ELEM,%15d\n",anz_eflaechen+1); // Das hier ist noch falsch! FIXME
fprintf(fp,"NUMOFF,MAT, %15d\n",1);
fprintf(fp,"NUMOFF,REAL,%15d\n",1);
fprintf(fp,"NUMOFF,TYPE,%15d\n",1);
fprintf(fp,"*ENDIF\n");
fprintf(fp,"/COM Elementtyp auf SHELL63 setzen\n");
fprintf(fp,"ET,1,63\n");
fprintf(fp,"/COM ***************************************************************************\n");
fprintf(fp,"/COM Beginn der Datenbeschreibung\n");
fprintf(fp,"/COM ***************************************************************************\n");
fprintf(fp,"/COM\n");
fprintf(fp,"/COM ==========================================\n");
fprintf(fp,"/COM Knotendaten\n");
fprintf(fp,"/COM ==========================================\n");
fprintf(fp,"NBLOCK,6,SOLID                        ! Knotenblock mit je 6 Daten pro Zeile\n");
fprintf(fp,"(3i8,6e16.9)                          ! Formatierung\n");
nodeswritten=0;

if (Globals.verbose<=VER_MAX)
printf("unbenutzte Knoten: ");
for (i=0;i<anz_eknoten;++i)
{
if (eknoten[i].flags & FLAG_USED)
{
nodeswritten++;
fprintf(fp,"%8d%8d%8d%16.9f%16.9f%16.9f\n",eknoten[i].id,0,0,eknoten[i].data[0].value,eknoten[i].data[1].value,eknoten[i].data[2].value);
}
else if (Globals.verbose<=VER_MAX)
printf("%d ",eknoten[i].id);
}
if (Globals.verbose<=VER_MAX)
printf("\n");
fprintf(fp,"N,R5.3,LOC,-1,                        ! Ende Markierung\n");
fprintf(fp,"/COM Ende Knotendaten\n");
fprintf(fp,"/COM\n");
fprintf(fp,"/COM ==========================================\n");
fprintf(fp,"/COM Flaechendaten\n");
fprintf(fp,"/COM ==========================================\n");
fprintf(fp,"EBLOCK,19,SOLID                       ! Flaechenblockdaten, 19 je Zeile\n");
fprintf(fp,"(19i7)                                ! Formatierung\n");
offset=0;
faceswritten=0;
if (Globals.verbose<=VER_MAX)
printf("unbenutzte Flaechen: ");

for (i=0;i<anz_eflaechen;++i)
{
if (eflaeche[i].flags & FLAG_USED)
{
if (eflaeche[i].e_anz==3)
{
// Punkte besorgen
p = GetNodes(&eflaeche[i]);
// Daten schreiben
faceswritten++;
fprintf(fp,"%7d%7d%7d%7d%7d%7d%7d%7d%7d%7d%7d%7d%7d%7d%7d\n",1,1,1,1,0,0,0,0,4,0,faceswritten,p[0],p[1],p[2],p[2]);
dreiecke++;
delete p;
}
else if (eflaeche[i].e_anz==4)
{
// Punkte besorgen
p = GetNodes(&eflaeche[i]);
// Daten schreiben
faceswritten++;
fprintf(fp,"%7d%7d%7d%7d%7d%7d%7d%7d%7d%7d%7d%7d%7d%7d%7d\n",1,1,1,1,0,0,0,0,4,0,faceswritten,p[0],p[1],p[2],p[3]);
vierecke++;
delete p;
}
else
{
if (Globals.verbose==VER_DEBUG)
{
printf("%s meldet: Eine nicht zerlegte Flaeche gefunden %d\n",funcname,eflaeche[i].id);
printf("Diese Flaeche wird nicht geschrieben.\n");
}
}
}
else
{
if (Globals.verbose<=VER_MAX)
printf("%d ",eflaeche[i].id);
}
}
if (Globals.verbose<=VER_MAX)
printf("\n");

fprintf(fp,"-1                                    ! Ende Markierung\n");
fprintf(fp,"/COM Ende Flaechendaten\n");
fprintf(fp,"/COM\n");
fprintf(fp,"/COM ==========================================\n");
fprintf(fp,"/COM Komponentendefinitionen\n");
fprintf(fp,"/COM ==========================================\n");
fprintf(fp,"CMBLOCK,%s ,%s , 2\n","ELEM","shell");
fprintf(fp,"(8i10)                                ! Formatierung\n");
fprintf(fp,"%10d%10d\n",1,-1*eflaeche[anz_eflaechen-1].id);
fprintf(fp,"CMBLOCK,%s ,%s , 2\n","NODE","allnodes");
fprintf(fp,"(8i10)                                ! Formatierung\n");
fprintf(fp,"%10d%10d\n",1,-1*eknoten[anz_eknoten-1].id);
fprintf(fp,"/COM Ende Komponentendefinition\n");
fprintf(fp,"/COM ***************************************************************************\n");
fprintf(fp,"/COM Fileende\n");
fprintf(fp,"/COM ***************************************************************************\n");
fprintf(fp,"/GOPR\n");
fprintf(fp,"EPLOT                                 ! Elemente Darstellen\n");

fclose(fp);
if (Globals.verbose<=VER_NORMAL)
{
printf("%s meldet: %d Knoten und %d Flaechen (3:%d 4:%d) geschrieben.\n",funcname,nodeswritten, faceswritten,dreiecke,vierecke);
}
return(0);
}
*/

/*
int RELDATA::WritePOVRAY(char *project)
{
  const char *funcname="RELDATA::WritePOVRAY";
  char filename[256];
  FILE *fp;
  int i,k,*nodes,unused;
  int triangles=0;
  double ko[6];   // immer x,y,z,NX,NY,NZ
  ELEMENT *actnode;
  double min[3],max[3],off[3],loc[3]={0,0,0};;

sprintf(filename,"%s.pov",project);
if ((fp=fopen(filename,"wt"))==NULL)
{
if (Globals.verbose!=VER_NONE)
printf("%s meldet: Fehler beim Erstellen des files %s\n",funcname,filename);
return(-1);
}

// Jetzt erst mal ein paar Eckdaten besorgen
min[X]=min[Y]=min[Z]=+999999;
max[X]=max[Y]=max[Z]=-999999;
for (i=0;i<anz_eknoten;++i)
{
if (eknoten[i].flags & FLAG_USED)
{
max[X] = MAX(max[X],eknoten[i].data[X].value);
min[X] = MIN(min[X],eknoten[i].data[X].value);
max[Y] = MAX(max[Y],eknoten[i].data[Y].value);
min[Y] = MIN(min[Y],eknoten[i].data[Y].value);
max[Z] = MAX(max[Z],eknoten[i].data[Z].value);
min[Z] = MIN(min[Z],eknoten[i].data[Z].value);
}
}

// Object auf (0,0,0) zentrieren, dazu Ortsvector des Mittelpunkts errechnen
off[X] = min[X]+(max[X]-min[X])/2;
off[Y] = min[Y]+(max[Y]-min[Y])/2;
off[Z] = min[Z]+(max[Z]-min[Z])/2;

// Kameraposition
loc[X] = 0;
loc[Y] = 0;
loc[Z] = -100;

if (Globals.verbose!=VER_NONE)
{
printf("Objektausmasse    : Breite(X) = %8.4f Hoehe(Y) = %8.4f Tiefe(Z) = %8.4f\n",max[X]-min[X],max[Z]-min[Z],max[Z]-min[Z]);
printf("Mittelpunktsvector: <%8.4f,%8.4f,%8.4f>\n",off[X],off[Y],off[Z]);
printf("Kameraposition    : <%8.4f,%8.4f,%8.4f>\n",loc[X],loc[Y],loc[Z]);
}

// File ist offen und Bereit:
fprintf(fp,"global_settings {\n  ambient_light rgb <1,1,1>\n  }\n");
fprintf(fp,"camera {\n  location <%8.4f,%8.4f,%8.4f>\n",loc[X],loc[Y],loc[Z]);
//  fprintf(fp,"  look_at <%8.4f,%8.4f,%8.4f>\n  rotate <0,0,0>}\n",off[X],off[Y],off[Z]);
fprintf(fp,"  look_at <0,0,0>\n  rotate <0,0,0>}\n");
fprintf(fp,"light_source { <0,0,%f> color rgb<1,1,1> }\n",loc[Z]);
fprintf(fp,"light_source { <%f,0,0> color rgb<1,1,1> }\n",loc[Z]);
fprintf(fp,"#declare RED = texture {\n  pigment { color rgb<1.0,0.0,0.0>}\n  finish {ambient 0.2 diffuse 0.5 }\n}\n");
// Alternative: Mit Texture-Mapping
fprintf(fp,"//#declare RED = texture {\n");
fprintf(fp,"//  pigment  {\n//    image_map {\n//      gif \"dicom.gif\"\n");
fprintf(fp,"//      map_type 1\n//      once\n//      interpolate 2\n    }\n//    }\n//    finish{ambient 0.3}\n//  }\n");

fprintf(fp,"mesh {\n");
unused=0;
for (i=0;i<anz_eflaechen;++i)
{
if (eflaeche[i].flags & FLAG_USED && eflaeche[i].e_anz==3)
{
nodes = GetNodes(&eflaeche[i]);
fprintf(fp,"  smooth_triangle { ");
//      fprintf(fp,"  triangle{ ");
for (k=0;k<3;++k)
{
actnode = GetElementPtr(nodes[k],ELEM_NODE);

ko[0]=actnode->data[X].value;   // originale Koordinaten Schreiben
ko[1]=actnode->data[Y].value;   // originale Koordinaten Schreiben
ko[2]=actnode->data[Z].value;   // originale Koordinaten Schreiben
ko[3]=actnode->data[actnode->d_anz-3].value;  // Normale X
ko[4]=actnode->data[actnode->d_anz-2].value;  // Normale Y
ko[5]=actnode->data[actnode->d_anz-1].value;  // Normale Z
fprintf(fp,"<%8.4f,%8.4f,%8.4f>,<%8.4f,%8.4f,%8.4f>",ko[0],ko[1],ko[2],ko[3],ko[4],ko[5]);
//        fprintf(fp,"<%8.4f,%8.4f,%8.4f>",ko[0],ko[1],ko[2]);
if (k<2)
fprintf(fp,",");
}
fprintf(fp,"}\n");
triangles++;
delete nodes;
nodes=NULL;
}
else
unused++;
}
// Das waren mal alle Dreiecke, auf die Mitte verschieben
fprintf(fp,"    texture {RED}\n    translate <%8.4f,%8.4f,%8.4f>\n  }\n",-off[X],-off[Y],-off[Z]);

fclose(fp);
if (Globals.verbose<=VER_NORMAL)
{
printf("%s meldet: %d Dreiecke geschrieben, %d Flaechen ignoriert.\n",funcname,triangles,unused);
}
return(0);
}
*/

// =======================================================
// ReadElementData()
// =======================================================
// Liest alle Knotenpunkte aus den Relationsfiles aus
// sortiert alle Knoten nach ID (kleine zuerst)
// liest nur die Elemente, die auch gebraucht werden
int RELDATA::ReadElementData(char *filename, ELEMENTTYPE etype, int *used, int anz_used)
{
    const char *funcname = "RELDATA::ReadElementData";
    char line[257];
    int i, linenumber = 0, fertig = 0, k;
    int *dataorder = NULL;
    int data_per_line = -1;
    int index = 0;
    int count, ptype;
    int ivalue;
    //  int exitsort;
    int datacount = 0;
    int elemcount = 0;
    int argcount;
    int anz_elemente = -1;
    double fvalue;
    ELEMENT actelem;
    ELEMENT *acttab; // zeigt auf die aktuelle Tabelle (z.B. eknoten)
    PTR_ELEM *ptrelem;
    PTR_DATA tempdata;
    FILE *fp;
    BOOL readall = FALSE;

    if ((fp = fopen(filename, "r")) == 0)
    {
        if (Globals.verbose != VER_NONE)
            printf("Methode %s meldet Fehler: File nicht gefunden [%s]\n", funcname, filename);
        return (-1);
    }

    // Jetzt mal den File parsen nach "! end of header"
    fseek(fp, 0, SEEK_SET);

    while (fertig == 0 && fgets(line, 256, fp) != NULL)
    {
        linenumber++;
        // Zeile durchstoebern
        i = 0;
        // Leerzeichen loeschen
        while (line[i] == ' ')
            ++i;
        // gefundenes Zeichen interpretieren
        switch (line[i])
        {
        case '!':
            ++i;
            // es folgt ein Befehl
            // Leerzeichen loeschen
            while (line[i] == ' ')
                ++i;
            // Befehl "end_of_header" suchen
            if (strncmp(line + i, "end_of_header", 13) == 0)
            {
                // Naechste Zeile kann Daten beinhalten
                fertig = 1;
            }
            else if (strncmp(line + i, "number_of_data=", 15) == 0)
            {
                // Schluesselwort ueberspringen
                i += 15;
                // leerzeilen ueberspringen
                while (line[i] == ' ')
                    ++i;
                // Zahl auslesen
                anz_elemente = atoi(line + i);
                if (Globals.verbose == VER_DEBUG)
                    printf("Methode %s meldet: Elementzahl ist %d\n", funcname, anz_elemente);
            }
            else if (strncmp(line + i, "property_numbers=", 17) == 0)
            {
                // Jetzt mal die Reihenfolge rausbekommen
                data_per_line = GetDataNameOrder(line + i + 17, &dataorder);
            }
            break;

        case '#':
            // es folgt ein Kommentar
            break;

        case '\n':
        case '\r':
            // Leerzeile
            break;

        case '\0':
            // EOF, fertig
            break;
        }
    }

    // Sicherheitsabfragen
    if (fertig == 0)
    {
        if (Globals.verbose != VER_NONE)
            printf("Methode %s meldet Fehler: Schluesselwort end_of_header nicht gefunden [%d] Zeilen gelesen\n", funcname, linenumber);
        return (-1);
    }

    if (anz_elemente == -1)
    {
        if (Globals.verbose != VER_NONE)
            printf("Methode %s meldet Fehler: Schluesselwort number_of_data nicht gefunden [%d] Zeilen gelesen\n", funcname, linenumber);
        return (-1);
    }

    if (data_per_line == -1)
    {
        if (Globals.verbose != VER_NONE)
            printf("Methode %s meldet Fehler: die Datentypen (property_numbers)\n", funcname);
        printf("konnten im File %s nicht identifiziert werden\n", filename);
        return (-1);
    }

    if (anz_used == 0)
    {
        // alle Daten einlesen, used ist eventuell ein Nullpointer
        readall = TRUE;
        anz_used = anz_elemente;
    }

    // Jetzt nur noch Kommentare oder Daten zu erwarten
    switch (etype)
    {
    case ELEM_NODE:
        // Kotenstruktur initialisieren
        //      eknoten = new struct ELEMENT[anz_elemente];
        eknoten = new struct ELEMENT[anz_used];
        anz_eknoten = anz_used;
        // alles auf Null. Id=0 bedeutet: nicht benutzt
        memset(eknoten, 0, sizeof(struct ELEMENT) * anz_eknoten);
        acttab = eknoten; // im Folgenden werden knoten bearbeitet
        break;

    case ELEM_EDGE:
        //      ekante = new struct ELEMENT[anz_elemente];
        ekante = new struct ELEMENT[anz_used];
        anz_ekanten = anz_used;
        // alles auf Null. Id=0 bedeutet: nicht benutzt
        memset(ekante, 0, sizeof(struct ELEMENT) * anz_ekanten);
        acttab = ekante; // im Folgenden werden knoten bearbeitet
        break;

    case ELEM_FACE:
        //      eflaeche = new struct ELEMENT[anz_elemente];
        eflaeche = new struct ELEMENT[anz_used];
        anz_eflaechen = anz_used;
        // alles auf Null. Id=0 bedeutet: nicht benutzt
        memset(eflaeche, 0, sizeof(struct ELEMENT) * anz_eflaechen);
        acttab = eflaeche; // im Folgenden werden knoten bearbeitet
        break;

    case ELEM_CELL:
        break;
    default:
        break;
    }

    // File parsen wieder aufnehmen
    fertig = 0;
    index = 0;
    // ACHTUNG: ich lese IMMER alle Zeilen aus dem File!
    while (index < anz_elemente && fgets(line, 256, fp) != NULL)
    {
        linenumber++;
        // Zeile durchstoebern
        i = 0;
        // Leerzeichen loeschen
        while (line[i] == ' ')
            ++i;
        // gefundenes Zeichen interpretieren
        switch (line[i])
        {
        case '#':
            // es folgt ein Kommentar
            break;

        default:
            switch (etype)
            {
            case ELEM_NODE:

                // die Zeile wird jetzt interpretiert. Die Form der Zeile steht in
                // der Schluesselwortzeile property_numbers= xx xx xx xx xx
                // Reihenfolge von PT_X, PT_Y etc steht in dataorder, die Anzahl der Daten pro Zeile in data_per_line
                // jetzt erst mal die Struktur erstellen
                memset(&actelem, 0, sizeof(ELEMENT));
                actelem.d_anz = data_per_line - 1; // index wird abgezogen
                actelem.typ = etype;
                actelem.data = new PTR_DATA[actelem.d_anz];

                datacount = 0;
                for (count = 0; count < data_per_line; ++count)
                {
                    ptype = GetDataType(dataorder[count]);
                    // aktuelle Leerzeichen loeschen
                    while (line[i] == ' ')
                        ++i;

                    switch (ptype)
                    {
                    case PT_FP:
                        fvalue = atof(line + i);
                        // Zahl ueberspringen
                        while (isfloat(*(line + i)))
                            ++i;
                        break;

                    case PT_INT:
                        if (Globals.verbose != VER_NONE)
                            printf("Methode %s meldet: Property_type integer noch nicht implementiert\n", funcname);
                        return (-1);

                    case PT_INDEX: // FIXME: nix passiert
                        ivalue = atoi(line + i);
                        // Zahl ueberspringen
                        while (isdigit(*(line + i)))
                            ++i;
                        break;

                    default:
                        if (Globals.verbose == VER_DEBUG)
                            printf("Methode %s meldet Fehler: unbekannter PROPERTY_TYPE [%d] fuer prop %d in Zeile %d\n", funcname, ptype, count, linenumber);
                        return (-1);
                    }
                    // jetzt die gefundene Zahl zuweisen:
                    switch (dataorder[count])
                    {
                    case PN_INDEX:
                        actelem.id = ivalue;
                        break;

                    case PN_X:
                    case PN_Y:
                    case PN_Z:
                    case PN_DENSITY:
                    case PN_GROUP:
                        actelem.data[datacount].typ = dataorder[count];
                        actelem.data[datacount++].value = fvalue;
                        break;
                    }
                }
                break;

            case ELEM_EDGE:
            case ELEM_FACE:
                // Die Datenzeile hat immer den Typ
                // [anz Daten in Zeile] [Index] [Daten]
                // also erst mal die Anzahl der Daten rausbekommen
                argcount = atoi(line + i) + 1; // Zahl selber zaehlt mit
                // Zahl ueberspringen
                while (isdigit(*(line + i)))
                    ++i;
                // Zahl der Daten und der Elemente ermitteln
                if (GetDataCount(line + i, dataorder, &datacount, &elemcount) == -1)
                    return (-1);

                memset(&actelem, 0, sizeof(ELEMENT));
                actelem.d_anz = datacount;
                actelem.e_anz = elemcount;
                actelem.typ = etype;

                // Datacount und elemcount werden jetzt zurueckgesetzt und als Zaehler
                // fuer die bereits eingesetzten Daten/Elemente verwendet
                datacount = elemcount = 0;

                if (actelem.e_anz == 0)
                    actelem.element = NULL;
                else
                    actelem.element = new PTR_ELEM[actelem.e_anz];

                if (actelem.d_anz == 0)
                    actelem.data = NULL;
                else
                    actelem.data = new PTR_DATA[actelem.d_anz];

                // count zaehlt die bearbeiteten Eintraege in einer Zeile
                // faengt bei 1 an, da die erste Zahl schon bearbeitet wurde
                for (count = 1; count < argcount; ++count)
                {
                    ptype = GetDataType(dataorder[count]);
                    // aktuelle Leerzeichen loeschen
                    while (line[i] == ' ')
                        ++i;

                    switch (ptype)
                    {
                    case PT_FP:
                        fvalue = atof(line + i);
                        // Zahl ueberspringen
                        while (isfloat(*(line + i)))
                            ++i;
                        break;

                    case PT_INT:
                    case PT_INDEX:
                        ivalue = atoi(line + i);
                        // Zahl ueberspringen
                        while (isfloat(*(line + i)))
                            ++i;
                        break;

                    default:
                        if (Globals.verbose == VER_DEBUG)
                            printf("Methode %s meldet Fehler: unbekannter PROPERTY_TYPE [%d] fuer prop %d in Zeile %d\n", funcname, ptype, count, linenumber);
                        return (-1);
                    }
                    // jetzt die gefundene Zahl zuweisen:
                    // die esrte Zahl wurde ausserhalb gelesen
                    switch (dataorder[count])
                    {
                    case PN_INDEX:
                        actelem.id = ivalue;
                        break;

                    case PN_X:
                    case PN_Y:
                    case PN_Z:
                    case PN_DENSITY:
                    case PN_GROUP:
                        actelem.data[datacount].typ = dataorder[count];
                        actelem.data[datacount++].value = fvalue;
                        break;

                    case PN_ELEM:
                        //                  actelem.element[elemcount].ptr  = GetElementPtr(abs(ivalue),GetPrevElementType(etype));
                        //                  if (actelem.element[elemcount].ptr==NULL)
                        //                    return(-1);

                        actelem.element[elemcount].id = ivalue;
                        // Elemente werden nach dem einlesen reconnected
                        actelem.element[elemcount].ptr = NULL;
                        elemcount++;
                        break;
                    }
                }
                break;

            default:
                if (Globals.verbose != VER_NONE)
                    printf("Methode %s ist fuer den Elementtyp %d nicht implementiert\n", funcname, etype);
                break;
            }

            if (elemcount < actelem.e_anz)
            {
                if (Globals.verbose == VER_DEBUG)
                    printf("%s meldet: nicht alle Elemente gelesen [%d von %d]\n", funcname, elemcount, actelem.e_anz);
                return (-1);
            }

            if (datacount < actelem.d_anz)
            {
                if (Globals.verbose == VER_DEBUG)
                    printf("%s meldet: nicht alle Daten gelesen [%d von %d]\n", funcname, datacount, actelem.d_anz);
                return (-1);
            }

            // wird dieses Element überhaupt benötigt?
            if (DoWeNeed(actelem.id, used, anz_used))
            {

                // Elementdaten intern sortieren
                // z.B. Nodes: x,y,z
                // oder Edges: von, nach
                switch (etype)
                {
                case ELEM_NODE:
                    for (k = 0; k < actelem.d_anz; ++k)
                    {
                        if (actelem.data[k].typ == PN_X && k != 0)
                        {
                            // vertausche diesen Wert mit dem nullten
                            memcpy(&tempdata, &actelem.data[0], sizeof(PTR_ELEM));
                            memcpy(&actelem.data[0], &actelem.data[k], sizeof(PTR_ELEM));
                            memcpy(&actelem.data[k], &tempdata, sizeof(PTR_ELEM));
                        }
                        if (actelem.data[k].typ == PN_Y && k != 1)
                        {
                            // vertausche diesen Wert mit dem nullten
                            memcpy(&tempdata, &actelem.data[1], sizeof(PTR_ELEM));
                            memcpy(&actelem.data[1], &actelem.data[k], sizeof(PTR_ELEM));
                            memcpy(&actelem.data[k], &tempdata, sizeof(PTR_ELEM));
                        }
                        if (actelem.data[k].typ == PN_Z && k != 2)
                        {
                            // vertausche diesen Wert mit dem nullten
                            memcpy(&tempdata, &actelem.data[2], sizeof(PTR_ELEM));
                            memcpy(&actelem.data[2], &actelem.data[k], sizeof(PTR_ELEM));
                            memcpy(&actelem.data[k], &tempdata, sizeof(PTR_ELEM));
                        }
                    }
                    break;

                case ELEM_EDGE:
                    if (actelem.element[0].id < 0) // nach ist zuerst eingetregen!
                    {
                        // NICHT LÖSCHEN
                        /*
                                      // In Erwartung einer Polyline:
                                      // Feld komplett umdrehen
                                      ptrelem = new PTR_ELEM[actelem.e_anz];
                                      for (k=0;k<actelem.e_anz;k++)
                                      {
                                        ptrelem[k].id = actelem.element[actelem.e_anz-1-k].id;
                                        ptrelem[k].ptr = actelem.element[actelem.e_anz-1-k].ptr;
                                      }
                                      // und reinkopieren
                                      for (k=0;k<actelem.e_anz;k++)
                        {
                        actelem.element[actelem.e_anz-1-k].id = ptrelem[k].id;
                        actelem.element[actelem.e_anz-1-k].ptr = ptrelem[k].ptr;
                        }
                        */
                        ptrelem = new PTR_ELEM; // Siehe Code oben
                        ptrelem->id = actelem.element[0].id;
                        ptrelem->ptr = actelem.element[0].ptr;
                        actelem.element[0].id = actelem.element[1].id;
                        actelem.element[0].ptr = actelem.element[1].ptr;
                        actelem.element[1].id = ptrelem->id;
                        actelem.element[1].ptr = ptrelem->ptr;
                        delete ptrelem;
                    }
                    // jetzt ist immer von >0 und nach <0
                    break;
                default:
                    break;
                }

                if (Globals.verbose == VER_DATAOUT)
                {
                    printf("Element %d: Daten: %d Elemente: %d  ", actelem.id, actelem.d_anz, actelem.e_anz);
                    printf("Elemente: ");
                    for (k = 0; k < actelem.e_anz; ++k)
                        printf(" %d ", actelem.element[k].ptr->id * actelem.element[k].id);
                    printf("  Daten   : ");
                    for (k = 0; k < actelem.d_anz; ++k)
                        printf(" %9.6f ", actelem.data[k].value);
                    printf("\n");
                }

                // Element unsortiert einfügen
                CopyElement(&acttab[index], &actelem);
                index++;
            } // Ende: doweneed
            //        else
            //          WriteText(VER_DEBUG,"%s meldet: Element %d nicht benoetigt.\n",funcname,actelem.id);

            // actnode loeschen
            if (actelem.d_anz > 0)
                delete actelem.data;
            if (actelem.e_anz > 0)
                delete actelem.element;

            actelem.data = NULL;
            actelem.element = NULL;
            break; // Ende: kein Kommentar
        }
    }

    if (Globals.verbose == VER_DATAOUT)
    {
        switch (etype)
        {
        case ELEM_NODE:
            printf("Schreibe Knoteninformationen:\n");
            acttab = eknoten;
            break;

        case ELEM_EDGE:
            printf("Schreibe Kanteninformationen:\n");
            acttab = ekante;
            break;

        case ELEM_FACE:
            printf("Schreibe Flaecheninformationen:\n");
            acttab = eflaeche;
            break;

        case ELEM_CELL:
            printf("Schreibe Zelleninformationen:\n");
            acttab = ezelle;
            break;
        default:
            break;
        }
        for (k = 0; k < anz_elemente; ++k)
        {

            printf("Element %d: Daten: %d Elemente: %d  ", acttab[k].id, acttab[k].d_anz, acttab[k].e_anz);
            printf("Elemente: ");
            for (i = 0; i < acttab[k].e_anz; ++i)
                printf(" %d ", acttab[k].element[i].ptr->id * acttab[k].element[i].id);
            printf("  Daten   : ");
            for (i = 0; i < acttab[k].d_anz; ++i)
                printf(" %9.6f ", acttab[k].data[i].value);
            printf("\n");
        }
    }

    // Tabelle wieder killen
    delete dataorder;
    fclose(fp);
    return (index);
}

BOOL RELDATA::DoWeNeed(int id, int *used, int size)
{
    if (used == NULL)
        return (TRUE);
    if (BinSearch(0, size - 1, id, used) == -1)
        return (FALSE);
    return (TRUE);
}

// =======================================================
// KillDoubleIds
// =======================================================
// löscht alle Elemente, die mindestens zweimal vorkommen.
// Es liefert also all die Elemente zurück, die nur einmal in der Liste sind
// Es läuft auch mit Mehrfacheinträgen und VZ
// Das Feld "feld" wird zerstört und ein neues Feld zurückgeliefert
// Das neue Feld hat dann genau "size" einträge
int *RELDATA::KillDoubleIDs(int *feld, int *size)
{
    int i = 0, k, j, *ret;
    int kill;
    BOOL killfirst;
    int newsize = *size;

    while (i < newsize - 1)
    {
        kill = abs(feld[i]);
        killfirst = FALSE;
        k = i + 1;
        while (k < newsize)
        {
            if (kill == abs(feld[k]))
            {
                // doppel gefunden
                // erstes Element zum Löschen markieren
                killfirst = TRUE;
                // Feld aufrollen
                for (j = k; j < newsize - 1; ++j)
                    feld[j] = feld[j + 1];
                newsize--;
            }
            else
                k++;
        }
        // Eventuell erstes Element löschen
        if (killfirst)
        {
            for (j = i; j < newsize - 1; ++j)
                feld[j] = feld[j + 1];
            newsize--;
        }
        else
            ++i;
    }
    *size = newsize;
    ret = new int[newsize];
    memcpy(ret, feld, sizeof(int) * newsize);
    delete feld;
    return (ret);
}

// =======================================================
// DeleteDoubleIDs
// =======================================================
// Sortiert ein int-Feld und löscht alle Doppelten IDs raus, so dass
// jedes Element nur noch einmal vorkommt. Sollte auch für Mehrfachvorkommen klappen
// Sollte auch für negative IDs klappen
// Das Feld "feld" wird gelöscht und ist nicht identisch mit dem Rückgabefeld
// Die Anweisung feld = DeleteDoubleIDs(feld,&size) ist zulässig
// Das zurückgelieferte Feld ist genauso gross wie des Quellfeld, es werden aber nur
// die ersten size Einträge benutzt.
int *RELDATA::DeleteDoubleIDs(int *feld, int *size)
{
    int i;
    int *dest;
    int newsize = 0;

    // für kleine Felder Bubblesort
    if (*size < 10)
        BubbleSort(feld, *size);
    else // sonst Mergesort
    {
        dest = MergeSort(feld, *size);
        memcpy(feld, dest, sizeof(int) * (*size));
        delete dest;
    }

    dest = new int[*size];
    dest[0] = feld[0];
    for (i = 1; i < *size; ++i)
    {
        if (abs(dest[newsize]) != abs(feld[i]))
        {
            dest[++newsize] = feld[i]; // ins nächste Feld eintragen
        }
    }
    *size = newsize + 1; // index->Anzahl umrechnen
    delete feld;
    return (dest);
}

/*
  for (i=0;i<max-1;++i)
  {
    while(i<max-1 && feld[i]==feld[i+1] )
    {
      // Feld vorrücken
      for (k=i+1;k<max-1;++k)
      {
        feld[k]=feld[k+1];
      }
      // letztes Feld löschen
feld[max-1]=0;
max--;  // obere Grenze eins runter
}
}
return(max);
}
*/

// =======================================================
// CreateArea
// =======================================================
// Liefert eine Kantenliste (in "richtiger" Reihenfolge), die
// ein Gebiet umschliesst das den Vorgaben enspricht
// Maxlocalarc gibt den maximalen Knickwinkel von zwei
// benachbarten Flächen an, maxglobalarc den maximalen Knickwinkel
// zwischen allen Flächen in dem Gebiet.
// Die zurückgelieferte Liste Hat an erster Stelle [0] die
// Anzahl der folgenden Elemente.
// Die Liste ist im Uhrzeigersinn sortiert
int *RELDATA::CreateArea(ELEMENT *elem, double globalarc, const double maxlocalarc, const double maxglobalarc)
{
    //Generates C4189
    //const char *funcname="RELDATA::CreateArea";
    int *ret = NULL, *dummy, *subret, edgecount = 0;
    int i, k;
    double arc;
    ECHAIN *root, *act;

    if (elem->flags & FLAG_AREA || !(elem->flags & FLAG_USED))
        return (NULL);

    // Kantenliste erst mal kopieren
    edgecount = elem->e_anz;
    ret = new int[edgecount + 1];
    ret[0] = edgecount;
    for (i = 0; i < elem->e_anz; ++i)
    {
        ret[i + 1] = elem->element[i].id;
    }
    elem->flags |= FLAG_AREA;

    // Nachbarn zu dieser Fläche ermitteln
    root = GetNeighbours(elem, 1);
    // Jetzt die Liste der Nachbarn durchgehen und den Winkeltest machen
    act = root;
    while (root != NULL)
    {
        if (!(root->element->flags & FLAG_AREA) && root->element->flags & FLAG_USED && elem != root->element)
        {
            arc = fabs(GetFaceAngle(elem, root->element));
            WriteText(VER_DATAOUT, "Winkel zwischen den Flaechen %d und %d: %8.4f Grad, global: %8.4f (max: %8.4f)\n", elem->id, root->id, arc, globalarc, maxglobalarc);
            if (arc <= maxlocalarc && globalarc + arc <= maxglobalarc)
            {
                // Diese Fläche passt!
                // Kriterien auch für diese Fläche testen:
                subret = CreateArea(root->element, arc + globalarc, maxlocalarc, maxglobalarc);
                if (subret != NULL)
                {
                    // Gebietsflag setzen
                    root->element->flags |= FLAG_AREA;
                    // Gebiet löschen
                    ChangeFlag(root->element, FLAG_USED, FALSE, FALSE);
                    // Quellelement markieren
                    elem->flags |= FLAG_OPT;

                    // Ergebnis dieser Suche in die bestehende Kantenliste aufnehmen
                    edgecount += subret[0];
                    dummy = new int[edgecount + 1];
                    memcpy(dummy + 1, ret + 1, sizeof(int) * ret[0]);
                    memcpy(dummy + ret[0] + 1, subret + 1, sizeof(int) * subret[0]);
                    dummy[0] = edgecount;
                    // Alte Felder löschen
                    delete subret;
                    delete ret;
                    ret = dummy;
                    dummy = NULL;
                }
            }
        } // Element gehörte schon zu einer Fläche
        // Dieses Element löschen
        act = root->next;
        delete root;
        root = act;
    }
    // in ret[0]steht jetzt die Länge des Feldes
    edgecount = ret[0];
    // ID Liste jetzt sortieren und doppelte komplett (beide Einträge!) löschen
    // doppelte Einträge sind innere Kanten
    // Obacht: edgecount ist eins kleiner als die größe von ret!!
    if (ret[0] > elem->e_anz) // es sind IDs dazu gekommen
    {
        // IDs sortieren, dazu erst mal alle VZ löschen
        dummy = new int[ret[0]];
        for (i = 0; i < ret[0]; ++i)
            dummy[i] = abs(ret[i + 1]);

        dummy = MergeSort(dummy, ret[0]);
        // Lösche doppelte IDs (=innere Kanten) komplett aus dem Feld
        i = 0;
        while (i < edgecount - 1)
        {
            if (dummy[i] == dummy[i + 1])
            {
                // beide löschen
                if (i != edgecount - 2) // letzten beiden Einträge braucht man nicht löschen
                {
                    for (k = i; k < edgecount - 2; ++k)
                    {
                        dummy[k] = dummy[k + 2];
                    }
                }
                edgecount -= 2;
            }
            else
                i++;
        }
        SortEdges(dummy, edgecount, SORT_CLOCKWISE);

        delete ret;
        ret = new int[edgecount + 1];
        memcpy(ret + 1, dummy, sizeof(int) * edgecount);
        ret[0] = edgecount;
    }
    return (ret);
}

// =======================================================
// FindMultipleEdges Version2
// =======================================================
// Finde Punkte, die nur von zwei Kanten verwendet werden
// und die Kanten sollten alle von den selben Flächen
// benutzt werden
int RELDATA::FindMultipleEdges2()
{
    //Generates C4189
    //const char *funcname="RELDATA::FindMultipleEdges2";
    int i, k, upcount, myid, mystartid, *p, newedgeid, size, *tmp;
    ECHAIN *actup, *nodes, *edgeact;
    ECHAIN *edge = NULL, *actedge, *actnode;
    ELEMENT *knoten1, *knoten2, *kante, *face;
    BOOL ende;
    int delnodes = 0, deledges = 0, delfaces = 0, count;
    clock_t start, end;

    enum LINETYPE
    {
        EDGE_BOUNDARY,
        EDGE_NORMAL
    };

    struct NEWJOB
    {
        int exeption;
        LINETYPE typ;
        ELEMENT *face1, *face2;
        int von, nach, nodes;
        ECHAIN *edgelist, *nodelist;
        NEWJOB *next;
    } *jroot = NULL, *jact;

    start = clock();
    // Bei allen Knoten das opt-Flag löschen
    ChangeFlag(ELEM_NODE, FLAG_OPT, FALSE, FALSE);
    for (i = 0; i < anz_eknoten; ++i)
    {
        if (eknoten[i].flags & FLAG_USED)
        {
            upcount = 0;
            actup = eknoten[i].up;
            while (actup != NULL)
            {
                upcount++;
                actup = actup->next;
            }
            if (upcount == 0)
                WriteText(VER_NORMAL, "Der Knoten %d ist nicht freigegeben, wird aber von niemandem benutzt.\n", eknoten[i].id);
            else if (upcount == 2)
            {
                // Neuen Eintrag in die Job-List erstellen
                //        WriteText(VER_DEBUG,"Der Knoten %d koennte optimiert werden (Kante1 %d, Kante2 %d).\n",eknoten[i].id,eknoten[i].up->id,eknoten[i].up->next->id);
                if (jroot == NULL)
                {
                    jroot = new NEWJOB;
                    jact = jroot;
                }
                else
                {
                    jact->next = new NEWJOB;
                    jact = jact->next;
                }
                memset(jact, 0, sizeof(NEWJOB));
                jact->nodes++; // mindestens ein Knoten ist zu löschen
                // Diesen Knoten schon mal in die Liste packen
                jact->nodelist = new ECHAIN;
                jact->nodelist->id = eknoten[i].id;
                jact->nodelist->next = NULL;
                jact->nodelist->element = &eknoten[i];
                actnode = jact->nodelist;

                mystartid = eknoten[i].id;
                myid = mystartid;
                // Jetzt mal die unmittelbaren Nachbarpunkt ermitteln (es sollte zwei geben)
                // Achtung: die routine ist anfällig für Elemente, die von anderen Elementen
                // komplett umhüllt werden oder die alleine im Raum stehen (freie Flächen)
                // Immer mal wieder schauen, ob da nicht ein BUg zu finden ist.
                nodes = GetNeighbours(&eknoten[i], 1);
                // erster Nachbarknoten: "VON"-Seite
                knoten1 = nodes->element;
                // Das hier sollte schon funzen: Der Knoten gehört zu zwei Kanten die beide
                // zu den selben Flächen gehören.
                jact->face1 = eknoten[i].up->element->up->element;
                // Kante am Rand
                if (eknoten[i].up->element->up->next == NULL)
                {
                    jact->face2 = NULL;
                    jact->typ = EDGE_BOUNDARY;
                }
                else
                {
                    jact->face2 = eknoten[i].up->element->up->next->element;
                    jact->typ = EDGE_NORMAL;
                }
                jact->von = knoten1->id;
                // Jetzt die beiden Kanten identifizieren, denen es an den Kragen gehen soll
                actup = eknoten[i].up; // hat genau zwei Einträge
                jact->edgelist = new ECHAIN;
                edgeact = jact->edgelist;

                edgeact->id = actup->id;
                edgeact->element = actup->element;
                edgeact->next = new ECHAIN; // zweites Element gleich anhängen
                edgeact = edgeact->next;

                actup = actup->next;
                edgeact->id = actup->id;
                edgeact->element = actup->element;
                edgeact->next = NULL; // Das war es erstmal

                // zweiter Nachbarknoten "NACH" Seite
                knoten2 = nodes->next->element;
                jact->nach = knoten2->id;
                // Kette löschen ohne Elemente darin zu löschen
                DeleteEChain(nodes, FALSE);
                ende = FALSE;
                while (!ende && jact->exeption == 0)
                {
                    // Kanten des neuen Knoten zählen
                    upcount = 0;
                    actup = knoten1->up;
                    while (actup != NULL)
                    {
                        upcount++;
                        actup = actup->next;
                    }
                    if (upcount != 2) // Hier könnte man noch seltsame Erscheinungen abfangen: <2 z.B.
                    {
                        // ende der Suche
                        jact->von = knoten1->id;
                        ende = TRUE;
                    }
                    else
                    {
                        // eventuell weitere Kanten zum Löschen freigeben und in die ->edge Liste aufnehmen
                        jact->nodes++;
                        // diesen Knoten in die Löschliste eintragen:
                        actnode = new ECHAIN;
                        actnode->id = knoten1->id;
                        actnode->next = NULL;
                        actnode->element = knoten1;

                        nodes = GetNeighbours(knoten1, 1);
                        // Einer dieser Knoten sollte jetzt der alte (myid) Knoten sein
                        if (nodes->element->id == myid)
                        {
                            // neuer "von" Knoten
                            myid = knoten1->id;
                            // zum nächsten Knoten weiter
                            knoten1 = nodes->next->element;
                        }
                        else
                        {
                            // neuer "von" Knoten
                            myid = knoten1->id;
                            // zum nächsten Knoten weiter
                            knoten1 = nodes->element;
                        }
                        // Neue Kante an die Kantenliste von jact anhängen
                        edgeact->next = new ECHAIN;
                        edgeact = edgeact->next;
                        edgeact->next = NULL;
                        edgeact->id = GetEdgeID(myid, knoten1->id);
                        edgeact->element = GetElementPtr(edgeact->id, ELEM_EDGE);

                        // Testen, ob neuer ID eventuell der startid ist, dann hätten wir einen oben
                        // bezeichneten Sonderfall.
                        if (mystartid == myid)
                        {
                            WriteText(VER_NORMAL, "ACHTUNG: geschlossenen Kantenzug mit zwei Nachbarflaechen gefunden (start bei Knoten %d).\n", mystartid);
                            jact->exeption = 1; // Geschlossener Kantenzug mit zwei Nachbarflächen
                        }
                        // Lösche Kette ohne Elemente darin zu löschen
                        DeleteEChain(nodes, FALSE);
                    }
                }
                // Nach-Ast
                ende = FALSE;
                myid = eknoten[i].id;
                while (!ende && jact->exeption == 0)
                {
                    // Kanten des neuen Knoten zählen
                    upcount = 0;
                    //          actup = knoten1->up;
                    actup = knoten2->up; // Blindflugänderung, ist das richtig?
                    while (actup != NULL)
                    {
                        upcount++;
                        actup = actup->next;
                    }
                    if (upcount != 2)
                    {
                        // ende der Suche
                        jact->nach = knoten2->id;
                        ende = TRUE;
                    }
                    else
                    {
                        jact->nodes++;
                        // diesen Knoten in die Löschliste eintragen:
                        actnode = new ECHAIN;
                        actnode->id = knoten1->id;
                        actnode->next = NULL;
                        actnode->element = knoten1;
                        nodes = GetNeighbours(knoten2, 1);
                        // Einer dieser Knoten sollte jetzt der "von" Knoten sein
                        if (nodes->element->id == myid)
                        {
                            // neuer "nach" Knoten
                            //              myid = knoten1->id;
                            myid = knoten2->id;
                            // zum nächsten Knoten weiter
                            knoten2 = nodes->next->element;
                        }
                        else
                        {
                            // neuer "nach" Knoten
                            //              myid = knoten1->id;
                            myid = knoten2->id;
                            // zum nächsten Knoten weiter
                            knoten2 = nodes->element;
                        }
                        // Neue Kante an die Kantenliste von jact anhängen
                        edgeact->next = new ECHAIN;
                        edgeact = edgeact->next;
                        edgeact->next = NULL;
                        edgeact->id = GetEdgeID(myid, knoten2->id);
                        edgeact->element = GetElementPtr(edgeact->id, ELEM_EDGE);

                        // Kette löschen, Elemente darin lassen
                        DeleteEChain(nodes, FALSE);
                    }
                }
            }
            // Alle anderen Punkte (mit mehr Kanten) sind uninteressant
        }
    }

    ChangeFlag(ELEM_FACE, FLAG_OPT, FALSE, FALSE);
    ChangeFlag(ELEM_EDGE, FLAG_OPT, FALSE, FALSE);
    ChangeFlag(ELEM_NODE, FLAG_OPT, FALSE, FALSE);
    // Jetzt die Jobliste durchforsten:
    while (jroot != NULL)
    {
        if (jroot->exeption == 0)
        {
            switch (jroot->typ)
            {
            case EDGE_BOUNDARY:
                WriteText(VER_DEBUG, "Rand erkannt von %d nach %d (%d Knoten), Flaeche %d. Raender werden nicht optimiert\n", jroot->von, jroot->nach, jroot->nodes, jroot->face1->id);
                break;

            case EDGE_NORMAL:
                // Abbrechen, wenn eine der beiden Flächen bereits bearbeitet wurde
                if (jroot->face1->flags & FLAG_OPT || jroot->face2->flags & FLAG_OPT)
                    break;

                if (jroot->face1->e_anz - jroot->nodes - 1 < 3 && jroot->face1->e_anz - jroot->nodes - 1 < 3)
                {
                    WriteText(VER_DEBUG, "Zielflaeche [%d oder %d] hat weniger als drei Ecken! Wird verworfen.\n", jroot->face1->id, jroot->face2->id);
                    break;
                }

                WriteText(VER_DEBUG, "Erstelle neue Kante von %d nach %d zwischen Flaechen %d und %d, loesche %d Knoten\n", jroot->von, jroot->nach, jroot->face1->id, jroot->face2->id, jroot->nodes);
                // Es können mehrere Operationen auf eine Fläche auftreten.
                // dazu gibt es zwei Lösungen:
                // 1) mehrere Durchläufe
                // 2) Operation am existierenden Objekt

                // ich probiere mal 2
                // Änderung zuerst am face1:
                // Kantenliste korrigieren
                // ausschnitt "von" bis "nach" löschen und durch neue (alte) Kante ersetzen
                newedgeid = GetEdgeID(jroot->von, jroot->nach);
                if (newedgeid == 0) // Kante existiert nicht
                {
                    if (edge == NULL)
                    {
                        edge = new ECHAIN;
                        actedge = edge;
                    }
                    else
                    {
                        actedge->next = new ECHAIN;
                        actedge = actedge->next;
                    }
                    actedge->next = NULL;
                    actedge->element = CreateElement(ELEM_EDGE);
                    kante = actedge->element;
                    actedge->id = kante->id;
                    newedgeid = actedge->id;
                    kante->e_anz = 2;
                    kante->element = new PTR_ELEM[2];
                    kante->element[VON].id = jroot->von;
                    kante->element[VON].ptr = NULL;
                    kante->element[NACH].id = -jroot->nach;
                    kante->element[NACH].ptr = NULL;
                }
                /*
                         else  // Das hier kann dann mal weg
                         {
                           // im else-Fall ist eine der Flächen ein Dreieeck und wird gelöscht! (siehe unten)
                           // die Kante muss aber auf alle Fälle behalten werden
                           kante = GetElementPtr(newedgeid,ELEM_EDGE);
                           ChangeFlag(kante,FLAG_USED | FLAG_OPT,TRUE,TRUE); // Punkte mit wiederbeleben
                           WriteText(VER_DEBUG,"Kante %d markiert und gesichert.\n",kante->id);
                         }
               */
                // Kante ist jetzt entweder die neue oder die alte :-)
                // Punktliste nebst Kanten-Indexliste jetzt so lange drehen, bis "von" an erster Stelle steht
                // Kanten stehen in jroot->edges in Form einer ECHAIN-Liste
                for (k = 0; k < 2; ++k)
                {
                    if (k == 0)
                        face = jroot->face1;
                    else
                        face = jroot->face2;

                    if (face != NULL)
                    {
                        // ist das Ergebnis dieser Fläche eventuell kleiner als ein Dreieck?
                        // Dann hüllt die andere Fläche diese fast komplett ein!
                        if (face->e_anz - jroot->nodes < 3)
                        {
                            // Face zum Löschen freigeben
                            if (k == 0)
                                WriteText(VER_NORMAL, "Ein Mehreck[ID:%d,ANZ:%d] umschliesst an %d Kanten eine kleinere Flaeche [ID:%d,ANZ:%d]. Dieses wird geloescht.\n", jroot->face2->id, jroot->face2->e_anz, jroot->nodes + 1, jroot->face1->id, jroot->face1->e_anz);
                            else
                                WriteText(VER_NORMAL, "Ein Mehreck[ID:%d,ANZ:%d] umschliesst an %d Kanten eine kleinere Flaeche [ID:%d,ANZ:%d]. Dieses wird geloescht.\n", jroot->face1->id, jroot->face1->e_anz, jroot->nodes + 1, jroot->face2->id, jroot->face2->e_anz);
                            ChangeFlag(face, FLAG_USED, FALSE, FALSE);
                            delfaces++;
                        }
                        else
                        {
                            size = face->e_anz + jroot->nodes + 2;
                            p = new int[size]; // Alte Kantenliste plus zu löschende Kanten (Knotenzahl+1) plus eine neue Kante
                            // alle alten Kanten
                            for (i = 0; i < face->e_anz; ++i)
                            {
                                p[i] = face->element[i].id;
                            }
                            // jetzt die zu löschende Kanten einfügen (sind jetzt doppelt)
                            actup = jroot->edgelist;
                            while (actup != NULL) // zu löschende Kanten
                            {
                                p[i++] = actup->id;
                                actup = actup->next;
                            }
                            // Käse sortieren
                            // Leider muss man sich jetzt irgendwie die Reihenfolge merken!
                            // ignoriere VZ
                            tmp = MergeSort(p, size - 1, TRUE);
                            delete p;
                            // doppelte Kanten löschen, hier klappt das direkt
                            // dazu erst mal den Zähler wieder eins runter (wegen nachträglicher Änderung von KillDoubleIDs)
                            size--;
                            tmp = KillDoubleIDs(tmp, &size);
                            // Reihenfolge in der ID-Liste wieder herstellen
                            // neue Kante hinten einfügen
                            p = new int[size + 1];
                            memcpy(p, tmp, sizeof(int) * size);
                            size++; // Zähler wieder hoch
                            p[size - 1] = newedgeid; // Reihenfolge also beibehalten!
                            // Die Methode Sortedge sucht sowohl in den Stammdaten als auch
                            // in der angegebenen ECHAIN-Kette nach dem ID
                            SortEdges(p, size, SORT_CLOCKWISE, edge);
                            // jetzt das Element modifizieren
                            delete face->element;
                            //                face->e_anz -=jroot->nodes;
                            face->e_anz = size;
                            if (face->e_anz < 3)
                                printf("Mist produziert: Flaeche %d hat nur noch %d Ecken.\n", face->id, face->e_anz);
                            face->element = new PTR_ELEM[face->e_anz];
                            for (i = 0; i < face->e_anz; ++i)
                            {
                                face->element[i].id = p[i];
                                face->element[i].ptr = NULL;
                            }
                            delete p;
                        }
                        // Markieren
                        face->flags |= FLAG_OPT;
                    }
                }

                // Nicht mehr benötigte Kanten zum Löschen freigeben, ebenso die Knoten
                size = (jroot->nodes + 1) * 2; // +1 wegen Rand (Kanten = Knoten+1), *2 wegen 2 Punkte pro Kante
                p = new int[size];
                count = 0;
                actup = jroot->edgelist;
                while (actup != NULL)
                {
                    // Knoten in einen Topf werfen
                    p[count++] = abs(actup->element->element[VON].id);
                    p[count++] = abs(actup->element->element[NACH].id);
                    actup = actup->next;
                }
                // Sortieren
                p = MergeSort(p, size);
                // Doppelte Knoten Freigeben
                for (i = 0; i < size - 1; ++i)
                {
                    if (p[i] == p[i + 1])
                    {
                        knoten1 = GetElementPtr(p[i], ELEM_NODE);
                        if (!(knoten1->flags & FLAG_OPT))
                        {
                            ChangeFlag(knoten1, FLAG_USED, FALSE, FALSE);
                            WriteText(VER_DEBUG, "Knoten %d freigegeben.\n", knoten1->id);
                            delnodes++;
                        }
                    }
                }
                delete p;

                // Jetzt Kanten freigeben
                actup = jroot->edgelist;
                while (actup != NULL)
                {
                    if (!(actup->element->flags & FLAG_OPT))
                    {
                        // Als optimiert markirete Kanten sind for dem Löschen geschützt
                        deledges++;
                        ChangeFlag(actup->element, FLAG_USED, FALSE, FALSE);
                    }
                    else
                    {
                        // eventuell wieder in Betrieb nehmen
                        actup->element->flags |= FLAG_USED;
                        WriteText(VER_DEBUG, "Die zum Loeschen bestimmte Kante %d wird noch benoetigt und wiederbelebt.\n", actup->id);
                    }
                    actup = actup->next;
                }
                break;
            }
        } // ende if exeption
        else
            WriteText(VER_DEBUG, "Ausnahme %d zwischen den Flaechen %d und %d. Jobauftrag wird ignoriert.\n", jroot->exeption, jroot->face1->id, jroot->face2->id);

        // Kantenliste Killen, wurde oben erstellt
        DeleteEChain(jroot->edgelist, FALSE);
        DeleteEChain(jroot->nodelist, FALSE);
        // Job erledigt (so odelist so!)
        jact = jroot->next;
        delete jroot;
        jroot = jact;
    }

    // neue Elemente Mergen
    Disconnect(ELEM_ALL);
    DeleteUnused(ELEM_ALL);
    Merge(edge, ELEM_EDGE);
    Reconnect(ELEM_ALL);
    end = clock();
    WriteText(VER_MAX, "Kantenfindung beendet [%8.4f sec]. %d Flaechen, %d Kanten und %d Knoten freigegeben\n", (double)(end - start) / CLOCKS_PER_SEC, delfaces, deledges, delnodes);
    return (delnodes + deledges);
}

/*

// =======================================================
// GetArea
// =======================================================
// Liefert eine Flächenliste, deren Normalen sich von dem
// der Referenzebene (startface) maximal um den Winkel arc
// unterscheiden. Alle Elemente sind nachbarn von actface.
// Die Liste kann doppelte Element enthalten (durch den rekursiven Aufruf)
// Die doppelten Elemente werden am Ende aber gelöscht.
// Die Methode liefert nur dann das actface Element zurück, wenn
// es mindestens einen passenden Nachbarn gibt.
// Ansonsten (zu dem angegebenen Element gibt es keine
// passenden Nachbarn) liefert die Methode NULL
ECHAIN *RELDATA::GetArea(ELEMENT* startface, ELEMENT* actface, double arc, int range)
{
const char *funcname="RELDATA::GetArea";
ECHAIN *root=NULL,*act,*addlist;
ECHAIN *tmp,*nachbar;
double newarc;

// Abbruchkriterium
if (range==0)
return(NULL);

// Schritt eins: Nachbarn dieser Fläche besorgen, Radius 1
nachbar = GetNeighbours(actface,1);
while(nachbar!=NULL)
{
if (!(nachbar->element->flags & FLAG_AREA) && nachbar->element->flags & FLAG_USED && actface->id != nachbar->id)
{
// Winkel zwischen den Flächen berechnen
newarc = fabs(GetFaceAngle(startface,nachbar->element));
//      WriteText(VER_DEBUG,"Der Winkel zwischen den Flaechen %d und %d beträgt %8.4f Grad.\n",startface->id,nachbar->id,newarc);
if (newarc<=arc)
{
// Diese Fläche passt! ACHTUNG dieses Flag kann später eventuell zurück genommen werden
nachbar->element->flags |= FLAG_AREA;
// Jetzt in die Jobliste eintragen
if (root==NULL)
{
root = new ECHAIN;
act  = root;
// Dazu dann gleich noch die Quellfläche eintragen
act->id = actface->id;
act->element = actface;
actface->flags |= FLAG_AREA;
// Jetzt das neue Element einfügen
act->next=new ECHAIN;
act=act->next;
}
else
{
act->next = new ECHAIN;
act=act->next;
}
memcpy(act,nachbar,sizeof(ECHAIN));
act->next=NULL;
// Testen, ob diese Fläche wiederum günstige Nachbarn hat
addlist = GetArea(startface,nachbar->element,arc,range-1);
// Addlist kann NULL sein, wenn alle folgenden Elemente nicht passen
if (addlist!=NULL)
{
act->next=addlist;
while(act->next!=NULL)
act=act->next;
}
}
}
tmp =nachbar->next;
delete nachbar;
nachbar=tmp;
}
// Jetzt die Jobliste von doppelten Elementen säubern
if (root!=NULL)
{
root=SortEChain(root);
act = root;
while (act->next!=NULL)
{
if (act->next->id == act->id)
{
tmp =act->next;
act->next= act->next->next;
delete tmp;
}
else
act=act->next;
}
}
// und raus
return(root);
}

// =======================================================
// CoarseFace
// =======================================================
// Zweiter Anlauf
int RELDATA::CoarseFaces()
{
const char *funcname="RELDATA::CoarseFaces";
int i,j,k,*ret,*tmp,*p;
int *einzeln, *doppelt,ecount,dcount,*nodelist,nodecount;
int *oldedges,numedges,anz;
ELEMENT *elem;
ECHAIN *root,*act; // Liste der beteiligten Flächen an einer neuen Fläche
ECHAIN *actup,*uptemp;
ECHAIN *newelem=NULL,*actelem; // Liste der neu erschaffenen Flächen
int newareas=0,delnodes=0,deledges=0;
BOOL found;
clock_t start,end;

start = clock();
// Area-Flag bei allen löschen
ChangeFlag(ELEM_FACE,FLAG_AREA,FALSE,FALSE);
for (i=0;i<anz_eflaechen;++i)
{
if (eflaeche[i].flags & FLAG_USED && !(eflaeche[i].flags & FLAG_AREA))
{
// Flächensuche mit folgenden Parametern:
// Die zusammengefasste neue Fläche muss aus noch nicht optimierten
// Elementen bestehen. Die Normalen aller an einem Gebiet beteiligten Fläche
// dürfen von der Normalen der Startfläche nur um einen bestimmten Winkel abweichen
// Die Methode markiert die hier gelieferten Flaechen nicht automatisch als benutzt
// (siehe Sonderfälle unten)
root = GetArea(&eflaeche[i],&eflaeche[i],Globals.optarc,Globals.optsteps+1);
// Nochmal Nachdenken: was gibt es denn noch für Sonderfälle für das gefundene Gebiet?
if (root!=NULL)
{
act = root;
WriteText(VER_DEBUG,"Es wurde ein Gebiet mit folgenden Flaechen gefunden: ");
while(act!=NULL)
{
WriteText(VER_DEBUG," %d ",act->id);
act=act->next;
}
WriteText(VER_DEBUG,"Quellflaeche: %d \n",eflaeche[i].id);
// wichtig jetzt: Alle nicht mehr benötigten Punkte und Kanten
// sollten sauber gelöscht werden um später Brachialmethoden zu vermeiden
// Für Flächen ist das Einfach: es werden alle Flächen mit dem
// AREA-Flag gelöscht, bei Kanten und Knoten kann das schon kniffliger werden
numedges=0;
oldedges=NULL;
act=root;
while(act!=NULL)
{
// REMEMBER: Get Nodes liefert bei Flächen zusätzlich noch die Kanten mit (alles sauber geordnet)
ret = GetNodes(act->element);
// Neue Knoten zu alten Knoten addieren, neue Kanten zu alten
anz = act->element->e_anz;
if (oldedges==NULL)
{
oldedges = new int[anz];
memcpy(oldedges,ret+anz,sizeof(int)*anz);
delete ret;
}
else
{
tmp = oldedges;
oldedges = new int[anz+numedges];
memcpy(oldedges,tmp,sizeof(int)*numedges);
memcpy(oldedges+numedges,ret+anz,sizeof(int)*anz);
delete tmp;
}
numedges+=anz;
act=act->next;
}

// Liste aller Kanten (manche doppelt oder mehrfach) jetzt in
// oldedges, Anzahl anzedges

// zu löschende Kanten erkennen:
// Liste sortieren, doppelte Elemente sind nicht mehr benötigte Kanten
// die doppelten Elemente liegen dabei in der Form -id,id

// MergeSort von Integern, ignoriere VZ
tmp = MergeSort(oldedges, numedges,TRUE);
delete oldedges;
oldedges=tmp;

// Doppelte Kanten jetzt von Einzelkanten isolieren
einzeln = new int[numedges];
doppelt = new int[numedges];
ecount=0;
dcount=0;
for (j=0;j<numedges-1;++j)
{
if (oldedges[j]==-oldedges[j+1])
{
doppelt[dcount++]=abs(oldedges[j]);
j++;  // das doppelte Element überspringen
}
else if (oldedges[j]==oldedges[j+1])
{
WriteText(VER_DEBUG,"%s meldet: Fehler in Kantenstruktur bei Kante %d an Flaeche %d\n",funcname,oldedges[j],eflaeche[i].id);
}
else
{
einzeln[ecount++]=oldedges[j];
}
}
// Letztes Element Sonderabfrage: (wegen zu kleinem Zähler!)
if (oldedges[numedges-2]!=-oldedges[numedges-1])
einzeln[ecount++] = oldedges[numedges-1];

// Sonderfall: Das Gebiet schliesst eine Fläche vollständig ein (bis auf einen Knoten)
// innere Fläche herausfinden und löschen
// Knotenliste besorgen
found=FALSE;
nodelist = new int[ecount*2];
nodecount=0;
for (j=0;j<ecount;++j)
{
p = GetNodes(GetElementPtr(einzeln[j],ELEM_EDGE));
nodelist[nodecount++]=p[0];
nodelist[nodecount++]=p[1];
delete p;
}
// Jetzt doppelte Knoten löschen
tmp = MergeSort(nodelist,nodecount);
delete nodelist;
nodelist = tmp;
// Jetzt sollten die doppelten Punkte viermal in der Liste auftauchen
for (j=0;j<nodecount-2;++j)
{
if (nodelist[j]==nodelist[j+2])
{
// Habe einen Knoten gefunden, erst mal nur ausgeben
WriteText(VER_DEBUG,"Doppelter Knoten %d\n",nodelist[j]);
found=TRUE; // Diese Fläche nicht benutzen
j+=2;
}
}
delete nodelist;

// Jetzt "einzeln"e Kanten zu einer neuen Fläche ordnen
// Bei einem Fehler ist die erzeugte Fläche eventuell von einer
// weiteren Fläche unterbrochen (diese hat dann nur die Startfläche als Nachbarn)
//        if (SortEdges(einzeln,ecount,SORT_CLOCKWISE)==0)  // kein Fehler
if (!found && SortEdges(einzeln,ecount,SORT_CLOCKWISE)==0)  // kein Fehler
{
// fertig, doppelte Kanten jetzt löschen
for (j=0;j<dcount;++j)
{
elem = GetElementPtr(doppelt[j],ELEM_EDGE);
ChangeFlag(elem,FLAG_USED,FALSE,FALSE);
WriteText(VER_DEBUG,"Kante %d freigegeben.\n",elem->id);
deledges++;
for (k=0;k<2;++k)
{
// Jetzt diese Kante aus der UP-Liste des Knotens löschen
actup = elem->element[k].ptr->up;  // Up-Liste des "Von" Knotens
while(actup!=NULL)
{
if (actup->id == elem->id)  // Gefunden
{
// diesen Eintrag löschen
if (actup==elem->element[k].ptr->up)
{
elem->element[k].ptr->up=elem->element[k].ptr->up->next;
// wenn jetzt der UP-Pointer NULL ist, kann man diesen Knoten löschen
if (elem->element[k].ptr->up==NULL)
{
WriteText(VER_DEBUG,"Knoten %d freigegeben.\n",elem->element[k].ptr->id);
ChangeFlag(elem->element[k].ptr,FLAG_USED,FALSE,FALSE);
delnodes++;
}
}
else
{
uptemp = elem->element[k].ptr->up;
while(uptemp->next!=actup)
uptemp=uptemp->next;
uptemp->next = actup->next;
}
delete actup;
break;
}
actup=actup->next;
} // while actup
} // for k
}
// Neues Element erstellen
if (newelem==NULL)
{
newelem = new ECHAIN;
actelem = newelem;
}
else
{
actelem->next = new ECHAIN;
actelem=actelem->next;
}
actelem->next=NULL;
actelem->element = CreateElement(ELEM_FACE);  // ID, Typ und USED schon gesetzt
elem = actelem->element;
actelem->id = elem->id;
elem->e_anz = ecount;
elem->element=new PTR_ELEM[ecount];
// Element jetzt füllen
for (k=0;k<ecount;++k)
{
elem->element[k].id  = einzeln[k];
elem->element[k].ptr = NULL;
}
newareas++;
}
else  // Sortedge liefert einen Fehler
{
// Verwendete Elemente als unbenutzt markieren
act=root;
while(act!=NULL)
{
ChangeFlag(act->element,FLAG_AREA,FALSE,FALSE);
act=act->next;
}
WriteText(VER_DEBUG,"Die erstellte Flaeche wird verworfen.\n");
}
// Root Kette löschen
DeleteEChain(root,FALSE);
// Alle Integer-Array löschen
delete oldedges;
delete einzeln;
delete doppelt;
// Das wars
}
}
}
// Alle "AREA"-Elemente jetzt löschen
for (i=0;i<anz_eflaechen;++i)
{
if (eflaeche[i].flags & FLAG_AREA)
ChangeFlag(&eflaeche[i],FLAG_USED,FALSE,FALSE);
}
// Erstellte Elemente jetzt mergen, überflüssige löschen
Disconnect(ELEM_ALL);
DeleteUnused(ELEM_ALL);         // Alle unbenutzten Knoten, Kanten und Flaechen jetzt löschen
Merge(newelem,ELEM_FACE);       // neue Elemente dazu
Reconnect(ELEM_ALL);            // wieder verknüpfen
end=clock();
WriteText(VER_MAX,"Vergroeberung des Gitters erzeugte %d neue Flaechen, geloescht wurden %d Knoten und %d Kanten\n",newareas,delnodes,deledges);
return(0);
}
*/

/*
// =======================================================
// CoarseFace
// =======================================================
int RELDATA::CoarseFaces()
{
  const char *funcname="RELDATA::CoarseFaces";
  int i,j,*ret;
  int areacount=0,maxid;
  ECHAIN *root=NULL,*eact;
  clock_t start,end;

start=clock();
maxid = GetMaxID(ELEM_FACE)+1;

WriteText(VER_NORMAL,"Vergroebere Oberflaechengitter...\n");
ChangeFlag(ELEM_FACE,FLAG_OPT,FALSE,FALSE);
for (i=0;i<anz_eflaechen;++i)
{
ret=NULL;
if (eflaeche[i].flags & FLAG_USED)
{
ret = CreateArea(&eflaeche[i],0,Globals.maxlocalarc,Globals.maxglobalarc);
if (ret!=NULL)
{
// Wurde die Fläche verändert? Es kann sein, dass die Kantenzahl trotzdem
// gleich geblieben ist!
if (eflaeche[i].flags & FLAG_OPT)
{
areacount++;
WriteText(VER_DEBUG,"Flaeche %d zusammengefasst zu %d Kanten\n",eflaeche[i].id,ret[0]);
// Eine Kette erstellen
if (root==NULL)
{
root=new ECHAIN;
eact=root;
}
else
{
eact->next = new ECHAIN;
eact=eact->next;
}
eact->next=NULL;
eact->id = maxid++;
eact->element= new ELEMENT;
memset(eact->element,0,sizeof(ELEMENT));
eact->element->id = eact->id;
eact->element->typ = ELEM_FACE;
eact->element->flags |= (FLAG_USED | FLAG_DISCON);
eact->element->e_anz = ret[0];
eact->element->element=new PTR_ELEM[ret[0]];
for (j=1;j<ret[0]+1;++j)
{
WriteText(VER_DEBUG,"%6d ",ret[j]);
eact->element->element[j-1].id = ret[j];
eact->element->element[j-1].ptr=NULL; // wird später connected
}
WriteText(VER_DEBUG,"\n");
}
else
ChangeFlag(&eflaeche[i],FLAG_AREA,FALSE,FALSE);
delete ret;
}
// Fixme:
if (eflaeche[i].flags & FLAG_AREA)
ChangeFlag(&eflaeche[i],FLAG_USED,FALSE,FALSE);
}
}
Disconnect(ELEM_ALL);
Merge(root,ELEM_FACE);
DeleteUnused(ELEM_FACE);
Reconnect(ELEM_ALL);
//  ChangeFlag(ELEM_ALL,FLAG_USED,FALSE,FALSE); // Alle Elemente werden als unbenutzt markiert
//  ChangeFlag(ELEM_FACE,FLAG_USED,TRUE,TRUE);  // Alle Elemente, die was mit aktuellen Flächen haben, werden mariert
//  Disconnect(ELEM_ALL);
//  DeleteUnused(ELEM_ALL); // Alle anderen werden gelöscht
//  Reconnect(ELEM_ALL);
end=clock();
WriteText(VER_NORMAL,"Es wurden %d Gebiete gefunden [%8.4f sec].\n",areacount,(double)(end-start)/CLOCKS_PER_SEC);
return(0);
}
*/

// =======================================================
// Errorscan
// =======================================================
int RELDATA::Errorscan()
{
    //Generates C4189
    //const char *funcname="RELDATA::Errorscan";
    int i, upcount, errcount = 0;
    ECHAIN *act;

    WriteText(VER_MAX, "Scanne Kanten nach mehrfachen Verbindungen...\n");
    for (i = 0; i < anz_ekanten; ++i)
    {
        upcount = 0;
        if (ekante[i].up == NULL)
        {
            errcount++;
            WriteText(VER_DEBUG, "Die Kante %d ist mit keiner Flaeche verbunden!\n", ekante[i].id);
            ChangeFlag(&ekante[i], FLAG_USED, FALSE);
        }
        act = ekante[i].up;
        while (act != NULL)
        {
            upcount++;
            act = act->next;
        }
        if (upcount > 2)
        {
            errcount++;
            WriteText(VER_DEBUG, "Die Kante %d ist mit %d Flaechen verbunden!\n", ekante[i].id, upcount);
        }
    }
    WriteText(VER_MAX, "Fertig mit Scannen. %d falsche Kanten gefunden\n", errcount);

    return (0);
}

/*
// =======================================================
// MarkusedElements
// =======================================================
int RELDATA::MarkUsedElements()
{
  const char *funcname="RELDATA::MarkUsedElements";
  int i,j,k;
  int faces=0,nodes=0,edges=0;
  ELEMENT *kante,*knoten;
  clock_t start,end;

// Alle Flags von Hand auf Null
start =clock();
WriteText(VER_MAX,"Markiere alle benutzten Elemente...\n");
for (i=0;i<anz_eknoten;++i)
eknoten[i].flags=0;

for (i=0;i<anz_ekanten;++i)
ekante[i].flags=0;
// So, jetzt von den Elementen zu den Knoten gehen
// Annahme: alle Existenten Flächen werden auch benutzt

for (i=0;i<anz_eflaechen;++i)
{
eflaeche[i].flags = FLAG_USED;  // alle anderen werden gelöscht
faces++;
for (k=0;k<eflaeche[i].e_anz;++k)
{
kante = GetElementPtr(eflaeche[i].element[k].id,ELEM_EDGE);
eflaeche[i].element[k].ptr=kante;
edges++;
kante->flags = FLAG_USED;
// Jetzt die Kanten durchlaufen
for (j=0;j<kante->e_anz;++j)
{
knoten = GetElementPtr(kante->element[j].id,ELEM_NODE);
kante->element[j].ptr=knoten;
nodes++;
knoten->flags = FLAG_USED;
}
}
}
WriteText(VER_MAX,"%d Flaechen, %d Kanten und %d Knoten als benutzt markiert.\n",faces,edges,nodes);
// Unnötiges Kruppzeuch löschen
Disconnect(ELEM_ALL);
DeleteUnused(ELEM_ALL);
Reconnect(ELEM_ALL);
end = clock();
WriteText(VER_MAX,"Markierung und Loeschung fertig [%8.4f sec]\n",(double)(end-start)/CLOCKS_PER_SEC);
return(0);
}

// =======================================================
// Extrude
// =======================================================
// Erstellt ein Volumenmodell aus einem Schalenmodell
// Version 1: Dicke ist überall gleich und wird als Parameter übergeben
int RELDATA::Extrude(int slices, double thickness,BOOL killfaces)
{
const char *funcname="RELDATA::Extrude";
ECHAIN *node=NULL,*actnode;
ECHAIN *edge=NULL,*actedge;
ECHAIN *face=NULL,*actface;
ECHAIN *cell=NULL,*actcell;
ECHAIN *ret=NULL,*tmp;
ELEMENT *knoten;

int nodecount=0,edgecount=0,facecount=0,cellcount=0;

double dist,vec[3];
int i,j,k,*p,*np,offset;
int divisor = MAX(anz_eflaechen/534,50);  // ab 26700 Flächen wächst der divisor!
clock_t start,end;

// Erst mal die Normalen an jedem Punkt erstellen
CalculatePhongNormals();
PrintStatement();

// jetzt durch die Punkte laufen und Volumen erstellen
// Schritt eins: Kanten (und Knoten) in Normalenrichtung erstellen
setvbuf(stdout,NULL,_IONBF,6);
dist = thickness/slices;
start=clock();
for (i=0;i<anz_eflaechen;++i)
{
if (eflaeche[i].flags & FLAG_USED)
{
// Zähler
if (i%divisor==0)
printf("%6.2f%%\r",(double)(100*i)/anz_eflaechen);
p = GetNodes(&eflaeche[i]);
offset = eflaeche[i].e_anz;
np = new int[offset*(slices+1)];
memcpy(np,p,sizeof(int)*offset); // in np stehen die Punkte für alle neuen Zellen

// Scheiben erstellen
for (k=0;k<slices;++k)
{
// immer in Scheiben organisiert
for (j=0;j<offset;++j)
{
knoten = GetElementPtr(p[j],ELEM_NODE); // immer vom Grundpunkt aus!
if (knoten==NULL)
{
WriteText(VER_NORMAL,"%s meldet: Der Knoten ID %d konnte nicht gefunden werden. Abbruch\n",funcname,np[j+k*offset]);
// Nodes Kette komplett löschen
DeleteEChain(node,TRUE);
DeleteEChain(edge,TRUE);
DeleteEChain(face,TRUE);
DeleteEChain(cell,TRUE);
delete np;
delete p;
return(1);
}
// Normalenvektor besorgen (ist normiert!)
vec[X]=knoten->data[4].value; // X-Koordinate des Normalenvektors FIXME: ist nicht dynamisch
vec[Y]=knoten->data[5].value; // Y-Koordinate des Normalenvektors FIXME: ist nicht dynamisch
vec[Z]=knoten->data[6].value; // Z-Koordinate des Normalenvektors FIXME: ist nicht dynamisch

if (node==NULL)
{
node = new ECHAIN;
actnode=node;
}
else
{
actnode->next= new ECHAIN;
actnode=actnode->next;
}

actnode->next=NULL;
actnode->element = CreateElement(ELEM_NODE,TRUE); // Erstellt einen Knoten aus dem Prototypen (koten[0])
nodecount++;
actnode->id = actnode->element->id;
// Jetzt die neuen Koordinaten errechnen
actnode->element->data[X].value = knoten->data[X].value+(k+1)*dist*vec[X];
actnode->element->data[Y].value = knoten->data[Y].value+(k+1)*dist*vec[Y];
actnode->element->data[Z].value = knoten->data[Z].value+(k+1)*dist*vec[Z];
//..noch genauer brauchen wirs nicht!
// das wars erst mal
// noch den ID des neuen Knoten merken
np[j+(k+1)*offset]= actnode->id;
}
}
// Knoten Sind erstellt: jetzt Zelle draus machen
// Der Aufruf erwartet die beiden Knotenscheiben in der form: unten, oben, Liste von neuen Punkten
delete p;
for (k=0;k<slices;++k)
{
// bei k=0 ist eine Fläche vorhanden, sonst wurde sie in dieser Routine neu erstellt
ret = CreateCell(&np[k*offset],&np[(k+1)*offset],offset);
// Einordnen:
while(ret!=NULL)
{
// einzuordnendes Element aus der Kette trennen
tmp = ret;
ret=ret->next;
tmp->next=NULL;

switch(tmp->element->typ)
{
case ELEM_EDGE:
if (edge==NULL)
{
edge=tmp;
actedge=edge;
}
else
{
actedge->next = tmp;
actedge=actedge->next;
}
edgecount++;
break;

case ELEM_FACE:
if (face==NULL)
{
face=tmp;
actface=face;
}
else
{
actface->next = tmp;
actface=actface->next;
}
facecount++;
break;

case ELEM_CELL:
if (cell==NULL)
{
cell=tmp;
actcell=cell;
}
else
{
actcell->next = tmp;
actcell=actcell->next;
}
cellcount++;
break;

default:
WriteText(VER_NORMAL,"%s meldet: CreateCell liefert unbekannten Elementtyp %d\n",funcname,ret->element->typ);
// Dieses Element wird gelöscht
DeleteEChain(tmp,TRUE);
break;
}
}
}
delete np;
}
}
setvbuf(stdout,NULL,_IONBF,BUFSIZ);

// Jetzt die gesamten alten Flächen löschen
ChangeFlag(ELEM_FACE,FLAG_USED,FALSE,FALSE);

Disconnect(ELEM_ALL);
DeleteUnused(ELEM_ALL);
Merge(node,ELEM_NODE);
Merge(edge,ELEM_EDGE);
Merge(face,ELEM_FACE);
if (!killfaces) // Nur mergen, wenn nicht nur die Oberfläche erzeugt werden soll
Merge(cell,ELEM_CELL);
else
DeleteEChain(cell,TRUE);
Reconnect(ELEM_ALL);

DeleteDoubleElement(ELEM_NODE);
DeleteDoubleElement(ELEM_EDGE);
if (!killfaces)
{
DeleteDoubleElement(ELEM_FACE);
}
else
{
KillDoubleElement(ELEM_FACE);
}

// Jetzt noch den Elementtyp anpassen
Globals.femtyp=45;    // Solid45

end=clock();
WriteText(VER_NORMAL,"%s meldet: %d Knoten, %d Kanten, %d Flaechen und %d Zellen erstellt [%8.4f sec].\n",funcname,nodecount,edgecount,facecount,cellcount,(double)(end-start)/CLOCKS_PER_SEC);

return(0);
}

// =======================================================
// CreateCell
// =======================================================
// Die erstellten Kanten, Flächen und Zellen werden als EINE
// Kette zurückgeliefert! Bitte noch auseinanderpfriemeln
// Man könnte sie natürlich auch gleich an die gelieferten
// Ketten anhängen, ist aber nicht so durchsichtig.
ECHAIN *RELDATA::CreateCell(int *unten, int *oben, int size)
{
const char *funcname="RELDATA::CreateCell";
ECHAIN *edge=NULL,*actedge=NULL;
ECHAIN *face=NULL,*actface=NULL;
ECHAIN *ret=NULL;
ELEMENT *elem;
int *faceid,*edgeid;
int i,k,actsize;
int *facenode;
int von, nach;

clock_t start,end;

faceid = new int[size+2];   // eine Fläche pro Kante + oben + unten

for (k=0;k<size+2;++k)
{
switch(k)
{
case 0: // Bodenfläche erstellen
actsize=size;
facenode = new int[actsize];
memcpy(facenode,unten,sizeof(int)*actsize);
break;

case 1: // Dachfläche erstellen
actsize=size;
facenode = new int[actsize];
// Drehsinn umdrehen, damit Normale erst mal nach aussen zeigt!
for (i=0;i<actsize;++i)
facenode[actsize-i-1]=oben[i];
break;

default: // Seitenflächen
actsize=4;
facenode = new int[actsize];
if (k!=size+1)
{
facenode[3]=unten[k-2];
facenode[2]=unten[k-1];
facenode[1]=oben[k-1];
facenode[0]=oben[k-2];
}
else
{
facenode[3]=unten[k-2];
facenode[2]=unten[0];
facenode[1]=oben[0];
facenode[0]=oben[k-2];
}
break;
}
// Gibt es schon eine Fläche mit den gewünschten Punkten?
start = clock();
//    faceid[k] = GetFaceID(facenode,actsize,neuflaechen,neukanten);
faceid[k]=0;  // FIXME
end = clock();
//    facetime+=end-start;
if (faceid[k]==0)
{
// Liste mit KantenIDs erstellen
edgeid = new int[actsize];
// untere Kanten erstellen, die Punkte liegen bereits in richtiger Reihenfolge!
for (i=0;i<actsize;++i)
{
von = facenode[i];
if (i!=actsize-1)
nach = facenode[i+1];
else
nach = facenode[0];

// Testen, ob die Kante schon existiert...
start=clock();
//        edgeid[i]=GetEdgeID(von,nach,neukanten);
edgeid[i]=0;
end=clock();
//        edgetime+=end-start;
if (edgeid[i]==0) // Nix ist, ID existiert nicht
{
if (edge==NULL)
{
edge = new ECHAIN;
actedge=edge;
}
else
{
actedge->next = new ECHAIN;
actedge=actedge->next;
}
actedge->next=NULL;
actedge->element = CreateElement(ELEM_EDGE,TRUE);
actedge->id = actedge->element->id;
edgeid[i]=actedge->id;
elem = actedge->element;
// Kantenwerte setzen
elem->e_anz = 2;
elem->element = new PTR_ELEM[2];
elem->element[VON].id  = von;
elem->element[VON].ptr = NULL;
elem->element[NACH].id = -nach;
elem->element[NACH].ptr =NULL;
}
}

// Jetzt die Fläche erstellen
if (face==NULL)
{
face = new ECHAIN;
actface=face;
}
else
{
actface->next= new ECHAIN;
actface = actface->next;
}
actface->next=NULL;
actface->element = CreateElement(ELEM_FACE);
elem = actface->element;
actface->id = elem->id;
faceid[k]=elem->id;
elem->e_anz = actsize;
elem->element= new PTR_ELEM[actsize];
for (i=0;i<actsize;++i)
{
elem->element[i].id = edgeid[i];
elem->element[i].ptr=NULL;
}
delete edgeid;
} // Ende: Fläche existiert nicht
delete facenode;
}
// Jetzt Zelle erstellen
ret = new ECHAIN;
ret->next=NULL;
ret->element = CreateElement(ELEM_CELL,TRUE);
ret->id = ret->element->id;
ret->element->e_anz = size+2;  // 2 Grundflächen + eine pro Kante
ret->element->element = new PTR_ELEM[size+2];
for (k=0;k<size+2;++k)
{
ret->element->element[k].id = faceid[k];
ret->element->element[k].ptr=NULL;
}

delete faceid;
// jetzt noch die Ketten von Kanten und Flächen zu der Ret-Kette addieren
ret->next = edge;
actedge->next = face;

return(ret);
}

*/
// =======================================================
// ReIndex
// =======================================================
// Diese Methode erzeugt für die angegebene Elementart
// neue Indices.
// Was ist zu tun?
// Genau andersrum wie Disconnect arbeiten: Die Ptr-Elemente
// stehen lassen, die id Elemente löschen und dann durch die
// Struktur gehen und die IDs durch die PTRs erfragen.
int RELDATA::ReIndex(ELEMENTTYPE etype)
{
    const char *funcname = "RELDATA::ReIndex";
    int i, k, anz_elem, anz_super;
    ELEMENT *acttab, *supertab, *elem;
    ECHAIN *actup;

    switch (etype)
    {
    case ELEM_NODE:
        anz_elem = anz_eknoten;
        anz_super = anz_ekanten;
        acttab = eknoten;
        supertab = ekante;
        break;

    case ELEM_EDGE:
        anz_elem = anz_ekanten;
        anz_super = anz_eflaechen;
        acttab = ekante;
        supertab = eflaeche;
        break;

    case ELEM_FACE:
        anz_elem = anz_eflaechen;
        anz_super = anz_ezellen;
        acttab = eflaeche;
        supertab = ezelle;
        break;

    case ELEM_CELL:
        anz_elem = anz_ezellen;
        anz_super = 0;
        acttab = ezelle;
        supertab = NULL;
        break;

    case ELEM_ALL:
        ReIndex(ELEM_NODE);
        ReIndex(ELEM_EDGE);
        ReIndex(ELEM_FACE);
        ReIndex(ELEM_CELL);
        return (0);

    default:
        WriteText(VER_NORMAL, "%s meldet: unbekannter Elementtyp  %d\n", funcname, etype);
        return (0);
    }

    // Abfrage Randbedingungen
    if (anz_elem <= 0)
        return (0);

    // Baum muss connected sein!

    if (GetFlagUsage(FLAG_DISCON, etype) != 0)
    {
        WriteText(VER_NORMAL, "%s meldet: Es sind nicht alle Elemente des Typs %d connected\n", funcname, etype);
        return (1);
    }

    // und los
    for (i = 0; i < anz_elem; ++i)
    {
        acttab[i].id = i + 1;
        // Alle uplinks löschen
        DeleteEChain(acttab[i].up, FALSE);
        acttab[i].up = NULL;
    }

    if (supertab != NULL)
    {
        for (i = 0; i < anz_super; ++i)
        {
            for (k = 0; k < supertab[i].e_anz; ++k)
            {
                elem = supertab[i].element[k].ptr;
                supertab[i].element[k].id = SIGN(supertab[i].element[k].id) * elem->id;
                // Jetzt sind noch die UP-Links falsch
                if (elem->up == NULL)
                {
                    elem->up = new ECHAIN;
                    actup = elem->up;
                }
                else
                {
                    actup = elem->up;
                    while (actup->next != NULL)
                    {
                        if (actup->id == supertab[i].id)
                        {
                            // Wenn man hier raus sprigt, ist alles futsch, einfach erst mal nur melden
                            WriteText(VER_DEBUG, "%s meldet: eigener ID %d (Typ %d) bereits in UP-Links des Elements %d (Typ %d) enthalten!\n", funcname, supertab[i].id, supertab[i].typ, elem->id, elem->typ);
                        }
                        actup = actup->next;
                    }
                    actup->next = new ECHAIN;
                    actup = actup->next;
                }
                actup->next = NULL;
                actup->id = supertab[i].id;
                actup->element = &supertab[i];
            }
        }
    }

    // Zähler neu einstellen
    switch (etype)
    {
    case ELEM_NODE:
        nextnodeid = anz_eknoten + 1;
        break;

    case ELEM_EDGE:
        nextedgeid = anz_ekanten + 1;
        break;

    case ELEM_FACE:
        nextfaceid = anz_eflaechen + 1;
        break;

    case ELEM_CELL:
        if (anz_ezellen != -1)
        {
            nextcellid = anz_ezellen + 1;
        }
        break;
    default:
        break;
    }
    // 10-4
    return (0);
}

/*
// =======================================================
// CutGeometryBy
// =======================================================
// Schneidet die vorhandene Geometrie mit einer
// Ebene parallel X,Y oder Z-Achse und liefert die
// ermittelten Kanten zurück
ECHAIN *RELDATA::CutGeometryBy(int typ, double val)
{
  const char *funcname="RELDATA::CutGeometryBy";
  int i,startidx,lastidx;
double *vec,n;
ELEMENT *von,*nach,*node;
ECHAIN *root=NULL,*act;
ECHAIN *eroot=NULL;
NODELIST *feld,*neufeld;
clock_t start,end;
int nodecount=0;

start =clock();

switch(typ)
{
case X:
case Y:
case Z:
break;

default:
WriteText(VER_NORMAL,"Methode %s nicht implementiert für Ebene %d\n",funcname,typ);
return(NULL);
}

for (i=0;i<anz_ekanten;++i)
{
if (ekante[i].flags & FLAG_USED)
{
// Schnitt mit der X-Achse
von = ekante[i].element[VON].ptr;
nach = ekante[i].element[NACH].ptr;
if ((von->data[typ].value > val && nach->data[typ].value<= val) ||
(von->data[typ].value < val && nach->data[typ].value>= val))
{
// Diese Kante wird geschnitten
if (root==NULL)
{
root=new ECHAIN;
act=root;
}
else
{
act->next=new ECHAIN;
act=act->next;
}
nodecount++;
act->next=NULL;
act->element = CreateElement(ELEM_NODE,TRUE);
node = act->element;
act->id = node->id;
vec = GetVector(&ekante[i]);
// Mit (X,Y,Z) = Vektor der Kante (Von->nach)
// (x0,Y0,z0) = VON-Punkt
// val = Schnittebene YZ also X=val;
// Es muss die Gleichung gelöst werden: (val,?,?) = (X0,Y0,Z0)+n*(X,Y,Z);
// für die X-Zeile ergibt sich: val = X0+n*X
// Es gilt also: n = (val-x0)/X
n = (val-von->data[typ].value)/vec[typ];
node->data[X].value = von->data[X].value+n*vec[X];
node->data[Y].value = von->data[Y].value+n*vec[Y];
node->data[Z].value = von->data[Z].value+n*vec[Z];
}
}
}

// Jetzt Kanten erstellen
feld = new NODELIST[nodecount];
act=root;
for (i=0;i<nodecount;++i)
{
feld[i].idx = act->id;
feld[i].x   = act->element->data[X].value;
feld[i].y   = act->element->data[Y].value;
feld[i].z   = act->element->data[Z].value;
act=act->next;
}

switch(typ)
{
case X:
neufeld = MergeSort(feld,nodecount,Z);
delete feld;
feld=neufeld;
break;

case Y:
neufeld = MergeSort(feld,nodecount,X);
delete feld;
feld=neufeld;
break;

case Z:
neufeld = MergeSort(feld,nodecount,Y);
delete feld;
feld=neufeld;
break;
}
// Alle Knoten sortiert
// Jetzt Kanten erstellen
startidx=0;
lastidx=0;
for (i=0;i<nodecount;++i) // erster Knoten ist dabei!
{
// Knotenfeld durchlaufen
if (i==0)
{
// im ersten Element wir
}
}

end=clock();

WriteText(VER_NORMAL,"%s Laufzeit: %8.4f\n",funcname,(double)(end-start)/CLOCKS_PER_SEC);
return(root);
}

// =======================================================
// MeltSmallTriangles
// =======================================================
// versucht kleine Dreiecke zu löschen
int RELDATA::MeltSmallTriangles(void)
{
const char *funcname="RELDATA::MeltSmallTriangles";
int meltedfaces=0,trianglecount=0;
int i,j,k,id;
int *nodes,cnt,ecount;
ELEMENT *kante,*knoten,*face,*neuknoten;
ECHAIN *nroot=NULL,*nact,*actedge,*actface;
ECHAIN *froot=NULL,*fact;
BOOL abbruch;

// FIXME: diesen Block löschen
Disconnect(ELEM_ALL);
DeleteUnused(ELEM_ALL);
Reconnect(ELEM_ALL);

for (i=0;i<anz_eflaechen;++i)
{
if (eflaeche[i].flags & FLAG_USED && eflaeche[i].e_anz==3)
{
// Alle einzelnen Dreiecke löschen
trianglecount++;
// alle angrenzenden Flächen werden um eine Kante verringert (wenn möglich)
abbruch=FALSE;
for (j=0;j<3 && abbruch==FALSE;++j)
{
kante = eflaeche[i].element[j].ptr;
actface = kante->up;
ecount=0;
while(actface!=NULL)
{
if (actface->id!=eflaeche[i].id)
{
// diese Fläche testen
if (actface->element->e_anz==3 || !(actface->element->flags & FLAG_USED))
{
// diese Fläche ist ein Dreieck und würde zu einem zweieck degeneriert, abbruch
abbruch=TRUE;
}
}
ecount++;
actface=actface->next;
}
if (ecount>2)
{
WriteText(VER_NORMAL,"%d meldet: Die Kante %d gehoert zu %d Flaechen!\n",funcname,kante->id,ecount);
abbruch=TRUE;
}
}
if (!abbruch)
{
printf("Dreieck gefunden, ID %d\n",eflaeche[i].id);
nodes = GetNodes(&eflaeche[i]);
neuknoten = CreateElement(ELEM_NODE);
// Mittelpunkt berechnen
neuknoten->data[X].value=0;
neuknoten->data[Y].value=0;
neuknoten->data[Z].value=0;
for (j=0;j<3;++j)
{
knoten = GetElementPtr(nodes[j],ELEM_NODE);
neuknoten->data[X].value+=knoten->data[X].value;
neuknoten->data[Y].value+=knoten->data[Y].value;
neuknoten->data[Z].value+=knoten->data[Z].value;
// jetzt die Kanten, die diesen Knoten benutzen umlabeln
actedge = knoten->up;
while(actedge!=NULL)
{
// Kante zu diesem Knoten ermitteln
if (actedge->element->element[VON].id == nodes[j])
{
actedge->element->element[VON].id  = neuknoten->id;
actedge->element->element[VON].ptr = NULL;
}
else if (actedge->element->id != eflaeche[i].id)
{
actedge->element->element[NACH].id  = -neuknoten->id;
actedge->element->element[NACH].ptr = NULL;
}
else
WriteText(VER_NORMAL,"%s meldet: Up Liste des Punkts %d eventuell falsch.\n",funcname,knoten->id);
actedge=actedge->next;
}
}
neuknoten->data[X].value/=3;
neuknoten->data[Y].value/=3;
neuknoten->data[Z].value/=3;
// Kettenelement erstellen
if (nroot==NULL)
{
nroot = new ECHAIN;
nact = nroot;
}
else
{
nact->next=new ECHAIN;
nact=nact->next;
}
nact->next=NULL;
nact->id = neuknoten->id;
nact->element = neuknoten;

// Jetzt die Nachbarflächen zu dieser Fläche rausbekommen
for (j=0;j<3;++j)
{
// Diese Kante aus den Flaechen löschen
kante = eflaeche[i].element[j].ptr; //GetElementPtr(nodes[j+3],ELEM_EDGE);
ecount=0;
actface = kante->up;
while(actface!=NULL)
{
if (actface->id != eflaeche[i].id)
{
face = actface->element;
}
ecount++;
actface=actface->next;
}
if (ecount>2)
WriteText(VER_NORMAL,"%d meldet: Die Kante %d haengt mit %d Flaechen zusammen\n",funcname,kante->id,ecount);

// Diese Fläche sperren und löschen
ChangeFlag(face,FLAG_USED,FALSE,FALSE);
// Dieser Fläche jetzt die aktuelle Kante klauen
if (froot==NULL)
{
froot = new ECHAIN;
fact=froot;
}
else
{
fact->next = new ECHAIN;
fact = fact->next;
}
fact->next=NULL;
fact->element = CreateElement(ELEM_FACE);
fact->id = fact->element->id;
// Daten kopieren
fact->element->e_anz = face->e_anz-1;
fact->element->element = new PTR_ELEM[fact->element->e_anz];
cnt=0;
for (k=0;k<face->e_anz;++k)
{
if (abs(face->element[k].id)!=kante->id)
{
fact->element->element[cnt].id = face->element[k].id;
fact->element->element[cnt++].ptr=NULL;
}
}
}
// Dieses Element komplett löschen
ChangeFlag(&eflaeche[i],FLAG_USED,FALSE,TRUE);
meltedfaces++;
// Knotenliste löschen
delete nodes;
}
}
}

id = Globals.verbose;
Globals.verbose=VER_DEBUG;
Disconnect(ELEM_ALL);
DeleteUnused(ELEM_ALL);
Merge(nroot,ELEM_NODE);
Merge(froot,ELEM_FACE);
Reconnect(ELEM_ALL);
Globals.verbose=id;

PrintStatement();
WriteText(VER_NORMAL,"%s meldet: %d von %d Dreiecken geloescht.\n",funcname,meltedfaces,trianglecount);

return(meltedfaces);
}

// =======================================================
// SwapEdges
// =======================================================
// Versucht zwischen zwei Dreiecken die kürzeste Diagonale zu finden
int RELDATA::SwapEdges(void)
{
const char *funcname="RELDATA::SwapEdges";
int swapcount=0;
int i,k,size,contact;
ECHAIN *faceroot=NULL,*fact;
ECHAIN *nachbar,*actn;
ELEMENT *elem,*kante,*von1,*nach1,*von2,*nach2;
int *nodes,*edges,cnt;
double *avec,*bvec;
BOOL swapped;
clock_t start,end;

start = clock();
// Alles auf "unoptimiert"
ChangeFlag(ELEM_FACE,FLAG_OPT,FALSE);
for (i=0;i<anz_eflaechen;++i)
{
if (eflaeche[i].flags & FLAG_USED && eflaeche[i].e_anz==3)
{
swapped=FALSE;
nachbar = GetNeighbours(&eflaeche[i],1);
actn=nachbar;
while(actn!=NULL && !swapped)
{
elem = actn->element;
if (elem->e_anz==3 && (elem->flags & FLAG_USED) && !(elem->flags & FLAG_OPT))
{
// Ist die Diagonale hierzu ok?
// umrahmendes Rechteck finden
size=6;
edges=new int[size]; // eben pro Dreieck drei
nodes = GetNodes(elem);
for (k=0;k<3;++k)
{
edges[k]    = nodes[k+3];  // Wie gehabt: getNodes liefert auch gleich die Kanten mit
edges[k+3]  = eflaeche[i].element[k].id; // hinten ran die Kanten der aktuellen Fläche
}
delete nodes;
// Berührungskante finden
BubbleSort(edges,size,TRUE);
contact=0;
for (k=0;k<6 && contact==0;++k)
{
if (edges[k]==-edges[k+1])
contact = abs(edges[k]);
}
if (contact==0)
{
WriteText(VER_NORMAL,"%s meldet: Fehler im Gefüge, Flaeche %d und Flaeche %d haben keine gegenläufige Kante gemeinsam: %d %d %d %d %d %d.\n",funcname,eflaeche[i].id, elem->id,edges[0],edges[1],edges[2],edges[3],edges[4],edges[5]);
}
// Jetzt umrandendes Viereck erkennen
edges = KillDoubleIDs(edges,&size); // size wird überschrieben und zu 4
if (size!=4 || contact==0)  // eventuell seltsame Quelldaten!
{
WriteText(VER_NORMAL,"%s meldet: seltsame Geometrie erkannt: Flaeche %d und Flaeche %d haben %d Kanten gemeinsam.\n",funcname,eflaeche[i].id, elem->id,4-size);
delete edges;
// Abbruch
}
else  // alles ok
{
SortEdges(edges,size,SORT_CLOCKWISE);  // in richtige Reihenfolge bringen WICHTIG!
// Jetzt Diagonale ermitteln.
nodes = new int[8]; // pro Kante zwei
cnt=0;
for (k=0;k<4;++k)
{
kante = GetElementPtr(edges[k],ELEM_EDGE);
nodes[cnt++]=kante->element[VON].id;
nodes[cnt++]=-kante->element[NACH].id; // ist jetzt positiv
}
// doppelte Knoten löschen
size=8;
nodes = DeleteDoubleIDs(nodes,&size); // size wird zu 4
// Erste Diagonale: nodes[0]->nodes[2] weil Kanten in richtiger Reihenfolge waren
// zweite Diagonale: nodes[1]->nodes[3];
avec = new double[4];
bvec = new double[4];
von1  = GetElementPtr(nodes[0],ELEM_NODE);
nach1 = GetElementPtr(nodes[2],ELEM_NODE);
avec[X] = von1->data[X].value- nach1->data[X].value;
avec[Y] = von1->data[Y].value- nach1->data[Y].value;
avec[Z] = von1->data[Z].value- nach1->data[Z].value;
avec[B] = sqrt(avec[X]*avec[X]+avec[Y]*avec[Y]+avec[Z]*avec[Z]);

von2  = GetElementPtr(nodes[1],ELEM_NODE);
nach2 = GetElementPtr(nodes[3],ELEM_NODE);
bvec[X] = von2->data[X].value- nach2->data[X].value;
bvec[Y] = von2->data[Y].value- nach2->data[Y].value;
bvec[Z] = von2->data[Z].value- nach2->data[Z].value;
bvec[B] = sqrt(bvec[X]*bvec[X]+bvec[Y]*bvec[Y]+bvec[Z]*bvec[Z]);

// welches ist die kürzere Diagonale?
// ist diese Diagonale die vorhandene Berührungskante?
kante = GetElementPtr(contact,ELEM_EDGE);

if (avec[B]< bvec[B]) // avec ist kürzer
{
// Vergleich: Ist die Kante mit von1->nach1 identisch?
// Dabei sollte der "NACH" Punkt egal sein...
if (!(kante->element[VON].id == von1->id && kante->element[NACH].id == -nach1->id) &&
!(kante->element[NACH].id == -von1->id && kante->element[VON].id == nach1->id))
{
swapped=TRUE;
}
}
else  // bvec ist kürzer
{
if (!(kante->element[VON].id == von2->id && kante->element[NACH].id == -nach2->id) &&
!(kante->element[NACH].id == -von2->id && kante->element[VON].id == nach2->id))
{
swapped=TRUE;
}
}
if (swapped)
{
swapcount++;
// später von BreakDownFaces zurechtschneiden lassen
if (faceroot==NULL)
{
faceroot= new ECHAIN;
fact=faceroot;
}
else
{
fact->next=new ECHAIN;
fact=fact->next;
}
fact->next = NULL;
fact->element = CreateElement(ELEM_FACE);
fact->id = fact->element->id;
fact->element->e_anz = 4;
fact->element->element=new PTR_ELEM[4];
for (k=0;k<4;++k)
{
fact->element->element[k].ptr=NULL;
fact->element->element[k].id=edges[k];
}
// Das wars, den Rest erledigt BreakDownFaces
// Jetzt noch die beteiligten Flächen und die diagonale Kante löschen
ChangeFlag(&eflaeche[i],FLAG_USED,FALSE);
ChangeFlag(elem,FLAG_USED,FALSE);
ChangeFlag(kante,FLAG_USED,FALSE);
}
}
}
// Nächster Nachbar
actn=actn->next;
}
DeleteEChain(nachbar,FALSE);
// Mit dieser Fläche ist man fertig, egal was passiert ist
ChangeFlag(&eflaeche[i],FLAG_OPT,TRUE);
}
}

Disconnect(ELEM_ALL);
DeleteUnused(ELEM_ALL);
Merge(faceroot,ELEM_FACE);
Reconnect(ELEM_ALL);
// erzeugte Vierecke zu optimalen Dreiecken
if (Globals.breakdownto==3) // Diese Methode erzeugt ja vierecke!
BreakDownFaces(3);
end=clock();

WriteText(VER_MAX,"%s meldet: %d Kanten gewechselt [%8.4f sec].\n",funcname,swapcount,(double)(end-start)/CLOCKS_PER_SEC);

return(swapcount);
}
*/
/*
int RELDATA::SwapEdges(void)
{
  const char *funcname="RELDATA::SwapEdges";
  int swapcount=0;
  int i,k,size;
  ECHAIN *faceroot=NULL,*edgeroot=NULL;
  ECHAIN *fact;
  ECHAIN *nachbar,*tmp;
  ELEMENT *elem,*kante,*knoten;
  int *nodes,kante1[2],kante2[2],*edges;
double *avec,*bvec;
BOOL swapped;

for (i=0;i<anz_eflaechen;++i)
{
if (eflaeche[i].flags & FLAG_USED && eflaeche[i].e_anz==3)
{
swapped=FALSE;
nachbar = GetNeighbours(&eflaeche[i],1);
while(nachbar!=NULL && !swapped)
{
elem = nachbar->element;
if (elem->e_anz==3 && (elem->flags & FLAG_USED))
{
// Edgeswapping testen
// schritt1 doppelte Kante suchen
kante=NULL;
edges=new int[6]; // für alle Kanten der zwei Dreiecke
for (k=0;k<3;++k)
{
// tmp zeigt auf die Up-Liste der Kanten des gefundenen Dreiecks
// hier sollte bei einer Kante der ID der aktuellen Fläche auftauchen
tmp = elem->element[k].ptr->up;
if (tmp->element->flags & FLAG_USED)
{
if ((tmp->id == elem->id && tmp->next->id == eflaeche[i].id) || (tmp->next->id == elem->id && tmp->id == eflaeche[i].id))
{
// das hier ist die Kante
kante=elem->element[k].ptr;
kante2[VON] = kante->element[VON].id;
kante2[NACH]= kante->element[NACH].id;  // ist negativ!
}
}
}
// jetzt die restlichen beiden Punkte suchen
// im gefundenen Punkt
bvec=new double[4];
nodes = GetNodes(elem);
for (k=0;k<3;++k)
{
edges[k]=nodes[k+3];
if (nodes[k]!= kante->element[VON].id && nodes[k]!=-kante->element[NACH].id)
{
// das ist unser Punkt!
knoten = GetElementPtr(nodes[k],ELEM_NODE);
bvec[X] = knoten->data[X].value;
bvec[Y] = knoten->data[Y].value;
bvec[Z] = knoten->data[Z].value;
kante1[VON]=nodes[k];
}
}
delete nodes;
// im aktuellen Punkt
nodes = GetNodes(&eflaeche[i]);
for (k=0;k<3;++k)
{
edges[k+3]=nodes[k+3];
if (nodes[k]!= kante->element[VON].id && nodes[k]!=-kante->element[NACH].id)
{
// das ist unser Punkt!
knoten = GetElementPtr(nodes[k],ELEM_NODE);
bvec[X] -= knoten->data[X].value;
bvec[Y] -= knoten->data[Y].value;
bvec[Z] -= knoten->data[Z].value;
kante1[NACH]=-nodes[k]; // jetzt auch negativ
}
}
delete nodes;
bvec[B]=sqrt(bvec[X]*bvec[X]+bvec[Y]*bvec[Y]+bvec[Z]*bvec[Z]);
avec=GetVector(kante);
// Ist das swapping nötig? Ja, wenn |bvec| < |avec| (neue Kante < alte Kante)
if (bvec[B]<avec[B])
{
// Kleiner Trick: wir erstellen ein Viereck und lassen alle Vierecke später von "BreakDownFaces()"
// teilen. Bei Vierecken macht es ja genau diese unterscheidung mit den längeren Diagonalen
if (faceroot==NULL)
{
faceroot= new ECHAIN;
fact=faceroot;
}
else
{
fact->next=new ECHAIN;
fact=fact->next;
}
fact->next = NULL;
fact->element = CreateElement(ELEM_FACE);
fact->id = fact->element->id;
fact->element->e_anz = 4;
fact->element->element=new PTR_ELEM[4];
size=6;
edges = KillDoubleIDs(edges,&size);
SortEdges(edges,size,SORT_CLOCKWISE);
for (k=0;k<4;++k)
{
fact->element->element[k].ptr=NULL;
fact->element->element[k].id=edges[k];
}
// Das wars, den Rest erledigt BreakDownFaces
swapped=TRUE;
swapcount++;
// Jetzt noch die beiden beteiligten Flächen frei geben
ChangeFlag(&eflaeche[i],FLAG_USED,FALSE);
ChangeFlag(elem,FLAG_USED,FALSE);
// Und die Kante hinterher schmeissen
ChangeFlag(kante,FLAG_USED,FALSE);
}
delete avec;
delete bvec;
delete edges;
}
tmp = nachbar;
nachbar=nachbar->next;
tmp->next=NULL;
DeleteEChain(tmp,FALSE);
}
DeleteEChain(nachbar,FALSE);
}
}
WriteText(VER_NORMAL," %d Dreiecke geloescht, %d Kanten geloescht.\n",swapcount*2,swapcount);

Disconnect(ELEM_ALL);
DeleteUnused(ELEM_ALL);
Merge(faceroot,ELEM_FACE);
Reconnect(ELEM_ALL);
// erzeugte Vierecke zu optimalen Dreiecken
BreakDownFaces(3);

return(swapcount);
}
*/

/*
// =======================================================
// LineCheck
// =======================================================
// Ein ideales Dreiecks-Gitter erlaubt zwischen zwei benachbarten
// Knoten genau drei Verbindungen, wobei eine Direkte und zwei indirekte
// dabei sind. Werden mehr Verbindungen gefunden, sollte die indirekte
// mit der kürzesten Kante gelöscht werden.
// sollten alle Verbindungen indirekt sein, sollte ebenfalls eine
// gelöscht werden
// oder anders Formuliert: alle Punkte, andenen nur drei Kanten hängen,
// sollten gelöscht werden. die Nachbarpunkte ergeben dann ein neues Dreieck.
int RELDATA::LineCheck(void)
{
const char *funcname="RELDATA::LineCheck";
int delface=0;
int deledge=0;
int delnode=0;
int i,j,k,num,cnt;
int *faceid,*edgeid,*nodeid;
ECHAIN *nachbar,*actn;
ECHAIN *tmp,*cface,*cedge;
ECHAIN *faceroot=NULL,*actf;
ELEMENT *face;
BOOL holzweg;
clock_t start,end;

start = clock();
for (i=0;i<anz_eknoten;++i)
{
if (eknoten[i].flags & FLAG_USED)
{
nachbar = GetNeighbours(&eknoten[i],1);
actn=nachbar;
num=0;
while(actn!=NULL)
{
if (actn->element->flags & FLAG_USED)
num++;
actn=actn->next;
}
if (num==3 || num==4)   // Erweitert für vier Kanten
{
// Kandidaten gefunden
// Beteiligte Flächen identifizieren
faceid= new int[num*2]; // je zwei pro beteiligter Kante
cnt=0;
cedge = eknoten[i].up;
holzweg=FALSE;
while(cedge!=NULL && !holzweg)
{
cface = cedge->element->up;
while(cface!=NULL)
{
// testen, ob das auch wirklich ein Dreieck ist...
if (cface->element->e_anz!=3 || !(cface->element->flags & FLAG_USED))
holzweg=TRUE;
// Diese Fläche ist OK
faceid[cnt++] = cface->id;
// nächste Fläche
cface=cface->next;
}
cedge=cedge->next;
}
// in cnt steht jetzt die Anzahl der als gültig befundenen Nachbarflächen
// wenn wir nicht auf dem Holzweg sind ;-)

if (!holzweg && cnt==num*2) // für jede Kante zwei Flächen identifiziert
{
// alles ok, die Flächen können gelöscht werden
faceid=DeleteDoubleIDs(faceid,&cnt);
// in dieser Liste stehen jetzt die zu löschenden Flächen
// jetzt die beteiligten Kanten besorgen, das sind für jede Fläche drei
edgeid=new int[num*3];
for (k=0;k<num;++k)
{
face = GetElementPtr(faceid[k],ELEM_FACE);
nodeid = GetNodes(face);
for (j=0;j<3;++j) // alles Dreiecke
edgeid[k*3+j]=nodeid[3+j];

delete nodeid;
ChangeFlag(face,FLAG_USED,FALSE);
delface++;
}
// doppelte Kanten löschen
cnt=num*3;
edgeid = KillDoubleIDs(edgeid,&cnt);
// Jetzt die Kanten freigeben
tmp = eknoten[i].up;
while(tmp!=NULL)
{
ChangeFlag(tmp->element,FLAG_USED,FALSE);
deledge++;
tmp=tmp->next;
}
// Die Kanten für die neue Fläche steht in facid, muß eventuell noch sortiert werden
SortEdges(edgeid,num,SORT_CLOCKWISE);
// Jetzt Fläche erstellen
if (faceroot==NULL)
{
faceroot=new ECHAIN;
actf=faceroot;
}
else
{
actf->next=new ECHAIN;
actf=actf->next;
}
actf->next=NULL;
actf->element = CreateElement(ELEM_FACE);
actf->id = actf->element->id;
face = actf->element;
face->e_anz = num;
face->element=new PTR_ELEM[num];
for (k=0;k<num;++k)
{
face->element[k].id = edgeid[k];
face->element[k].ptr=NULL;
}
// diesen Punkt zum Abschuss frei geben
//          WriteText(VER_DEBUG,"%s erstellt neue Flaeche: %d [%d ecken]\n",funcname,face->id,num);
ChangeFlag(&eknoten[i],FLAG_USED,FALSE);
delnode++;
delete edgeid;
}
delete faceid;
}
DeleteEChain(nachbar,FALSE);
}
}
Disconnect(ELEM_ALL);
DeleteUnused(ELEM_ALL);
Merge(faceroot,ELEM_FACE);
Reconnect(ELEM_ALL);
end = clock();

WriteText(VER_MAX,"%s meldet: %d Flaechen, %d Kanten und %d Punkte geloescht [%8.4f sec].\n",funcname,delface,deledge,delnode,(double)(end-start)/CLOCKS_PER_SEC);

return (delnode);
}

// =======================================================
// RefineMesh
// =======================================================
int RELDATA::RefineMesh(double gradval)
{
const char *funcname="RELDATA::RefineMesh";
return(0);
}
*/

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// EOF, jetzt noch undefs
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
