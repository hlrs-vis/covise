/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#define COIDENT "$Header: /vobs/covise/src/application/general/READ_PAM/ReadDSY2.cpp /main/vir_main/1 18-Dec-2001.11:12:04 we_te $"
#include <util/coIdent.h>

#include "ReadDSY.h"

// special global stuff

void pad(char *title, int num)
{
    int i;
    for (i = num; i >= 0; --i)
    {
        if (title[i] == ' ' || title[i] == '\n')
            title[i] = '\0';
        if (title[i] != '\0')
            break;
    }
}

void ReadDSY::addSpecialGlobals(coStringObj::ElemType typa)
{
    int itypa = typa;
    int noTitle = 0; // assume we have titles
    int nall, iall;
    char *titles = 0;
    int *ids = 0;

    const char *what[] = {
        "material", "trans. sec.", "cont. inter.",
        "rig. wall", "ab./fl. cell", "Not used",
        "ab. cham.", "ab. wall"
    };
    const char *matVars[] = {
        "Int. energy", "Transl. kin. energy",
        "Hourglass energy", "Equiv. membrane energy"
    };
    const char *trSecVars[] = {
        "Sec. X Force", "Sec. Y Force", "Sec. Z Force"
                                        "Sec. X Mom.",
        "Sec. Y Mom.", "Sec. Z Mom."
    };
    const char *coIntVars[] = {
        "Cont. X Force", "Cont. Y Force", "Cont. Z Force",
        "Elas. cont. energy", "Frict. cont. energy", "Cont. penetration"
    };
    const char *rigWVars[] = {
        "R. wall X force", "R. wall Y force", "R. wall Z force",
        "R. wall veloc.", "R. wall energy"
    };
    const char *abVars[] = {
        "AB gas pres./FC vol.", "AB vol./FC pres.",
        "AB gas Temp.", "AB gas Mass", "AB inlet mass",
        "AB outlet mass", "AB outer surf."
    };
    const char *abChVars[] = {
        "Cham. gas pres.", "Cham. gas vol.", "Cham. gas temp.",
        "Gas mass in cham.", "Mass from infl. into cham.",
        "Leak. mass through fabric", "Mass inflow from other chams.",
        "Mass outflow to neigh. chams.", "Cham. outer skin surf."
    };
    const char *abWVars[] = {
        "Surf. of wall opening",
        "Cum. mass flow through the wall (1->2)",
        "Cum. mass flow through the wall (2->1)",
        "Total cum. mass through the wall"
    };

    const char **Vars;
    int limit;
    switch (typa)
    {
    case coStringObj::MATERIAL:
        limit = MAT_GLOBAL_REDUCE;
        Vars = matVars;
        break;
    case coStringObj::TRANS_SECTION:
        limit = TS_GLOBAL_REDUCE;
        Vars = trSecVars;
        break;
    case coStringObj::CONTACT_INTERFACE:
        limit = CI_GLOBAL_REDUCE;
        Vars = coIntVars;
        break;
    case coStringObj::RIGID_WALL:
        limit = RW_GLOBAL_REDUCE;
        Vars = rigWVars;
        break;
    case coStringObj::AIRBAG:
        limit = AB_GLOBAL_REDUCE;
        Vars = abVars;
        break;
    case coStringObj::AIRBAG_CHAM:
        limit = ABCH_GLOBAL_REDUCE;
        Vars = abChVars;
        break;
    case coStringObj::AIRBAG_WALL:
        limit = ABW_GLOBAL_REDUCE;
        Vars = abWVars;
        break;
    default:
        Covise::sendWarning("Not an accepted type for ReadDSY::addSpecialGlobals");
        return;
    }

    const char *whatNow;
    whatNow = what[itypa - coStringObj::MATERIAL];

    dsyhal_(&itypa, &nall, &iall, &rtn_);
    if (rtn_ == 0 && itypa >= coStringObj::MATERIAL && itypa <= coStringObj::AIRBAG)
    {
        titles = new char[48 * nall];
        ids = new int[nall];
        dsynam_(&itypa, &nall, titles, ids, &rtn_);
        if (rtn_) // make our own titles...
        {
            int counter;
            noTitle = 1;
            if (nall > limit)
                nall = limit;
            for (counter = 0; counter < nall; ++counter)
            {
                ids[counter] = counter;
                std::string myOwnTitle(whatNow);
                myOwnTitle += " Var. ";
                char labNum[16];
                sprintf(labNum, "%d", counter);
                myOwnTitle += labNum;
                strcpy(titles + 48 * counter, myOwnTitle.c_str());
            }
            rtn_ = 0;
        }
    }
    if (rtn_ == 0 && itypa == coStringObj::AIRBAG_CHAM)
    {
        noTitle = 1; // dsynam not supported!?
        titles = new char[48 * nall];
        ids = new int[2 * nall];
        dsyair_(&itypa, ids, &rtn_);
        if (rtn_ == 0) // make our own titles...
        {
            int counter;
            if (nall > limit)
                nall = limit;
            for (counter = 0; counter < nall; ++counter)
            {
                std::string myOwnTitle(whatNow);
                myOwnTitle += " ";
                char labNum[16];
                sprintf(labNum, "%d", ids[2 * counter]);
                myOwnTitle += labNum;
                myOwnTitle += " airbag ";
                sprintf(labNum, "%d", ids[2 * counter + 1]);
                myOwnTitle += labNum;
                strcpy(titles + 48 * counter, myOwnTitle.c_str());
            }
        }
    }
    if (rtn_ == 0 && itypa == coStringObj::AIRBAG_WALL)
    {
        noTitle = 1; // dsynam not supported!?
        titles = new char[48 * nall];
        ids = new int[4 * nall];
        dsyair_(&itypa, ids, &rtn_);
        if (rtn_ == 0) // make our own titles...
        {
            int counter;
            if (nall > limit)
                nall = limit;
            for (counter = 0; counter < nall; ++counter)
            {
                std::string myOwnTitle(whatNow);
                myOwnTitle += " ";

                char labNum[16];
                sprintf(labNum, "%d", ids[2 * counter]);
                myOwnTitle += labNum;

                myOwnTitle += " ch1: ";
                sprintf(labNum, "%d", ids[2 * counter + 1]);
                myOwnTitle += labNum;

                myOwnTitle += " ch2: ";
                sprintf(labNum, "%d", ids[2 * counter + 2]);
                myOwnTitle += labNum;

                myOwnTitle += " ab: ";
                sprintf(labNum, "%d", ids[2 * counter + 3]);
                myOwnTitle += labNum;

                strcpy(titles + 48 * counter, myOwnTitle.c_str());
            }
        }
    }
    if (rtn_)
    {
        Covise::sendWarning("No titles or identifier info for %s", whatNow);
    }
    else if (iall == 0)
    {
        Covise::sendWarning("No %s variables", whatNow);
    }
    else
    {
        int i;
        int mat;
        if (nall > limit)
            nall = limit;
        for (mat = 0; mat < nall; ++mat)
        {
            char thisTitle[TITLE_MAT + 1];
            memset(thisTitle, '\0', TITLE_MAT + 1);
            strncpy(thisTitle, titles + 48 * mat, TITLE_MAT);
            pad(thisTitle, TITLE_MAT);
            std::string coTitle(thisTitle);
            if (noTitle == 0) // only if we have real titles,...
            {
                coTitle += " ";
                coTitle += whatNow;
                sprintf(thisTitle, " label(%d)", ids[mat]);
                coTitle += thisTitle;
            }
            /* else {
                     coTitle += " ";
                  } */
            for (i = 1; i <= iall; ++i)
            {
                std::string coTitleVar(coTitle);
                coTitleVar += ", ";
                coTitleVar += Vars[i - 1];
                global_contents_.add(coTitleVar.c_str(), coStringObj::SCALAR,
                                     typa, &i, '1', mat + 1);
            }
        }
    }
    delete[] titles;
    delete[] ids;
}
