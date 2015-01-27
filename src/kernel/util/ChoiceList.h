/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CHOICE_LIST_H_
#define __CHOICE_LIST_H_

#include "coExport.h"
#include <iostream>
#include <cstring>

namespace covise
{

class UTILEXPORT ChoiceList
{

private:
    ChoiceList(const ChoiceList &);
    ChoiceList &operator=(const ChoiceList &);
    ChoiceList();

    char *choiceString[64];
    int choiceNo[64];
    int numChoices;

public:
    ChoiceList(const char *choice0, int choiceNo0)
    {
        choiceString[0] = new char[strlen(choice0) + 1];
        strcpy(choiceString[0], choice0);
        choiceNo[0] = choiceNo0;
        numChoices = 1;
    }

    int add(const char *choice, int number)
    {
        int retval;
        if (numChoices == 64)
            retval = 1;
        else
        {
            choiceString[numChoices] = new char[strlen(choice) + 1];
            strcpy(choiceString[numChoices], choice);
            choiceNo[numChoices] = number;
            numChoices++;
            retval = 0;
        }
        return retval;
    }

    int change(const char *newchoice, int number)
    {
        delete[] choiceString[number];
        choiceString[number] = new char[strlen(newchoice) + 1];
        strcpy(choiceString[number], newchoice);
        return 0;
    }
    int get_orig_num(int choice) const
    {
        return choiceNo[choice];
    }

    const char *getString(int choice)
    {
        return choiceString[choice];
    }

    const char *const *get_strings() const
    {
        return choiceString;
    }

    int get_num() const
    {
        return numChoices;
    }

    friend std::ostream &operator<<(std::ostream &, const ChoiceList &);
};

std::ostream &operator<<(std::ostream &, const ChoiceList &);
}
#endif
