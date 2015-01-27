/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CHOICE_LIST_H_
#define __CHOICE_LIST_H_

#include <string.h>
#include <iostream.h>

class ChoiceList
{

private:
    ChoiceList(const ChoiceList &);
    ChoiceList &operator=(const ChoiceList &);
    ChoiceList();

    char *choiceString[64];
    int choiceNo[64];
    int numChoices;

public:
    ~ChoiceList(){};
    ChoiceList(const char *choice0, int choiceNo0)
    {
        choiceString[0] = new char[strlen(choice0) + 2];
        strcpy(choiceString[0], choice0);
        strcat(choiceString[0], "\n");
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
            choiceString[numChoices] = new char[strlen(choice) + 2];
            strcpy(choiceString[numChoices], choice);
            strcat(choiceString[numChoices], "\n");
            choiceNo[numChoices] = number;
            numChoices++;
            retval = 0;
        }
        return retval;
    }

    int get_orig_num(int choice) const
    {
        return choiceNo[choice - 1];
    }

    const char *const *get_strings() const
    {
        return choiceString;
    }

    int get_num() const
    {
        return numChoices;
    }

    friend ostream &operator<<(ostream &, const ChoiceList &);
};

ostream &operator<<(ostream &, const ChoiceList &);
#endif
