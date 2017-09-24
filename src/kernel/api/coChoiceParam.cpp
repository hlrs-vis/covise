/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coChoiceParam Parameter handling class                                +
// +                                                                         +
// +                           Andreas Werner                                +
// +         (C)  Computing Center University of Stuttgart                   +
// +                          Allmandring 30a                                +
// +                          70550 Stuttgart                                +
// + Date:  19.07.99                                                         +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>
#include "coChoiceParam.h"
#include "coBlankConv.h"
#include <appl/ApplInterface.h>
#include <util/unixcompat.h>

using namespace covise;

/////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
// Helper: change all blanks in string to \177
static void blankConvert(char *str)
{
    while (str && *str)
    {
        if (*str == ' ')
            *str = '\177';
        ++str;
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Static data
const char *coChoiceParam::s_type = "CHOICE";
coUifPara::Typeinfo coChoiceParam::s_paraType = coUifPara::numericType("CHOICE");

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Constructor

coChoiceParam::coChoiceParam(const char *name, const char *desc)
    : coUifPara(name, desc)
{
    d_numChoices = d_activeChoice = 0;
    d_choice = NULL;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Destructor : virtual in case we derive objects

coChoiceParam::~coChoiceParam()
{
    if (d_numChoices)
    {
        for (int i = 0; i < d_numChoices; i++)
            delete[] d_choice[i];
        delete[] d_choice;
        d_choice = NULL;
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Check the type

int coChoiceParam::isOfType(coUifPara::Typeinfo type)
{
    return (type == s_paraType);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get my type

coUifPara::Typeinfo coChoiceParam::getType()
{
    return s_paraType;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// handle parameter changes: called by paramCB

int coChoiceParam::paramChange()
{
    int no_of_choices = Covise::get_reply_choice(&d_activeChoice);
    if (no_of_choices == 0)
        return 0;

    // delete old labels
    for (int i = 0; i < d_numChoices; i++)
        delete[] d_choice[i];
    delete[] d_choice;
    d_numChoices = no_of_choices;

    // static one is easier here
    d_choice = new char *[d_numChoices];

    int res = 0;
    for (int i = 0; i < d_numChoices; i++)
    {
        string label;
        res = Covise::get_reply_choice(i, &label);
        if (res != 0)
        {
            d_choice[i] = new char[label.length() + 1];
            strcpy(d_choice[i], (char *)label.c_str());
            blankConvert(d_choice[i]);
        }
    }
    return res;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// give dafault values to Covise -> automatically called !

void coChoiceParam::initialize()
{
    if (d_numChoices)
    {
        // count the length
        size_t i, length = 0;
        for (i = 0; i < d_numChoices; i++)
            length += strlen(d_choice[i]) + 2;
        length += 16; // for selection

        // we must allocate this due to Covise Appl-Lib impelentation bugs
        d_defString = new char[length];

        sprintf(d_defString, "%d ", d_activeChoice);
        for (i = 0; i < d_numChoices; i++)
        {
            // @@@@@ char *buffer = coBlankConv::all(d_choice[i]);
            // @@@@@ strcat(d_defString,buffer);
            // @@@@@ delete [] buffer;
            strcat(d_defString, d_choice[i]);
            strcat(d_defString, " ");
        }
    }
    else
    {
        d_defString = new char[8];
        strcpy(d_defString, "1 ---");
    }
    Covise::add_port(PARIN, d_name, "Choice", d_desc);
    Covise::set_port_default(d_name, d_defString);

    d_init = 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// print this to a stream

void coChoiceParam::print(ostream &str) const
{
    coUifPara::print(str);
    cerr << "Choice : " << d_numChoices
         << " selections, active is " << d_activeChoice
         << endl;
    for (int i = 0; i < d_numChoices; i++)
    {
        cerr << "    '" << d_choice[i] << "'" << endl;
    }
}

/// make a string of only alphanumeric characters
void makeAlpha(char *tgt, const char *src)
{
    int src_pos = 0, tgt_pos = 0;
    char c;

    while ((c = src[src_pos++]))
    {
        if (isalpha(c))
        {
            tgt[tgt_pos++] = c;
        }
    }
    tgt[tgt_pos] = '\0';
}

int coChoiceParam::updateValue(int numChoices,
                               const char *const *choice,
                               int actChoice,
                               cmpMode cmp)
{
    int curChoice = actChoice;
    char label_i[512], actLabel[512];
    bool end = false;

    // assume choice number one has the content "nothing selected"
    if (d_choice && d_activeChoice != 1 && d_numChoices > 1)
    {

        if (cmp == ALPHA)
        {
            makeAlpha(actLabel, d_choice[d_activeChoice - 1]);
        }
        for (int i = 0; i < numChoices && !end; i++)
        {
            //const char *cmp_label = coBlankConv::all(choice[i]);
            char *cmp_label = strcpy(new char[1 + strlen(choice[i])], choice[i]);
            blankConvert(cmp_label);

            switch (cmp)
            {

            case CASE_SENSITIVE:
            {
                if (strcmp(d_choice[d_activeChoice - 1], cmp_label) == 0)
                {
                    curChoice = i;
                    end = true;
                }
                break;
            }

            case CASE_INSENSITIVE:
            {
                if (strcasecmp(d_choice[d_activeChoice - 1], cmp_label) == 0)
                {
                    curChoice = i;
                    end = true;
                }
                break;
            }

            case ALPHA:
            {
                makeAlpha(label_i, choice[i]);
                if (strcasecmp(actLabel, label_i) == 0)
                {
                    curChoice = i;
                    end = true;
                }
                break;
            }
            }
        }
    }

    return setValue(numChoices, choice, curChoice);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value: if called after init() : update on map

int coChoiceParam::setValue(int numChoices, const vector<string> &list, int actChoice)
{
    if (d_choice)
    {
        for (int i = 0; i < d_numChoices; i++)
            delete[] d_choice[i];
        delete[] d_choice;
        d_choice = NULL;
    }

    if (numChoices)
    {
        d_numChoices = numChoices;
        d_activeChoice = actChoice + 1;
        d_choice = new char *[d_numChoices];
        for (int i = 0; i < d_numChoices; i++)
        {
            if (i < list.size())
                d_choice[i] = strcpy(new char[1 + list[i].length()], list[i].c_str());
            else
                d_choice[i] = strcpy(new char[1 + 100], "UNINITIALIZED: VECTOR TOO SHORT");
            blankConvert(d_choice[i]);
        }
        if (d_init)
            return Covise::update_choice_param(d_name, d_numChoices, d_choice, d_activeChoice);

        else
            return 1;
    }

    return 0;
}

int coChoiceParam::setValue(int numChoices, const char *const *choice,
                            int actChoice)
{
    // if we had one before: erase it
    if (d_choice)
    {
        for (int i = 0; i < d_numChoices; i++)
            delete[] d_choice[i];
        delete[] d_choice;
        d_choice = NULL;
    }

    if (numChoices && choice)
    {
        d_numChoices = numChoices;
        d_activeChoice = actChoice + 1;
        d_choice = new char *[d_numChoices];
        for (int i = 0; i < d_numChoices; i++)
        {
            //d_choice[i] = coBlankConv::all(choice[i]);
            d_choice[i] = strcpy(new char[1 + strlen(choice[i])], choice[i]);
            blankConvert(d_choice[i]);
        }
        if (d_init)
            return Covise::update_choice_param(d_name, d_numChoices, d_choice, d_activeChoice);

        else
            return 1;
    }

    return 0;
}

/// set the value: if called after init() : update on map

int coChoiceParam::setValue(int actChoice)
{
    if (d_numChoices && d_choice)
    {
        d_activeChoice = actChoice + 1;

        if (d_init)
            return Covise::update_choice_param(d_name, d_numChoices, d_choice, d_activeChoice);

        else
            return 1;
    }
    return 0;
}

/// set the value: if called after init() : update on map

int coChoiceParam::setValue(const char *choiceLabel)
{
    //char *convertedLabel = coBlankConv::all(choiceLabel);
    char *convertedLabel = strcpy(new char[1 + strlen(choiceLabel)], choiceLabel);
    blankConvert(convertedLabel);

    // look for label
    int actChoice = 0;
    while (actChoice < d_numChoices
           && strcmp(convertedLabel, d_choice[actChoice]) != 0)
        actChoice++;
    delete[] convertedLabel;

    if (actChoice == d_numChoices || actChoice == 0) // not found it
        return 0;
    else // found it
        return setValue(actChoice);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value

int coChoiceParam::getValue() const
{
    return d_activeChoice - 1; // return 0 - n-1
}

const char *coChoiceParam::getActLabel() const
{
    return getLabel(d_activeChoice - 1);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// read out the choice labels

/// get label on number given number
const char *coChoiceParam::getLabel(int i) const
{
    if (i < 0 || i >= d_numChoices)
        return NULL;
    else
        return d_choice[i];
}

/// get number of choices
int coChoiceParam::getNumChoices() const
{
    return d_numChoices;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the type string of this parameter
const char *coChoiceParam::getTypeString() const
{
    return s_type;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value of this parameter as a string
const char *coChoiceParam::getValString() const
{
    static char *valString = NULL;
    size_t i, len = 32; // 2 ints : num/active
    for (i = 0; i < d_numChoices; i++)
        len += 2 * strlen(d_choice[i]) + 2; //  2* for masking worst case
    delete[] valString;
    valString = new char[len];
    sprintf(valString, "%d %d $", d_numChoices, d_activeChoice);

    // append all labels, $ separated, $ and \ masked
    char *vPtr = valString + strlen(valString);
    for (i = 0; i < d_numChoices; i++)
    {
        char *cPtr = d_choice[i];
        while (*cPtr)
        {
            if (*cPtr == '$' || *cPtr == '\\')
            {
                *vPtr = '\\';
                vPtr++;
            }
            *vPtr = *cPtr;
            vPtr++;
            cPtr++;
        }
        *vPtr = '$';
        vPtr++;
    }
    // vPtr--;
    *vPtr = '\0';
    return valString;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value of this parameter from a string
void coChoiceParam::setValString(const char *str)
{
    int i;
    for (i = 0; i < d_numChoices; i++)
        delete[] d_choice;
    delete[] d_choice;
    d_choice = NULL;

    size_t retval;
    retval = sscanf(str, "%d %d", &d_numChoices, &d_activeChoice);
    if (retval != 2)
    {
        std::cerr << "coChoiceParam::setValString: sscanf failed" << std::endl;
        return;
    }

    if (d_numChoices == 0)
        return;

    //skip to first '$' (cannot be a masked one)
    while (*str && *str != '$')
        str++;

    // allocate choice label array
    d_choice = new char *[d_numChoices];

    for (i = 0; i < d_numChoices; i++)
    {
        // skip leading '$'
        str++;

        // find end and count elements
        const char *ePtr = str + 1;
        int len = 0;
        while (*ePtr != '$'
               || (ePtr[-1] == '\\' && ePtr[-2] != '\\'))
        {
            len++;
            ePtr++;
        }
        d_choice[i] = new char[len + 1];
        char *dPtr = d_choice[i];
        while (str != ePtr)
        {
            *dPtr = *str;
            dPtr++;
            str++;
        }
        *dPtr = '\0';
    }
}
