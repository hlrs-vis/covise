/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coColormapChoiceParam Parameter handling class                         +
// +                                                                         +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>
#include "coColormapChoiceParam.h"
#include "coBlankConv.h"
#include <appl/ApplInterface.h>

using namespace covise;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Static data
const char *coColormapChoiceParam::s_type = "COLORMAPCHOICE";
coUifPara::Typeinfo coColormapChoiceParam::s_paraType = coUifPara::numericType("COLORMAPCHOICE");

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Constructor

coColormapChoiceParam::coColormapChoiceParam(const char *name, const char *desc)
    : coUifPara(name, desc)
{
    colormaps = new TColormapChoice[1];
    colormaps[0].mapName = ("Standard");
    colormaps[0].mapValues.push_back(0.);
    colormaps[0].mapValues.push_back(0.);
    colormaps[0].mapValues.push_back(1.);
    colormaps[0].mapValues.push_back(1.);
    colormaps[0].mapValues.push_back(0.);

    colormaps[0].mapValues.push_back(1.);
    colormaps[0].mapValues.push_back(0.);
    colormaps[0].mapValues.push_back(0.);
    colormaps[0].mapValues.push_back(1.);
    colormaps[0].mapValues.push_back(0.5);

    colormaps[0].mapValues.push_back(1.);
    colormaps[0].mapValues.push_back(1.);
    colormaps[0].mapValues.push_back(0.);
    colormaps[0].mapValues.push_back(1.);
    colormaps[0].mapValues.push_back(1.);

    d_numChoices = 1;
    d_activeChoice = 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Destructor : virtual in case we derive objects

coColormapChoiceParam::~coColormapChoiceParam()
{
    delete[] colormaps;
    d_numChoices = d_activeChoice = 0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Check the type

int coColormapChoiceParam::isOfType(coUifPara::Typeinfo type)
{
    return (type == s_paraType);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get my type

coUifPara::Typeinfo coColormapChoiceParam::getType()
{
    return s_paraType;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// handle parameter changes: called by paramCB

int coColormapChoiceParam::paramChange()
{
    int no_of_choices = Covise::get_reply_colormapchoice(&d_activeChoice);
    if (no_of_choices == 0)
        return 0;

    // delete old colormaps
    if (colormaps)
        delete[] colormaps;

    colormaps = new TColormapChoice[no_of_choices];
    int res = Covise::get_reply_colormapchoice(colormaps);
    return res;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// give dafault values to Covise -> automatically called !

void coColormapChoiceParam::initialize()
{
    string text;
    if (d_numChoices)
    {
        ostringstream os;
        os << d_activeChoice << " " << d_numChoices;
        for (int i = 0; i < d_numChoices; i++)
        {
            int ll = colormaps[i].mapValues.size();
            os << " " << colormaps[i].mapName << " " << ll / 5;
            for (int k = 0; k < ll; k++)
                os << " " << colormaps[i].mapValues[k];
        }

        text = os.str();
    }

    else
        text = "1 1 Standard 3 0.0 0.0 1.0 1.0 0.0 1.0 0.0 0.0 1.0 0.5 1.0 1.0 0.0 1.0 1.0";

    d_defString = new char[text.length() + 2];
    strcpy(d_defString, text.c_str());
    Covise::add_port(PARIN, d_name, "ColormapChoice", d_desc);
    Covise::set_port_default(d_name, d_defString);

    d_init = 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// print this to a stream

void coColormapChoiceParam::print(ostream & /*str*/) const
{
    /*   coUifPara::print(str);
   cerr << "Choice : " << d_numChoices
      << " selections, active is " << d_activeChoice
      << endl;
   for (int i=0;i<d_numChoices;i++)
   {
      cerr << "    '" << d_choice[i] << "'" << endl;
   }
*/
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value: if called after init() : update on map

int coColormapChoiceParam::setValue(int numChoices, int actChoice, TColormapChoice *list)
{

    if (colormaps)
        delete[] colormaps;

    colormaps = new TColormapChoice[numChoices];

    d_numChoices = numChoices;
    d_activeChoice = actChoice + 1;
    for (int i = 0; i < d_numChoices; i++)
    {
        colormaps[i].mapName = list[i].mapName;
        colormaps[i].mapValues.assign(list[i].mapValues.begin(), list[i].mapValues.end());
    }
    if (d_init)
        return Covise::update_colormapchoice_param(d_name, d_numChoices, d_activeChoice, colormaps);

    else
        return 1;
}

/// set the value: if called after init() : update on map

int coColormapChoiceParam::setValue(int actChoice)
{
    if (colormaps)
    {
        d_activeChoice = actChoice + 1;

        if (d_init)
            return Covise::update_colormapchoice_param(d_name, d_numChoices, d_activeChoice, colormaps);

        else
            return 1;
    }
    return 0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value

int coColormapChoiceParam::getValue() const
{
    return d_activeChoice - 1; // return 0 - n-1
}

TColormapChoice coColormapChoiceParam::getValue(int i) const
{
    return colormaps[i];
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// read out the choice labels

/// get label on number given number
const string coColormapChoiceParam::getLabel(int i) const
{
    if (i < 0 || i >= d_numChoices)
        return NULL;
    else
        return colormaps[i].mapName;
}

const string coColormapChoiceParam::getActLabel() const
{
    return getLabel(d_activeChoice - 1);
}

/// get number of choices
int coColormapChoiceParam::getNumChoices() const
{
    return d_numChoices;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the type string of this parameter
const char *coColormapChoiceParam::getTypeString() const
{
    return s_type;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value of this parameter as a string
const char *coColormapChoiceParam::getValString() const
{
    return ("Don't know what to do");
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value of this parameter from a string
void coColormapChoiceParam::setValString(const char *)
{
    cerr << "--------------------------- setValString " << endl;
}
