/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_COLORMAPCHOICE_PARAM_H_
#define _CO_COLORMAPCHOICE_PARAM_H_

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coColormapChoiceParam Parameter handling class                            +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>
#include <appl/CoviseBase.h>
#include "coUifPara.h"

namespace covise
{

/// parameter to choose values from a list
class APIEXPORT coColormapChoiceParam : public coUifPara
{
public:
    /// Constructor
    coColormapChoiceParam(const char *name, const char *desc);

    /// Destructor : virtual in case we derive objects
    virtual ~coColormapChoiceParam();

    /// Check the type
    virtual int isOfType(coUifPara::Typeinfo type);

    /// get my type
    static coUifPara::Typeinfo getType();

    /// handle parameter changes: called by paramCB
    virtual int paramChange();

    /// give dafault values to Covise -> automatically called !
    virtual void initialize();

    /// print this to a stream
    virtual void print(ostream &str) const;

    /// set/update the value: return 0 on error
    int setValue(int numChoices, int actChoice, TColormapChoice *list);

    /// set/update the value: return 0 on error
    int setValue(int actChoice);

    /// get the value
    int getValue() const;

    /// get material info
    TColormapChoice getValue(int i) const;

    /// the the name of the label of the current selection
    const string getActLabel() const;

    /// get label on number given number [1...numChoices]
    const string getLabel(int i) const;

    /// get number of choices
    int getNumChoices() const;

    /// get the type string of this parameter
    virtual const char *getTypeString() const;

    /// get the value of this parameter as a string
    virtual const char *getValString() const;

    /// set the value of this parameter from a string
    virtual void setValString(const char *str);

private:
    enum
    {
        MAX_CHOICE_LABELS = 256
    };

    /// Copy-Constructor: NOT  IMPLEMENTED
    coColormapChoiceParam(const coColormapChoiceParam &);

    /// Assignment operator: NOT  IMPLEMENTED
    coColormapChoiceParam &operator=(const coColormapChoiceParam &);

    /// Default constructor: NOT  IMPLEMENTED
    coColormapChoiceParam();

    /// my type info
    static coUifPara::Typeinfo s_paraType;

    /// Port data fields
    int d_numChoices;
    int d_activeChoice;
    TColormapChoice *colormaps;

    // Parameter type name
    static const char *s_type;
};
}
#endif
