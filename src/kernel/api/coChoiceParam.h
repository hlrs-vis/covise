/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_CHOICE_PARAM_H_
#define _CO_CHOICE_PARAM_H_

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
#include "coUifPara.h"

namespace covise
{

/// parameter to choose values from a list
class APIEXPORT coChoiceParam : public coUifPara
{

private:
    enum
    {
        MAX_CHOICE_LABELS = 256
    };

    /// Copy-Constructor: NOT  IMPLEMENTED
    coChoiceParam(const coChoiceParam &);

    /// Assignment operator: NOT  IMPLEMENTED
    coChoiceParam &operator=(const coChoiceParam &);

    /// Default constructor: NOT  IMPLEMENTED
    coChoiceParam();

    /// my type info
    static coUifPara::Typeinfo s_paraType;

    /// Port data fields
    int d_numChoices;
    int d_activeChoice;
    char **d_choice;

    // Parameter type name
    static const char *s_type;

public:
    /// Constructor
    coChoiceParam(const char *name, const char *desc);

    /// Destructor : virtual in case we derive objects
    virtual ~coChoiceParam();

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

    /// string compare modes for update
    /// ALPHA: compare only alphanumeric characters
    typedef enum
    {
        CASE_SENSITIVE,
        CASE_INSENSITIVE,
        ALPHA
    } cmpMode;

    /// update the value: if possible select string from current selection in new choice list
    /// return 0 on error
    int updateValue(int numChoices, const char *const *choice, int actChoice, cmpMode cmp = CASE_INSENSITIVE);

    /// set/update the value: return 0 on error
    int setValue(int numChoices, const char *const *choice, int actChoice);
    int setValue(int numChoices, const std::vector<string> &list, int actChoice);

    /// set/update the value: return 0 on error
    int setValue(int actChoice);

    /// set/update the value by giving choice label: return 0 on error
    int setValue(const char *choiceLabel);

    /// get the value
    int getValue() const;

    /// the the name of the label of the current selection
    const char *getActLabel() const;

    /// get label on number given number [1...numChoices]
    const char *getLabel(int i) const;

    /// get number of choices
    int getNumChoices() const;

    /// get the type string of this parameter
    virtual const char *getTypeString() const;

    /// get the value of this parameter as a string
    virtual const char *getValString() const;

    /// set the value of this parameter from a string
    virtual void setValString(const char *str);
};
}
#endif
