/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_USERSTUDY_H_
#define _CUI_USERSTUDY_H_

#include <osg/Geometry>

#include <Interaction.h>
#include <Dial.h>
#include <Button.h>
#include <FloatOMeter.h>
#include <Calculator.h>
#include <Message.h>

class UserStudy : public cui::Widget, public cui::FloatOMeterListener, public cui::CalculatorListener, public cui::CardListener
{
public:
    enum NumberMode
    {
        SMALL_NUMBERS,
        LARGE_NUMBERS,
        ADJUST_NUMBERS,
        SEARCH
    };

    UserStudy(cui::Interaction *);
    ~UserStudy();

    void setMethod(cui::Dial::AdvancedInput);
    void setNumberMode(NumberMode);
    void generateNumbers(NumberMode, int, int);
    void setNextValue();
    void restart();
    void update();

    virtual void setLogFile(cui::LogFile *);
    virtual void setVisible(bool);

private:
    static const char *SMALL_NUMBERS_FILENAME;
    static const char *LARGE_NUMBERS_FILENAME;
    static const char *ADJUST_NUMBERS_FILENAME;
    static const char *SEARCH_FILENAME;
    static const double EPSILON;
    static const float TOP;
    static const float BOTTOM;
    static const float ACCEPT_SEARCH_RANGE;

    cui::Dial *_dial;
    cui::Message *_msg;
    cui::Message *_errorMsg;
    cui::Message *_topMark;
    cui::Message *_bottomMark;
    cui::Message *_moveableMark;
    cui::Message *_rangeMark;
    cui::Button *_done;

    FILE *_file;

    double _valueToSet;
    double _rangeMin, _rangeMax;
    double *_values;
    double *_rangeMins;
    double *_rangeMaxs;
    double _maxDiff;
    int _counter;
    int _numberOfValues;
    bool _nextValue;
    bool _inputStarted;

    cui::Dial::AdvancedInput _method;
    char *_methodText;
    NumberMode _numberMode;
    char *_numberModeText;

    void shakeValues();

    virtual void valueSet(double){};
    virtual void calculatorOpened();
    virtual void calculatorClosed();

    virtual void floatOMeterValueChanged(double){};
    virtual void floatOMeterOpened();
    virtual void floatOMeterClosed();

    virtual bool cardButtonEvent(cui::Card *, int, int);
    virtual bool cardCursorUpdate(cui::Card *, cui::InputDevice *)
    {
        return false;
    };
};

#endif
