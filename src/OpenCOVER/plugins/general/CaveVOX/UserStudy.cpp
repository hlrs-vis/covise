/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// CUI:
#include <Message.h>

// Local:
#include "UserStudy.h"

using namespace cui;
using namespace osg;

const char *UserStudy::SMALL_NUMBERS_FILENAME = "smallNumbers.txt";
const char *UserStudy::LARGE_NUMBERS_FILENAME = "largeNumbers.txt";
const char *UserStudy::ADJUST_NUMBERS_FILENAME = "adjustNumbers.txt";
const char *UserStudy::SEARCH_FILENAME = "search.txt";
const double UserStudy::EPSILON = 1.0e-7;
const float UserStudy::TOP = 7.0;
const float UserStudy::BOTTOM = 1.0;
const float UserStudy::ACCEPT_SEARCH_RANGE = 0.03;

UserStudy::UserStudy(Interaction *interaction)
    : Widget()
    , FloatOMeterListener()
    , CalculatorListener()
    , CardListener()
{
    Matrix matrix;
    Vec3 pos;
    Vec4 color;

    matrix.makeTranslate(4, 5, 0);

    _dial = new Dial(interaction);
    _dial->getNode()->setMatrix(matrix);
    _dial->addCardListener(this);
    _dial->setVisible(false);

    matrix.makeTranslate(6.5, 5, 0);

    _done = new Button(interaction);
    _done->setText("Done");
    _done->setVisible(false);
    _done->getNode()->setMatrix(matrix);
    _done->addCardListener(this);

    _msg = new cui::Message(cui::Message::CENTER);
    pos.set(4, 2, 0);
    _msg->setPosition(pos);
    _msg->setSize(0.5);

    _errorMsg = new cui::Message(cui::Message::CENTER);
    pos.set(4, 1, 0);
    _errorMsg->setPosition(pos);
    _errorMsg->setSize(0.5);

    _topMark = new cui::Message(cui::Message::CENTER);
    pos.set(7.5, TOP, 0);
    _topMark->setPosition(pos);
    _topMark->setSize(0.5);

    _bottomMark = new cui::Message(cui::Message::CENTER);
    pos.set(7.5, BOTTOM, 0);
    _bottomMark->setPosition(pos);
    _bottomMark->setSize(0.5);

    _moveableMark = new cui::Message(cui::Message::CENTER);
    pos.set(7.5, BOTTOM, 0);
    _moveableMark->setPosition(pos);
    _moveableMark->setSize(0.5);

    _rangeMark = new cui::Message(cui::Message::CENTER);
    pos.set(7.5, TOP, 0);
    _rangeMark->setPosition(pos);
    _rangeMark->setSize(0.5);

    _node->addChild(_dial->getNode());
    _node->addChild(_done->getNode());
    _node->addChild(_msg->getNode());
    _node->addChild(_errorMsg->getNode());
    _node->addChild(_topMark->getNode());
    _node->addChild(_bottomMark->getNode());
    _node->addChild(_moveableMark->getNode());
    _node->addChild(_rangeMark->getNode());

    osgObj->addFrontChild(getNode());

    _values = 0;
    _rangeMins = 0;
    _rangeMaxs = 0;
    _logFile = 0;

    _method = Dial::NO;
    _methodText = "normal dial";
    _numberMode = SMALL_NUMBERS;
    _numberModeText = "small numbers";

    _nextValue = false;
    _inputStarted = true;
}

UserStudy::~UserStudy()
{
}

void UserStudy::update()
{
    double tmp, posY;
    Vec3 pos;

    if (_numberMode == SEARCH)
    {
        tmp = fabs(_dial->getValue() - _valueToSet);

        if (tmp >= _maxDiff)
            tmp = BOTTOM;
        else
            posY = (1.0 - (tmp / _maxDiff)) * (TOP - BOTTOM) + BOTTOM;

        if (posY < BOTTOM)
            posY = BOTTOM;

        pos.set(7.5, posY, 0);
        _moveableMark->setPosition(pos);

        _topMark->update();
        _bottomMark->update();
        _moveableMark->update();
        _rangeMark->update();
    }

    _msg->update();
    _errorMsg->update();
}

void UserStudy::setVisible(bool flag)
{
    Widget::setVisible(flag);

    _done->setVisible(flag);
    _dial->setVisible(flag);
}

void UserStudy::setMethod(Dial::AdvancedInput method)
{
    _method = method;

    if (_method == Dial::CALCULATOR)
    {
        _dial->setAdvancedInput(Dial::CALCULATOR);
        _dial->setText("Calc");
        _dial->setInteger(false);
        _methodText = "calculator";
    }
    else if (_method == Dial::FLOATOMETER)
    {
        _dial->setAdvancedInput(Dial::FLOATOMETER);
        _dial->setText("FloatO");
        _dial->setInteger(false);
        _methodText = "floatOMeter";
    }
    else if (_method == Dial::NO)
    {
        _dial->setAdvancedInput(Dial::NO);
        _dial->setText("Dial");
        _dial->setKnobRange(100.0);
        _dial->setInteger(true);
        _methodText = "normal dial";
    }

    if (_logFile)
    {
        sprintf(_logBuf, "method set: %s", _methodText);
        _logFile->addLog(_logBuf);
    }
}

void UserStudy::setNumberMode(NumberMode numberMode)
{
    _numberMode = numberMode;

    if (_numberMode == SMALL_NUMBERS)
        _numberModeText = "small numbers";
    else if (_numberMode == LARGE_NUMBERS)
        _numberModeText = "large numbers";
    else if (_numberMode == ADJUST_NUMBERS)
        _numberModeText = "adjust numbers";
    else if (_numberMode == SEARCH)
        _numberModeText = "search";

    if (_numberMode == SEARCH)
    {
        _bottomMark->setText("-----");
        _topMark->setText("-----");
        _rangeMark->setText("-----");
        _moveableMark->setText("xxx");
    }
    else
    {
        _bottomMark->setText("");
        _topMark->setText("");
        _rangeMark->setText("");
        _moveableMark->setText("");
    }

    if (_logFile)
    {
        sprintf(_logBuf, "number mode set: %s", _numberModeText);
        _logFile->addLog(_logBuf);
    }
}

void UserStudy::setLogFile(LogFile *lf)
{
    Widget::setLogFile(lf);

    if (lf == 0)
        std::cerr << "no logfile" << endl;
    else
        std::cerr << "logfile fine" << endl;

    _dial->setLogFile(_logFile);
    _done->setLogFile(_logFile);

    if (_logFile)
        _logFile->addLog("logfile added to user study");
}

void UserStudy::restart()
{
    char buf[64];
    int num = 63;
    int min;
    int i;
    string dollarG, path;
    double diff, tmp;

    dollarG = getenv("G");
    path = dollarG + "/src/vox-cave/userstudy/";

    std::cerr << "UserStudy::restart" << endl;

    if (_numberMode == SEARCH)
    {
        path = path + SEARCH_FILENAME;

        _file = fopen(path.c_str(), "r");

        assert(_file);

        assert(fgets(buf, num, _file));

        sscanf(buf, "%d", &_numberOfValues);

        fgets(buf, num, _file);

        if (_values != 0)
            delete[] _values;
        _values = new double[_numberOfValues];

        if (_rangeMins != 0)
            delete[] _rangeMins;
        _rangeMins = new double[_numberOfValues];

        if (_rangeMaxs != 0)
            delete[] _rangeMaxs;
        _rangeMaxs = new double[_numberOfValues];

        srand(time(NULL));

        for (i = 0; i < _numberOfValues; i++)
        {
            assert(fgets(buf, num, _file));
            sscanf(buf, "%lf", &_rangeMins[i]);

            assert(fgets(buf, num, _file));
            sscanf(buf, "%lf", &_rangeMaxs[i]);

            diff = _rangeMaxs[i] - _rangeMins[i];

            tmp = rand() / float(RAND_MAX);

            _values[i] = _rangeMins[i] + tmp * diff;
        }
    }
    else
    {
        if (_numberMode == SMALL_NUMBERS)
        {
            path = path + SMALL_NUMBERS_FILENAME;

            _file = fopen(path.c_str(), "r");

            assert(_file);
        }
        else if (_numberMode == LARGE_NUMBERS)
        {
            path = path + LARGE_NUMBERS_FILENAME;

            _file = fopen(path.c_str(), "r");

            assert(_file);
        }
        else if (_numberMode == ADJUST_NUMBERS)
        {
            path = path + ADJUST_NUMBERS_FILENAME;

            _file = fopen(path.c_str(), "r");

            assert(_file);
        }

        assert(fgets(buf, num, _file));

        sscanf(buf, "%d", &_numberOfValues);

        fgets(buf, num, _file);

        if (_values != 0)
            delete[] _values;

        _values = new double[_numberOfValues];

        for (i = 0; i < _numberOfValues; i++)
        {
            assert(fgets(buf, num, _file));
            sscanf(buf, "%lf", &_values[i]);
        }

        if (_numberMode != ADJUST_NUMBERS)
            shakeValues();
    }

    _counter = 0;

    if (_logFile)
    {
        sprintf(_logBuf, "new pass started: number mode = %s, method = %s quantity = %d", _numberModeText, _methodText, _numberOfValues);
        _logFile->addLog(_logBuf);
    }

    _errorMsg->setText("new pass started", 3.0);

    _dial->setValue(0.0);
    setNextValue();
}

void UserStudy::calculatorOpened()
{
    if (_logFile)
    {
        if (_nextValue)
            _logFile->addLog("calculator opened - input started");
        else
            _logFile->addLog("calculator opened");
    }
}

void UserStudy::calculatorClosed()
{
    if (_logFile)
        _logFile->addLog("calculator closed");
}

void UserStudy::floatOMeterOpened()
{
    if (_logFile)
    {
        if (_nextValue)
            _logFile->addLog("floatOMeter opened - input started");
        else
            _logFile->addLog("floatOMeter opened");
    }
}

void UserStudy::floatOMeterClosed()
{
    if (_logFile)
        _logFile->addLog("floatOMeter closed");
}

void UserStudy::setNextValue()
{
    char buf[128];
    int i, num, min, tmp;
    bool flag;
    Vec3 pos;

    if (_counter == _numberOfValues)
    {
        if (_logFile)
        {
            sprintf(_logBuf, "pass finished: number mode = %s, method = %s quantity = %d", _numberModeText, _methodText, _numberOfValues);
            _logFile->addLog(_logBuf);
        }
        _dial->setValue(0.0);

        _msg->setText("");
        _errorMsg->setText("You're done for this pass");

        return;
    }

    _valueToSet = _values[_counter];

    num = sprintf(buf, "%lf", _valueToSet);

    for (i = 0, tmp = 0, min = 0, flag = false; i < num; i++)
    {
        if (flag)
        {
            tmp++;
            if (buf[i] != '0')
                min = tmp;
        }
        else if (buf[i] == '.')
            flag = true;
    }

    sprintf(buf, "%.*lf", min, _valueToSet);

    if (_numberMode == SEARCH)
    {
        _rangeMin = _rangeMins[_counter];
        _dial->setMin(_rangeMin);
        _rangeMax = _rangeMaxs[_counter];
        _dial->setMax(_rangeMax);

        if (fabs(_rangeMax - _valueToSet) > fabs(_rangeMin - _valueToSet))
            _maxDiff = fabs(_rangeMax - _valueToSet);
        else
            _maxDiff = fabs(_rangeMin - _valueToSet);

        pos.set(7.5, (1.0 - ACCEPT_SEARCH_RANGE) * (TOP - BOTTOM) + BOTTOM, 0);
        _rangeMark->setPosition(pos);
    }

    if (_numberMode != ADJUST_NUMBERS)
        _dial->setValue(0.0);

    _counter++;
    _nextValue = true;
    _inputStarted = false;

    if (_numberMode != SEARCH)
        _msg->setText(buf);
    else
    {
        sprintf(buf, "find the interesting range");
        _msg->setText(buf);
    }

    if (_logFile)
    {
        if (_numberMode == SEARCH)
            sprintf(_logBuf, "next serach started: value\t%lf range min\t%lf range max\t%lf", _valueToSet, _rangeMin, _rangeMax);
        else
            sprintf(_logBuf, "next value displayed: %lf", _valueToSet);

        _logFile->addLog(_logBuf);
    }
}

bool UserStudy::cardButtonEvent(Card *card, int button, int state)
{
    double eps;

    if ((card == _done) && (state == 1) && (_counter <= _numberOfValues))
    {
        if (_numberMode == SEARCH)
            eps = ACCEPT_SEARCH_RANGE * _maxDiff;
        else
            eps = EPSILON;

        if (fabs(_dial->getValue() - _valueToSet) < eps)
        {
            if (_logFile)
                _logFile->addLog("input finished correctly");

            _errorMsg->setText("correct input", 1.0);
            setNextValue();
        }
        else
        {
            _nextValue = false;
            if (_logFile)
                _logFile->addLog("input was wrong");

            if (_numberMode == SEARCH)
                _errorMsg->setText("ERROR: interesting range not found", 1.0);
            else
                _errorMsg->setText("ERROR: wrong number", 1.0);
        }

        return true;
    }

    if (card == _dial)
    {
        if ((button == 0) && (state == 1) && (!_inputStarted) && (_method == Dial::NO))
        {
            if (_logFile)
                _logFile->addLog("dial - input started");
            _inputStarted = true;
        }

        return true;
    }

    return false;
}

void UserStudy::generateNumbers(NumberMode numberMode, int quantity, int seed)
{
    int i;
    int small, large, shift, sign;
    double value;
    string dollarG, path;

    srand(seed);

    dollarG = getenv("G");
    path = dollarG + "/src/vox-cave/userstudy/";

    if (numberMode == SMALL_NUMBERS)
    {
        path = path + SMALL_NUMBERS_FILENAME;

        _file = fopen(path.c_str(), "w");

        assert(_file);

        fprintf(_file, "%d\n\n", quantity);

        for (i = 0; i < quantity; i++)
        {
            value = rand() / (float)RAND_MAX;
            small = (int)(value * 100);
            fprintf(_file, "%d\n", small);
        }

        if (_logFile)
            _logFile->addLog("small numbers generated");

        fclose(_file);
    }
    else if (numberMode == LARGE_NUMBERS)
    {
        path = path + LARGE_NUMBERS_FILENAME;

        _file = fopen(path.c_str(), "w");

        assert(_file);

        fprintf(_file, "%d\n\n", quantity);

        for (i = 0; i < quantity; i++)
        {
            value = rand() / (double)RAND_MAX;
            large = (int)(value * pow(10.0, 5));
            value = rand() / (double)RAND_MAX;
            shift = (int)(value * -7) + 2;
            value = (double)large * pow(10.0, shift);
            sign = rand();
            if (rand() % 2)
                value *= -1.0;

            fprintf(_file, "%.*lf\n", abs(shift) - 1, value);
        }

        if (_logFile)
            _logFile->addLog("large numbers generated");

        fclose(_file);
    }
}

void UserStudy::shakeValues()
{
    int i, a, b;
    double tmp;

    srand(time(NULL));

    for (i = 0; i < 1000; i++)
    {
        a = int((rand() / (float)RAND_MAX) * _numberOfValues);
        b = int((rand() / (float)RAND_MAX) * _numberOfValues);

        if ((a < 0) || (a > 9))
        {
            std::cerr << "should not be" << endl;
            a = 9;
        }
        if ((b < 0) || (b > 9))
        {
            std::cerr << "should not be" << endl;
            b = 9;
        }

        tmp = _values[a];
        _values[a] = _values[b];
        _values[b] = tmp;
    }
}
