/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <list>

#include "DigitPanel.h"

// OSG:
#include <osg/Matrix>

// Virvo:
#include <virvo/vvtoolshed.h>

using namespace cui;
using namespace osg;

DigitPanel::DigitPanel(Interaction *interaction, Appearance appMode, Movability moveMode)
    : Panel()
{
    _interaction = interaction;
    _setPos = false;
    _topLeft.set(0, 0, 0);
    _moveMode = moveMode;
    _appearance = appMode;
    initObject();
    _panelGeode->setNodeMask(~1);
    (*_BGcolor)[0] = Vec4(0.0, 0.0, 0.0, 0.0);
}

DigitPanel::~DigitPanel()
{
}

void DigitPanel::initGraphics()
{
    _panelGeode = new Geode();
    _panelGeode->addDrawable(createGeometry());
}

void DigitPanel::addDigit(DigitLabel *digit, int col, int row)
{
    _node->addChild(digit->getNode());
    _digits.push_back(new PanelDigit(digit, col, row));

    layout();
}

void DigitPanel::removeDigit(DigitLabel *digit)
{
    std::list<PanelDigit *>::iterator iter;

    for (iter = _digits.begin(); iter != _digits.end(); ++iter)
    {
        if ((*iter)->_digit == digit)
        {
            _node->removeChild(digit->getNode());
            iter = _digits.erase(iter);
        }
    }

    layout();
}

bool DigitPanel::setDigitPos(DigitLabel *digit, int col, int row)
{
    std::list<PanelDigit *>::iterator iter;
    for (iter = _digits.begin(); iter != _digits.end(); ++iter)
    {
        if ((*iter)->_digit == digit)
        {
            (*iter)->_pos[0] = col;
            (*iter)->_pos[1] = row;
            layout();
            return true;
        }
    }

    return false;
}

void DigitPanel::layout()
{
    Matrix trans;
    float gridWidth, gridHeight;
    float offset[2];
    float x, y;
    int minCol, maxCol, minRow, maxRow;

    if (_digits.empty())
    {
        _numCols = _numRows = 0;
        _width = 2.0f * _borderSizeX;
        _height = 2.0f * _borderSizeY;
        updateGeometry();
    }
    else
    {
        gridWidth = _digits.front()->_digit->getWidth();
        gridHeight = _digits.front()->_digit->getHeight();

        minRow = maxRow = minCol = maxCol = 0;

        // calculate grid size:
        std::list<PanelDigit *>::const_iterator iter;
        for (iter = _digits.begin(); iter != _digits.end(); ++iter)
        {
            maxCol = ts_max(maxCol, (*iter)->_pos[0] + 1);
            maxRow = ts_max(maxRow, (*iter)->_pos[1] + 1);
            minCol = ts_min(minCol, (*iter)->_pos[0]);
            minRow = ts_min(minRow, (*iter)->_pos[1]);
        }

        _numRows = maxRow - minRow;
        _numCols = maxCol - minCol;

        _width = _numCols * gridWidth;
        _height = _numRows * gridHeight;

        _topLeft.set((minCol * gridWidth) - gridWidth / 2.0, -(minRow * gridHeight) + gridHeight / 2.0, 0.0);
        _setPos = true;

        updateGeometry();

        // place digits:
        offset[0] = 0.0;
        offset[1] = 0.0;
        for (iter = _digits.begin(); iter != _digits.end(); ++iter)
        {
            x = offset[0] + (*iter)->_pos[0] * gridWidth;
            y = offset[1] - (*iter)->_pos[1] * gridHeight;
            trans.makeTranslate(x, y, 2.0f * EPSILON_Z);
            (*iter)->_digit->setMatrix(trans);
        }
    }
}

PanelDigit::PanelDigit(DigitLabel *digit, int col, int row)
{
    _digit = digit;
    _pos[0] = col;
    _pos[1] = row;
}

PanelDigit::~PanelDigit()
{
}
