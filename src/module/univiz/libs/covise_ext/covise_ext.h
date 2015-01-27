/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// helpers for Covise
// filip sadlo
// cgl eth 2007

#ifndef _COVISE_EXT_H_
#define _COVISE_EXT_H_

#include "api/coSimpleModule.h"
#include "do/coDoUnstructuredGrid.h"
#include <vector>

using namespace std;

#ifdef COVISE
using namespace covise;
#endif

// Parameters: for forcing min / max ------------------------------------------

class myParam
{
public:
    virtual ~myParam()
    {
    }
    virtual void adaptValue(void)
    {
    }
};

extern std::vector<myParam *> myParams;

class coIntScalarParamExt : myParam
{
private:
    int min, max;

public:
    coIntScalarParam *p;

    coIntScalarParamExt()
    {
        myParams.push_back(this);
    }
    ~coIntScalarParamExt()
    {
        for (int i = 0; i < (int)myParams.size(); i++)
        {
            if ((myParams.size() > 0) && (myParams[i] == this))
            {
                myParams[i] = myParams[myParams.size() - 1];
                break;
            }
        }
    }

    void setValue(int v)
    {
        p->setValue(v);
    }
    int getValue(void)
    {
        return p->getValue();
    }

    void enable(void)
    {
        p->enable();
    }
    void disable(void)
    {
        p->disable();
    }

    void show(void)
    {
        p->show();
    }
    void hide(void)
    {
        p->hide();
    }

    void setMin(int v)
    {
        min = v;
    }
    void setMax(int v)
    {
        max = v;
    }

    void setValue(int val, int min, int max)
    {
        setValue(val);
        setMin(min);
        setMax(max);
    }

    virtual void adaptValue(void)
    {
        int val = p->getValue();
        if (val < min)
            p->setValue(min);
        if (val > max)
            p->setValue(max);
    }
};

class coFloatParamExt : myParam
{
private:
    float min, max;

public:
    coFloatParam *p;

    coFloatParamExt()
    {
        myParams.push_back(this);
    }
    ~coFloatParamExt()
    {
        for (int i = 0; i < (int)myParams.size(); i++)
        {
            if ((myParams.size() > 0) && (myParams[i] == this))
            {
                myParams[i] = myParams[myParams.size() - 1];
                break;
            }
        }
    }

    void setValue(float v)
    {
        p->setValue(v);
    }
    float getValue(void)
    {
        return p->getValue();
    }

    void enable(void)
    {
        p->enable();
    }
    void disable(void)
    {
        p->disable();
    }

    void show(void)
    {
        p->show();
    }
    void hide(void)
    {
        p->hide();
    }

    void setMin(float v)
    {
        min = v;
    }
    void setMax(float v)
    {
        max = v;
    }

    void setValue(float val, float min, float max)
    {
        setValue(val);
        setMin(min);
        setMax(max);
    }

    virtual void adaptValue(void)
    {
        float val = p->getValue();
        if (val < min)
            p->setValue(min);
        if (val > max)
            p->setValue(max);
    }
};

void adaptMyParams(void);

// Basic ----------------------------------------------------------------------

coDoUnstructuredGrid *generateUniformUSG(const char *name,
                                         float originX, float originY, float originZ,
                                         int cellsX, int cellsY, int cellsZ,
                                         float cellSize);

#endif // _COVISE_EXT_H_
