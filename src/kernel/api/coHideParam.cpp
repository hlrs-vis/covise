/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include "coHideParam.h"

#include "coFloatParam.h"
#include "coIntScalarParam.h"
#include "coChoiceParam.h"
#include "coBooleanParam.h"
#include "coFloatVectorParam.h"
#include "coIntVectorParam.h"
#include "coFloatSliderParam.h"
#include "coIntSliderParam.h"
#include "coColorParam.h"

#include <appl/ApplInterface.h>
//#include <util/coString.h>

using namespace covise;

coHideParam::coHideParam(coUifPara *param)
{
    param_ = param;
}

coHideParam::~coHideParam()
{
}

void
coHideParam::load(const char *value)
{
    std::istringstream input(value);
    /*
      int len = strlen(value)+1;
      coString pname(len);
   */
    string pname;
    input >> pname;
    if (pname != param_->getName())
    {
        return;
    }
    hidden_ = true;
    if (strcmp(param_->getTypeString(), "INTSCA") == 0
        || strcmp(param_->getTypeString(), "CHOICE") == 0
        || strcmp(param_->getTypeString(), "BOOL") == 0
        || strcmp(param_->getTypeString(), "INTSLI") == 0)
    {
        int num;
        if (!(input >> num))
        {
            Covise::sendWarning("HideParam::load error when reading int for %s", param_->getName());
            hidden_ = false;
            return;
        }
        inumbers_.push_back(num);
    }
    else if (strcmp(param_->getTypeString(), "FLOSCA") == 0
             || strcmp(param_->getTypeString(), "FLOSLI") == 0)
    {
        float num;
        if (!(input >> num))
        {
            Covise::sendWarning("HideParam::load error when reading float for %s", param_->getName());
            hidden_ = false;
            return;
        }
        fnumbers_.push_back(num);
    }
    else if (strcmp(param_->getTypeString(), "INTVEC") == 0)
    {
        int num;
        while (input >> num)
        {
            inumbers_.push_back(num);
        }
    }
    else if (strcmp(param_->getTypeString(), "FLOVEC") == 0)
    {
        float num;
        while (input >> num)
        {
            fnumbers_.push_back(num);
        }
    }
}

void
coHideParam::reset()
{
    hidden_ = false;
    fnumbers_.clear();
    inumbers_.clear();
}

float
coHideParam::getFValue()
{
    float ret = 0.0;
    if (strcmp(param_->getTypeString(), "FLOSCA") == 0
        || strcmp(param_->getTypeString(), "FLOSLI") == 0)
    {
        if (hidden_)
        {
            if (fnumbers_.size() > 0)
            {
                ret = fnumbers_[0];
            }
            else
            {
                cerr << fnumbers_[0] << endl;
                ret = 10.0;
            }
        }
        else if (strcmp(param_->getTypeString(), "FLOSCA") == 0)
        {
            ret = ((coFloatParam *)(param_))->getValue();
        }
        else if (strcmp(param_->getTypeString(), "FLOSLI") == 0)
        {
            ret = ((coFloatSliderParam *)(param_))->getValue();
        }
    }
    else
    {
        Covise::sendWarning("HideParam::getFValue(): bug for parameter %s", param_->getName());
    }
    return ret;
}

int
coHideParam::getIValue()
{
    int ret = 0;
    if (strcmp(param_->getTypeString(), "INTSCA") == 0
        || strcmp(param_->getTypeString(), "CHOICE") == 0
        || strcmp(param_->getTypeString(), "BOOL") == 0
        || strcmp(param_->getTypeString(), "INTSLI") == 0)
    {
        if (hidden_)
        {
            ret = inumbers_[0];
        }
        else if (strcmp(param_->getTypeString(), "INTSCA") == 0)
        {
            ret = ((coIntScalarParam *)(param_))->getValue();
        }
        else if (strcmp(param_->getTypeString(), "CHOICE") == 0)
        {
            ret = ((coChoiceParam *)(param_))->getValue();
        }
        else if (strcmp(param_->getTypeString(), "BOOL") == 0)
        {
            ret = ((coBooleanParam *)(param_))->getValue();
        }
        else if (strcmp(param_->getTypeString(), "INTSLI") == 0)
        {
            ret = ((coIntSliderParam *)(param_))->getValue();
        }
    }
    else
    {
        Covise::sendWarning("HideParam::getIValue(): bug for parameter %s", param_->getName());
    }
    return ret;
}

float
coHideParam::getFValue(int index)
{
    float ret = 0.0;
    if (strcmp(param_->getTypeString(), "FLOVEC") == 0)
    {
        if (hidden_)
        {
            ret = fnumbers_[index];
        }
        else
        {
            ret = ((coFloatVectorParam *)(param_))->getValue(index);
        }
    }
    else
    {
        Covise::sendWarning("HideParam::getFValue(int): bug for parameter %s", param_->getName());
    }
    return ret;
}

int
coHideParam::getIValue(int index)
{
    int ret = 0;
    if (strcmp(param_->getTypeString(), "INTVEC") == 0)
    {
        if (hidden_)
        {
            ret = inumbers_[index];
        }
        else
        {
            ret = ((coIntVectorParam *)(param_))->getValue(index);
        }
    }
    else
    {
        Covise::sendWarning("HideParam::getIValue(int): bug for parameter %s", param_->getName());
    }
    return ret;
}

void
coHideParam::getFValue(float &data0, float &data1, float &data2)
{
    if (strcmp(param_->getTypeString(), "FLOVEC") == 0)
    {
        if (hidden_)
        {
            if (fnumbers_.size() >= 3)
            {
                data0 = fnumbers_[0];
                data1 = fnumbers_[1];
                data2 = fnumbers_[2];
            }
            else
            {
                cerr << fnumbers_.size() << endl;
                data0 = -1;
                data1 = 0.0;
                data2 = 0.0;
            }
        }
        else
        {
            ((coFloatVectorParam *)(param_))->getValue(data0, data1, data2);
        }
    }
    else
    {
        Covise::sendWarning("HideParam::getFValue(3*float&): bug for parameter %s", param_->getName());
    }
}
