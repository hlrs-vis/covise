/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +                                                                         +
// +  coColormapParam Parameter handling class                                +
// +                                                                         +
// +                           Andreas Werner                                +
// +         (C)  Computing Center University of Stuttgart                   +
// +                          Allmandring 30a                                +
// +                          70550 Stuttgart                                +
// + Date:  19.07.99                                                         +
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>
#include "coColormapParam.h"
#include <appl/ApplInterface.h>

using namespace covise;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Static data
const char *coColormapParam::s_type = "COLORMAP";
coUifPara::Typeinfo coColormapParam::s_paraType = coUifPara::numericType("COLORMAP");

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Constructor

coColormapParam::coColormapParam(const char *name, const char *desc)
    : coUifPara(name, desc)
    , valString(NULL)
{
    d_len = 3;
    d_min = 0.0;
    d_max = 1.0;

    d_rgbax = new float[d_len * 5];
    d_rgbax[0] = 0.;
    d_rgbax[1] = 0.;
    d_rgbax[2] = 1.;
    d_rgbax[3] = 1.;
    d_rgbax[4] = 0.;

    d_rgbax[5] = 1.;
    d_rgbax[6] = 0.;
    d_rgbax[7] = 0.;
    d_rgbax[8] = 1.;
    d_rgbax[9] = 0.5;

    d_rgbax[10] = 1.;
    d_rgbax[11] = 1.;
    d_rgbax[12] = 0.;
    d_rgbax[13] = 1.;
    d_rgbax[14] = 1.;

    d_colormapType = RGBAX;

    d_lenData = 0;
    d_data = NULL;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Destructor : virtual in case we derive objects

coColormapParam::~coColormapParam()
{
    delete[] d_rgbax;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// Check the type

int coColormapParam::isOfType(coUifPara::Typeinfo type)
{
    return (type == s_paraType);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get my type

coUifPara::Typeinfo coColormapParam::getType()
{
    return s_paraType;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// handle parameter changes: called by paramCB

int coColormapParam::paramChange()
{
    delete[] d_rgbax;
    d_rgbax = NULL;

    int retVal = Covise::get_reply_colormap(&d_min, &d_max, &d_len, &d_colormapType);
    if (retVal && d_colormapType == RGBAX)
    {
        d_rgbax = new float[d_len * 5];
        for (int i = 0; i < d_len; i++)
        {
            retVal = Covise::get_reply_colormap(i, &d_rgbax[i * 5], &d_rgbax[i * 5 + 1], &d_rgbax[i * 5 + 2], &d_rgbax[i * 5 + 3], &d_rgbax[i * 5 + 4]);
            if (!retVal)
            {
                fprintf(stderr, "coColormapParam::get_reply_colormap: retVal=%d\n", retVal);
                break;
            }
        }
        if (!retVal)
        {
            delete[] d_rgbax;
            d_rgbax = NULL;
        }
    }

    if (!retVal)
    {
        d_len = 0;
        d_min = 0.0;
        d_max = 1.0;
        d_colormapType = RGBAX;
    }
    return retVal;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// give dafault values to Covise -> automatically called !

void coColormapParam::initialize()
{
    // we must allocate this due to Covise Appl-Lib implementation bugs
    d_defString = new char[256];
    snprintf(d_defString, 256, "0.0 1.0  RGBAX  3  0.0 0.0 1.0 1.0 0.0  1.0 0.0 0.0 1.0 0.5  1.0 1.0 0.0 1.0 1.0");

    Covise::add_port(PARIN, d_name, "Colormap", d_desc);
    Covise::set_port_default(d_name, d_defString);
    d_init = 1;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// print this to a stream

void coColormapParam::print(ostream &str) const
{
    coUifPara::print(str);
    fprintf(stderr, "colormap: print\n");
#if 0
   cerr << "coColormapParam : " << d_value << endl;
#endif
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value: if called after init() : update on map

int coColormapParam::setValue(int len, float min, float max, const float *rgbax)
{
    /// If we have been initialized, update the map
    if (d_init)
    {
        Covise::sendWarning("Cannot update colormap");
        return 0;
    }
    else
    {
        d_len = len;
        d_min = min;
        d_max = max;
        d_rgbax = new float[len * 5];
        d_colormapType = RGBAX;
        memcpy(d_rgbax, rgbax, sizeof(float) * len * 5);

        return 1;
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value: if called after init() : update on map

int coColormapParam::setData(int np, const float *data)
{

    /// If we have been initialized, update the map
    if (d_init)
    {
        if (d_colormapType == 0)
        {
            d_lenData = np;
            d_data = new float[np];
            memcpy(d_data, data, sizeof(float) * np);

            ostringstream os;

            os << "RGBAX"
               << " " << d_len;
            for (int i = 0; i < d_len; i++)
            {
                for (int j = 0; j < 5; j++)
                    os << " " << d_rgbax[i * 5 + j];
            }

            os << " HISTO"
               << " " << d_lenData;
            for (int i = 0; i < np; i++)
                os << " " << d_data[i];

            return Covise::update_colormap_param(d_name, d_min, d_max, os.str());
        }
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value: if called after init() : update on map

int coColormapParam::setMinMax(float min, float max)
{
    /// If we have been initialized, update the map
    if (d_init)
    {
        d_min = min;
        d_max = max;

        if (d_colormapType == 0)
        {
            ostringstream os;
            os << "RGBAX"
               << " " << d_len;

            for (int i = 0; i < d_len; i++)
            {
                for (int j = 0; j < 5; j++)
                    os << " " << d_rgbax[i * 5 + j];
            }

            if (d_data)
            {
                os << " HISTO"
                   << " " << d_lenData;
                for (int i = 0; i < d_lenData; i++)
                    os << " " << d_data[i];
            }

            string buffer = os.str();
            return Covise::update_colormap_param(d_name, d_min, d_max, buffer);
        }
    }
    return 0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value

int coColormapParam::getValue(float *min, float *max, colormap_type *type, const float **rgbax) const
{
    if (min)
        *min = d_min;
    if (max)
        *max = d_max;
    if (type)
        *type = d_colormapType;
    if (rgbax)
        *rgbax = d_rgbax;

    return d_len;
}

/// ----- Prevent auto-generated functions by assert -------
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the type string of this parameter
const char *coColormapParam::getTypeString() const
{
    return s_type;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// get the value of this parameter as a string
const char *coColormapParam::getValString() const
{
    stringstream str;
    str << d_min << d_max << endl;
    str << "RGBAX" << endl;
    str << d_len << endl;
    for (int i = 0; i < d_len; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            str << d_rgbax[i * 5 + j];
        }
        str << endl;
    }

    string s = str.str();
    size_t len = strlen(s.c_str());
    if (valString)
    {
        delete[] valString;
    }
    valString = new char[len + 1];
    strcpy(valString, s.c_str());

    return valString;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// set the value of this parameter from a string
void coColormapParam::setValString(const char *str)
{
    size_t retval;
    retval = sscanf(str, "%f %f\n", &d_min, &d_max);
    if (retval != 2)
        std::cerr << "coColormapParam::setValString: sscanf failed" << std::endl;
    char typestr[100];
    retval = sscanf(str, "%99s\n", typestr);
    if (retval != 1)
        std::cerr << "coColormapParam::setValString: sscanf failed" << std::endl;
    retval = sscanf(str, "%i\n", &d_len);
    if (retval != 1)
        std::cerr << "coColormapParam::setValString: sscanf failed" << std::endl;
    if (d_rgbax)
    {
        delete[] d_rgbax;
    }
    d_rgbax = new float[5 * d_len];

    for (int i = 0; i < d_len; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            retval = sscanf(str, "%f", &d_rgbax[i * 5 + j]);
            if (retval != 1)
                std::cerr << "coColormapParam::setValString: sscanf failed" << std::endl;
        }
    }
}
