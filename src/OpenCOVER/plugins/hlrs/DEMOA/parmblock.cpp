/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// parmblock.cpp
// parmblock class definition
//
////////////////////////////////////////////////////////////////////////

#include "parmblock.h"

#ifdef WIN32
#pragma warning(disable : 4786)
#endif

#include <cctype>
#include <cstdlib>

#include <algorithm>
#include <iostream>

#include "string_parse.h"

void parmblock::print_values() const
{
    typedef values_t::const_iterator cit;
    for (cit p = values.begin(); p != values.end(); ++p)
    {
        std::cout << p->first << ": " << p->second << std::endl;
    }
}

string_parse &parmblock::read(string_parse &data)
{
    if (!data.readword(key_val))
    {
        return data;
    }

    data.readquoted(name_val, '"');
    if (data.eos())
        return data;

    string_parse buf;
    data.readquoted(buf, '{', '}');
    if (data.eos())
        return data;

    typedef int (*isspace_t)(int ch);
    isspace_t isspace = &std::isspace;
    buf.erase(std::remove_if(buf.begin(), buf.end(), isspace), buf.end());

    int lo = 0;
    values.clear();
    while (true)
    {
        unsigned int hi = buf.find('=', lo);
        if (hi > buf.size() - 1)
        {
            break;
        }
        const std::string &key = buf.substr(lo, hi - lo);
        lo = hi + 1;
        hi = buf.find(';', lo);
        if (hi > buf.size() - 1)
        {
            break;
        }
        values[key] = buf.substr(lo, hi - lo);
        lo = hi + 1;
    }
    return data;
}

template <typename T>
static T clone(const T &t)
{
    return t;
}

template <typename T, typename F>
T parmblock::get_value(const std::string &valname, const F &f, const T &d) const
{
    const values_t::const_iterator p = values.find(valname);
    if (p == values.end())
    {
        std::cerr << "Warning! Element '"
                  << name_val
                  << "': Key '"
                  << valname
                  << "' not found. Continuing using default."
                  << std::endl;
        return d;
    }
    return f(p->second.c_str());
}

int parmblock::get_ivalue(const std::string &valname) const
{
    return get_value(valname, &std::atoi, 142857143);
}

double parmblock::get_dvalue(const std::string &valname) const
{
    return get_value(valname, &std::atof, 1.4285714e-286);
}

std::string parmblock::getstring(const std::string &valname) const
{
    return get_value(valname, &::clone<std::string>, std::string());
}

int parmblock::getvalue(const std::string &valname, int dflt) const
{
    return get_value(valname, &std::atoi, dflt);
}

float parmblock::getvalue(const std::string &valname, float dflt) const
{
    return get_value(valname, &std::atof, dflt);
}

double parmblock::getvalue(const std::string &valname, double dflt) const
{
    return get_value(valname, &std::atof, dflt);
}

template <typename T>
void parmblock::get_vector(T v[], const int dim, const std::string &valname,
                           const T dflt[]) const
{
    const std::string &all = getstring(valname);
    if (all.empty())
    {
        std::cerr << "Warning! Using default.\n";
        std::copy(dflt, dflt + dim, v);
        return;
    }
    getvector(v, dim, valname);
}

void parmblock::getvector(int v[], const int dim, const std::string &valname,
                          const int dflt[]) const
{
    parmblock::get_vector(v, dim, valname, dflt);
}

void parmblock::getvector(float v[], const int dim, const std::string &valname,
                          const float dflt[]) const
{
    parmblock::get_vector(v, dim, valname, dflt);
}

void parmblock::getvector(double v[], const int dim, const std::string &valname,
                          const double dflt[]) const
{
    parmblock::get_vector(v, dim, valname, dflt);
}

template <typename T>
void parmblock::get_vector(T v[], const int dim, const std::string &valname,
                           const T &dflt) const
{
    const std::string &all = getstring(valname);
    if (all.empty())
    {
        std::cerr << "Warning! Using default.\n";
        std::fill(v, v + dim, dflt);
        return;
    }
    getvector(v, dim, valname);
}

void parmblock::getvector(int v[], const int dim, const std::string &valname,
                          int dflt) const
{
    get_vector(v, dim, valname, dflt);
}

void parmblock::getvector(float v[], const int dim, const std::string &valname,
                          float dflt) const
{
    get_vector(v, dim, valname, dflt);
}

void parmblock::getvector(double v[], const int dim, const std::string &valname,
                          double dflt) const
{
    get_vector(v, dim, valname, dflt);
}

template <typename T, typename F>
void parmblock::get_vector(T v[], const int dim, const std::string &valname,
                           const F &f) const
{
    const std::string &all = getstring(valname);
    if (all.empty())
    {
        return;
    }

    int lo = 0;
    std::string single;
    for (int ii = 0; ii < dim - 1; ++ii)
    {
        const int hi = all.find(',', lo);
        if (hi > static_cast<int>(all.size()) - 1)
        {
            std::cerr << "Warning! Element '"
                      << name_val
                      << "': Key '"
                      << valname
                      << "': Less items than expected. Continuing."
                      << std::endl;
            break;
        }
        single = all.substr(lo, hi - lo);
        v[ii] = f(single.c_str());
        lo = hi + 1;
    }
    single = all.substr(lo, all.size() - lo);
    v[dim - 1] = f(single.c_str());
}

void parmblock::getvector(int v[], const int dim,
                          const std::string &valname) const
{
    parmblock::get_vector(v, dim, valname, &std::atoi);
}

void parmblock::getvector(float v[], const int dim,
                          const std::string &valname) const
{
    parmblock::get_vector(v, dim, valname, &std::atof);
}

void parmblock::getvector(double v[], const int dim,
                          const std::string &valname) const
{
    parmblock::get_vector(v, dim, valname, &std::atof);
}
