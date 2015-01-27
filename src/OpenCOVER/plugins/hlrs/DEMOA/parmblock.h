/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// parmblock.h
// parmblock class declaration
//
////////////////////////////////////////////////////////////////////////

#ifndef __PARMBLOCK_H
#define __PARMBLOCK_H

#include <map>
#include <string>

#include "common_include.h"

class string_parse;

class parmblock
{
    friend class ifstream_parse;

public:
    inline void set_keyword(const std::string &key);
    inline std::string keyword() const;
    inline void set_name(const std::string &name);
    inline std::string name() const;

    void print_values() const;
    string_parse &read(string_parse &data);
    std::string getstring(const std::string &valname) const;
    int getvalue(const std::string &valname, int dflt) const;
    float getvalue(const std::string &valname, float dflt) const;
    double getvalue(const std::string &valname, double dflt) const;
    void getvector(int v[], const int dim, const std::string &valname) const;
    void getvector(int v[], const int dim, const std::string &valname,
                   const int dflt[]) const;
    void getvector(int v[], const int dim, const std::string &valname,
                   int dflt) const;
    void getvector(float v[], const int dim, const std::string &valname) const;
    void getvector(float v[], const int dim, const std::string &valname,
                   const float dflt[]) const;
    void getvector(float v[], const int dim, const std::string &valname,
                   float dflt) const;
    void getvector(double v[], const int dim, const std::string &valname) const;
    void getvector(double v[], const int dim, const std::string &valname,
                   const double dflt[]) const;
    void getvector(double v[], const int dim, const std::string &valname,
                   double dflt) const;

    template <typename T>
    inline void getvector(T &v, const std::string &valname,
                          const double dflt[]) const;

private:
    template <typename T, typename F>
    T get_value(const std::string &valname, const F &f, const T &d) const;

    int get_ivalue(const std::string &valname) const;
    double get_dvalue(const std::string &valname) const;

    template <typename T>
    void get_vector(T v[], const int dim, const std::string &valname,
                    const T dflt[]) const;

    template <typename T>
    void get_vector(T v[], const int dim, const std::string &valname,
                    const T &dflt) const;

    template <typename T, typename F>
    void get_vector(T v[], const int dim, const std::string &valname,
                    const F &f) const;

private:
    std::string key_val;
    std::string name_val;

    typedef std::map<std::string, std::string> values_t;
    values_t values;
};

inline void parmblock::set_keyword(const std::string &key)
{
    key_val = key;
}

inline std::string parmblock::keyword() const
{
    return key_val;
}

inline void parmblock::set_name(const std::string &name)
{
    name_val = name;
}

inline std::string parmblock::name() const
{
    return name_val;
}

template <typename T>
inline void parmblock::getvector(T &v, const std::string &valname,
                                 const double dflt[]) const
{
    double vector[3];
    getvector(vector, 3, valname, dflt);

    v[0] = vector[0];
    v[1] = vector[1];
    v[2] = vector[2];
}

#endif // __PARMBLOCK_H
