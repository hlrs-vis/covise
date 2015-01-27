/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GAALET_SYMBOL_H
#define __GAALET_SYMBOL_H

#include "string"
#include "sstream"

namespace gaalet
{

///symbex: symbolic expression
struct symbex
{
    symbex(const std::string &expr_ = "0")
        : expr(expr_)
    {
    }

    //template<typename A>
    //symbex(const A& expr_)
    symbex(const double &expr_)
    {
        std::stringstream as;
        as << expr_;
        expr = as.str();
    }

    //template<typename A>
    //symbex(const A& expr_)
    symbex(const char expr_[])
    {
        std::stringstream as;
        as << expr_;
        expr = as.str();
    }

    std::string expr;
};

template <>
struct null_element<symbex>
{
    static symbex value()
    {
        return symbex("0");
    }
};

} //end namespace gaalet

std::ostream &operator<<(std::ostream &os, const gaalet::symbex &s)
{
    os << s.expr;
    return os;
}

/// \brief Unary plus.
/// \ingroup symbex_ops
gaalet::symbex
operator+(const gaalet::symbex &l)
{
    return gaalet::symbex("(+" + l.expr + ")");
}

/// \brief Unary minus.
/// \ingroup symbex_ops
gaalet::symbex
operator-(const gaalet::symbex &l)
{
    return gaalet::symbex("(-" + l.expr + ")");
}

/// \brief Addition of two symbex operands.
/// \ingroup symbex_ops
gaalet::symbex
operator+(const gaalet::symbex &l, const gaalet::symbex &r)
{
    return gaalet::symbex("(" + l.expr + "+" + r.expr + ")");
}

/// \brief Subtraction of two symbex operands.
/// \ingroup symbex_ops
gaalet::symbex
operator-(const gaalet::symbex &l, const gaalet::symbex &r)
{
    return gaalet::symbex("(" + l.expr + "-" + r.expr + ")");
}

/// \brief Multiplication of two symbex operands.
/// \ingroup symbex_ops
gaalet::symbex
operator*(const gaalet::symbex &l, const gaalet::symbex &r)
{
    return gaalet::symbex(l.expr + "*" + r.expr);
}

/// \brief Multiplication of a double and a symbex operand.
/// \ingroup symbex_ops
gaalet::symbex
operator*(const double &l, const gaalet::symbex &r)
{
    if (l == 1.0)
    {
        return gaalet::symbex(r.expr);
    }
    else if (l == -1.0)
    {
        return gaalet::symbex("(-" + r.expr + ")");
    }
    else if (l == 0.0)
    {
        return gaalet::symbex("");
    }
    else
    {
        std::stringstream ls;
        ls << l;

        return gaalet::symbex(ls.str() + "*" + r.expr);
    }
}

/// \brief Multiplication of a symbex operand and a double.
/// \ingroup symbex_ops
gaalet::symbex
operator*(const gaalet::symbex &l, const double &r)
{
    if (r == 1.0)
    {
        return gaalet::symbex(l.expr);
    }
    else if (r == -1.0)
    {
        return gaalet::symbex("(-" + l.expr + ")");
    }
    else if (r == 0.0)
    {
        return gaalet::symbex("");
    }
    else
    {
        std::stringstream rs;
        rs << r;

        return gaalet::symbex(l.expr + "*" + rs.str());
    }
}

/// \brief Division of a symbex operand by another symbex operand.
/// \ingroup symbex_ops
gaalet::symbex
operator/(const gaalet::symbex &l, const gaalet::symbex &r)
{
    return gaalet::symbex(l.expr + "/" + r.expr);
}

/// \brief Sine of symbex operand.
/// \ingroup symbex_ops
gaalet::symbex
sin(const gaalet::symbex &a)
{
    return gaalet::symbex("sin(" + a.expr + ")");
}

/// \brief Cosine of symbex operand.
/// \ingroup symbex_ops
gaalet::symbex
cos(const gaalet::symbex &a)
{
    return gaalet::symbex("cos(" + a.expr + ")");
}

/// \brief Square root of symbex operand.
/// \ingroup symbex_ops
gaalet::symbex
sqrt(const gaalet::symbex &a)
{
    return gaalet::symbex("sqrt(" + a.expr + ")");
}

#endif
