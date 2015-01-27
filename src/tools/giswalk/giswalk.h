/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef GISWALK_H
#define GISWALK_H

#include <list>
#include <vector>
#include <math.h>
#define MaxHabitatValues 256

#ifdef HAVE_XERCESC
#include <xercesc/dom/DOM.hpp>
#if XERCES_VERSION_MAJOR < 3
#include <xercesc/dom/DOMWriter.hpp>
#else
#include <xercesc/dom/DOMLSSerializer.hpp>
#endif
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLUni.hpp>
#endif
#ifdef HAVE_TIFF
extern unsigned char *tifread(const char *url, int *w, int *h, int *nc);
#endif

class gwApp;
#ifndef COV_WINCOMPAT_H
#ifdef WIN32
#if !defined(__MINGW32__)
inline int strcasecmp(const char *s1, const char *s2)
{
    return stricmp(s1, s2);
}
#endif

#if !defined(__MINGW32__)
inline int strncasecmp(const char *s1, const char *s2, size_t n)
{
    return strnicmp(s1, s2, n);
}
#endif
#endif
#endif
class vec2
{
public:
    /** Type of Vec class.*/
    typedef float value_type;

    /** Number of vector components. */
    enum
    {
        num_components = 2
    };

    /** Vec member varaible. */
    value_type _v[2];

    vec2()
    {
        _v[0] = 0.0;
        _v[1] = 0.0;
    }
    vec2(value_type x, value_type y)
    {
        _v[0] = x;
        _v[1] = y;
    }

    inline bool operator==(const vec2 &v) const
    {
        return _v[0] == v._v[0] && _v[1] == v._v[1];
    }

    inline bool operator!=(const vec2 &v) const
    {
        return _v[0] != v._v[0] || _v[1] != v._v[1];
    }

    inline bool operator<(const vec2 &v) const
    {
        if (_v[0] < v._v[0])
            return true;
        else if (_v[0] > v._v[0])
            return false;
        else
            return (_v[1] < v._v[1]);
    }

    inline value_type *ptr()
    {
        return _v;
    }
    inline const value_type *ptr() const
    {
        return _v;
    }

    inline void set(value_type x, value_type y)
    {
        _v[0] = x;
        _v[1] = y;
    }

    inline value_type &operator[](int i)
    {
        return _v[i];
    }
    inline value_type operator[](int i) const
    {
        return _v[i];
    }

    inline value_type &x()
    {
        return _v[0];
    }
    inline value_type &y()
    {
        return _v[1];
    }

    inline value_type x() const
    {
        return _v[0];
    }
    inline value_type y() const
    {
        return _v[1];
    }

    /** Dot product. */
    inline value_type operator*(const vec2 &rhs) const
    {
        return _v[0] * rhs._v[0] + _v[1] * rhs._v[1];
    }

    /** Multiply by scalar. */
    inline const vec2 operator*(value_type rhs) const
    {
        return vec2(_v[0] * rhs, _v[1] * rhs);
    }

    /** Unary multiply by scalar. */
    inline vec2 &operator*=(value_type rhs)
    {
        _v[0] *= rhs;
        _v[1] *= rhs;
        return *this;
    }

    /** Divide by scalar. */
    inline const vec2 operator/(value_type rhs) const
    {
        return vec2(_v[0] / rhs, _v[1] / rhs);
    }

    /** Unary divide by scalar. */
    inline vec2 &operator/=(value_type rhs)
    {
        _v[0] /= rhs;
        _v[1] /= rhs;
        return *this;
    }

    /** Binary vector add. */
    inline const vec2 operator+(const vec2 &rhs) const
    {
        return vec2(_v[0] + rhs._v[0], _v[1] + rhs._v[1]);
    }

    /** Unary vector add. Slightly more efficient because no temporary
          * intermediate object.
        */
    inline vec2 &operator+=(const vec2 &rhs)
    {
        _v[0] += rhs._v[0];
        _v[1] += rhs._v[1];
        return *this;
    }

    /** Binary vector subtract. */
    inline const vec2 operator-(const vec2 &rhs) const
    {
        return vec2(_v[0] - rhs._v[0], _v[1] - rhs._v[1]);
    }

    /** Unary vector subtract. */
    inline vec2 &operator-=(const vec2 &rhs)
    {
        _v[0] -= rhs._v[0];
        _v[1] -= rhs._v[1];
        return *this;
    }

    /** Negation operator. Returns the negative of the vec2. */
    inline const vec2 operator-() const
    {
        return vec2(-_v[0], -_v[1]);
    }

    /** Length of the vector = sqrt( vec . vec ) */
    inline value_type length() const
    {
        return sqrtf(_v[0] * _v[0] + _v[1] * _v[1]);
    }

    /** Length squared of the vector = vec . vec */
    inline value_type length2(void) const
    {
        return _v[0] * _v[0] + _v[1] * _v[1];
    }

    /** Normalize the vector so that it has length unity.
          * Returns the previous length of the vector.
        */
    inline value_type normalize()
    {
        value_type norm = vec2::length();
        if (norm > 0.0)
        {
            value_type inv = 1.0f / norm;
            _v[0] *= inv;
            _v[1] *= inv;
        }
        return (norm);
    }

}; // end of class vec2

#endif
