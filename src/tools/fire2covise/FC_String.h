/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//----------------------------------------------------------------------
//  $Id: FC_String.h,v 1.15 2000/03/01 12:53:55 goesslet Exp goesslet $
//----------------------------------------------------------------------
//
//  $Log: FC_String.h,v $
//  Revision 1.15  2000/03/01 12:53:55  goesslet
//  return len after read
//
//  Revision 1.14  2000/02/17 11:42:48  goesslet
//  added virtal to destructor
//
//  Revision 1.13  2000/01/05 10:18:25  goesslet
//  removed operator[]
//
//  Revision 1.12  2000/01/04 13:01:17  goesslet
//  changed len and size to unsigned long for 64bit
//
//  Revision 1.11  1999/12/16 14:20:57  goesslet
//  fileName tools added
//
//  Revision 1.10  1999/12/15 08:51:02  goesslet
//  removed PLATFORM dependent includes
//
//  Revision 1.9  1999/12/15 07:29:48  goesslet
//  return of len is int
//
//  Revision 1.8  1999/12/06 12:39:52  goesslet
//  new function Print which is simular to printf
//  but returns an FC_String
//
//  Revision 1.7  1999/12/01 12:12:52  goesslet
//  resolved compile bugs on hpux
//
//  Revision 1.6  1999/11/30 16:58:11  goesslet
//  new functions for checking if is a string ant to convert
//  from and to a string
//
//  Revision 1.5  1999/11/23 09:43:30  goesslet
//  new functions IsSubStr and IsEqual with option caseSensitve yse or no
//
//  Revision 1.4  1999/11/16 14:13:59  goesslet
//  initial version of FC_StdIO
//
//  Revision 1.3  1999/11/11 12:49:47  goesslet
//  1. Function Read from a FILE *id
//  2. Function Extract to get a new string at postion and len
//
//  Revision 1.2  1999/11/10 13:10:55  goesslet
//  adde function for spliting FC_String
//  bacause of a FC_String and return an array
//  of all FC_Strings
//
//  Revision 1.1  1999/11/08 13:24:25  goesslet
//  initial version
//
//----------------------------------------------------------------------

#ifndef _FC_STRING_H_
#define _FC_STRING_H_

#include "FC_Base.h"
#include "FC_Array.h"
#include "FC_ListArray.h"

class FC_String;
typedef FC_Array<FC_String> FC_StringArray;
typedef FC_ListArray<FC_String> FC_StringListArray;

class FC_String : public FC_Base
{
    IMPLEMENT_FIRE_BASE(FC_String, FC_Base)

    // Relational operations between FC_String objects
    friend int operator==(const FC_String &x, const char *y);
    friend int operator==(const FC_String &x, const FC_String &y);
    friend int operator!=(const FC_String &x, const FC_String &y);
    friend int operator!=(const FC_String &x, const char *y);
    friend int operator!(const FC_String &x);
    friend int operator<(const FC_String &x, const FC_String &y);
    friend int operator>(const FC_String &x, const FC_String &y);
    friend int operator<=(const FC_String &x, const FC_String &y);
    friend int operator>=(const FC_String &x, const FC_String &y);

    // Concatenation
    friend FC_String operator+(const FC_String &, const FC_String &);
    friend FC_String operator+(const FC_String &, const char *);
    friend FC_String operator+(const char *, const FC_String &);
    friend FC_String operator+(const FC_String &, const int &i);
    friend FC_String operator+(const FC_String &, const float &i);
    friend FC_String operator+(const FC_String &, const double &i);

private:
    char *s;
    unsigned long len;
    unsigned long size;

    static const char pathSeperator;

public:
    // Constructors
    // Default convert text to text with special format
    FC_String(const char *val = "", const char *fmt = "%s");
    // text and len
    FC_String(const char *text, unsigned lenIn);
    FC_String(const FC_String &); // From another FC_String

    // convert int to text
    FC_String(const int &val, const char *fmt = "%d");
    // convert float to text
    FC_String(const float &val, const char *fmt = "%g");
    // convert double to text
    FC_String(const double &val, const char *fmt = "%lg");

    // Destructor
    virtual ~FC_String();

    // Assignment
    const FC_String &operator=(const FC_String &text);
    const FC_String &operator=(const char *text);

    // Subtract a substring from a string
    FC_String operator-(const FC_String &) const;
    FC_String operator-(const char *) const;

    // Searches
    FC_String Find(const FC_String &) const;
    FC_String Find(const char &c) const;
    FC_String FindLast(const char &c) const;

    FC_String Extract(const unsigned long &pos, const unsigned long &len = 0) const;

    int Split(const FC_String &split, FC_StringListArray &data) const;

    unsigned long Length() const;

    const char *MakeStr() const;
    operator const char *() const;

    void Replace(const char &c1, const char &c2);

    // Conversions to char*, double, int, long int, unsigned long int, char
    const char &ToChar(unsigned long pos) const
    {
        return s[pos];
    }
    char &ToChar(unsigned long pos)
    {
        return s[pos];
    }
    FC_String ToString() const; // adds a '"' in front and end of the string
    double ToDouble() const;
    float ToFloat() const;
    int ToInt() const;
    long int ToLong() const;
    unsigned long int ToULong() const;

    // calculate string
    double Calculate() const;

    // text conversions
    void ToNormalString(); // removes '"' in front and end
    void ToUpper();
    void ToLower();
    void DeleteBlanksAndTabs();
    void DeleteBlanksAndTabsAtEnd();
    void DeleteBlanksAndTabsAtBorder();

    // check for valuetype
    int IsString() const; // check if first and last character is '"'
    int IsValue() const; // check if int or real value
    int IsIntValue() const; // check if int value
    int IsRealValue() const; // check if real value

    // check if is substring yes/no case sensitive
    int IsSubStr(const FC_String &nameIn, const int &caseSensitive = 0) const;
    // check if string is equal yes/no case sensitive
    int IsEqual(const FC_String &nameIn, const int &caseSensitive = 0) const;

    // filenameTools
    FC_String GetExtensionOfFileName() const;
    FC_String GetFileName() const;
    FC_String GetPathOfFileName() const;
    void RemoveExtensionOfFileName();
    void RemovePathOfFileName();
    void AddExtensionToFileName(const FC_String &extension);
    void AddPathToFileName(const FC_String &path);
    void AddFileNameToPath(const FC_String &fileName);

    // I/O
    void Write(ostream & = cout) const;
    int Read(istream & = cin, const char &c = '\n');
    int Read(FILE *id, const int &len);

private:
    void init(); // init all private variables
    void initSize(const unsigned long &); // Used by + operator

    void replaceDtoE(); // used for conv to float or double
};

extern FC_String Print(const char *format, ...);

ostream &operator<<(ostream &os, const FC_String &s);
istream &operator>>(istream &is, FC_String &s);
#endif // __STRING_CLASS_H
