/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// disable warnings about long names
#ifdef WIN32
#pragma warning(disable : 4786)
#endif

#include <iostream>
#include "string_utils.h"
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <functional>

using namespace std;

// returns a lower case version of the string
string tolower(const string &s)
{
    string d(s);

    transform(d.begin(), d.end(), d.begin(), (int (*)(int))tolower);
    return d;
} // end of tolower

// returns an upper case version of the string
string toupper(const string &s)
{
    string d(s);

    transform(d.begin(), d.end(), d.begin(), (int (*)(int))toupper);
    return d;
} // end of toupper

// transformation function for tocapitals that has a "state"
// so it can capitalise a sequence
// returns a capitalized version of the string
string tocapitals(const string &s)
{
    string d(s);

    transform(d.begin(), d.end(), d.begin(), fCapitals());
    return d;
} // end of tocapitals

// split a line into the first word, and rest-of-the-line
string GetWord(string &s,
               const string delim,
               const bool trim_spaces)
{

    // find delimiter
    string::size_type i(s.find(delim));

    // split into before and after delimiter
    string w(s.substr(0, i));

    // if no delimiter, remainder is empty
    if (i == string::npos)
        s.erase();
    else
        // erase up to the delimiter
        s.erase(0, i + delim.size());

    // trim spaces if required
    if (trim_spaces)
    {
        w = trim(w);
        s = trim(s);
    }

    // return first word in line
    return w;

} // end of GetWord

// To be symmetric, we assume an empty string (after trimming spaces)
// will give an empty vector.
// However, a non-empty string (with no delimiter) will give one item
// After that, you get an item per delimiter, plus 1.
// eg.  ""      => empty
//      "a"     => 1 item
//      "a,b"   => 2 items
//      "a,b,"  => 3 items (last one empty)

void StringToVector(const string s,
                    vector<string> &v,
                    const string delim,
                    const bool trim_spaces)
{

    // start with initial string, trimmed of leading/trailing spaces if required
    string s1(trim_spaces ? trim(s) : s);

    v.clear(); // ensure vector empty

    // no string? no elements
    if (s1.empty())
        return;

    // add to vector while we have a delimiter
    while (!s1.empty() && s1.find(delim) != string::npos)
        v.push_back(GetWord(s1, delim, trim_spaces));

    // add final element
    v.push_back(s1);
} // end of StringToVector

// Takes a vector of strings and converts it to a string
// like "apples,peaches,pears"
// Should be symmetric with StringToVector (excepting any spaces that might have
//  been trimmed).

string VectorToString(const vector<string> &v,
                      const string delim)
{
    // vector empty gives empty string
    if (v.empty())
        return "";

    // for copying results into
    ostringstream os;

    // copy all but last one, with delimiter after each one
    copy(v.begin(), v.end() - 1,
         ostream_iterator<string>(os, delim.c_str()));

    // return string with final element appended
    return os.str() + *(v.end() - 1);

} // end of VectorToString

/*

Output

Converted string into 9 items.
Vector back to string = (happy, days, are, here, again, , , , )
original             -->  happy/days/ are /here/again////  <--
trim_right           -->  happy/days/ are /here/again////<--
trim_left            -->happy/days/ are /here/again////  <--
trim                 -->happy/days/ are /here/again////<--
tolower              -->  happy/days/ are /here/again////  <--
toupper              -->  HAPPY/DAYS/ ARE /HERE/AGAIN////  <--
tocapitals           -->  Happy/Days/ Are /Here/Again////  <--
GetWord              -->happy<--
After GetWord, s =   -->days/ are /here/again////<--

*/

#include <string>
#include <iostream>
#include <functional>

using namespace std;

// case-independent (ci) string compare
// returns true if strings are EQUAL

bool ciStringEqual(const string &s1, const string &s2)
{
    return ci_equal_to()(s1, s2);
} // end of ciStringEqual

// compare strings for less-than using the binary function above
// returns true if s1 < s2
bool ciStringLess(const string &s1, const string &s2)
{
    return ci_less()(s1, s2);
} // end of ciStringLess

// compare to see if start of s1 is s2
//  eg. returns true for: strPrefix ("abacus", "aba");

bool strPrefix(const string &s1, // string to search
               const string &s2, // what to look for
               const bool no_case) // case-insensitive?
{

    // if either string is empty or s1 is smaller than s2
    //  then they can't be identical
    if (s1.empty() || s2.empty() || s1.size() < s2.size())
        return false;

    if (no_case)
        return ciStringEqual(s1.substr(0, s2.size()), s2);
    else
        return s1.substr(0, s2.size()) == s2;

} // end of strPrefix

// compares to see if (s1, offset by pos, for length s2) == s2

bool strPrefix(const string &s1, // string to search
               const string::size_type pos, // where in s1 to start
               const string &s2, // what to look for
               const bool no_case) // case-insensitive?
{
    // can't be true if position outside string size
    // casts are to ensure a signed comparison
    if ((int)pos >= ((int)s1.size() - (int)s2.size()))
        return false;

    // make a substring of s1 for s2's size
    return strPrefix(s1.substr(pos, s2.size()), s2, no_case);

} // end of strPrefix
