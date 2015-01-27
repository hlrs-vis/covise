/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// disable warnings about long names
#ifdef WIN32
#pragma warning(disable : 4786)
#endif

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <functional>

using namespace std;

#define SPACES " \t\r\n"

inline string trim_right(const string &s, const string &t = SPACES)
{
    string d(s);
    string::size_type i(d.find_last_not_of(t));
    if (i == string::npos)
        return "";
    else
        return d.erase(i + 1);
} // end of trim_right

inline string trim_left(const string &s, const string &t = SPACES)
{
    string d(s);
    return d.erase(0, s.find_first_not_of(t));
} // end of trim_left

inline string trim(const string &s, const string &t = SPACES)
{
    string d(s);
    return trim_left(trim_right(d, t), t);
} // end of trim

string tolower(const string &s);
string toupper(const string &s);

class fCapitals : public unary_function<char, char>
{
    bool bUpper;

public:
    // first letter in string will be in capitals
    fCapitals()
        : bUpper(true){}; // constructor

    char operator()(const char &c)
    {
        char c1;
        // capitalise depending on previous letter
        if (bUpper)
            c1 = toupper(c);
        else
            c1 = tolower(c);

        // work out whether next letter should be capitals
        bUpper = isalnum(c) == 0;
        return c1;
    }
}; // end of class fCapitals

void StringToVector(const string s,
                    vector<string> &v,
                    const string delim = " ",
                    const bool trim_spaces = true);

string VectorToString(const vector<string> &v,
                      const string delim = " ");

struct ci_equal_to : binary_function<string, string, bool>
{

    struct compare_equal
        : public binary_function<unsigned char, unsigned char, bool>
    {
        bool operator()(const unsigned char &c1, const unsigned char &c2) const
        {
            return tolower(c1) == tolower(c2);
        }
    }; // end of compare_equal

    bool operator()(const string &s1, const string &s2) const
    {

        pair<string::const_iterator,
             string::const_iterator> result = mismatch(s1.begin(), s1.end(), // source range
                                                       s2.begin(), // comparison start
                                                       compare_equal()); // comparison

        // match if both at end
        return result.first == s1.end() && result.second == s2.end();
    }
}; // end of ci_equal_to

// compare strings for equality using the binary function above
// returns true is s1 == s2
bool ciStringEqual(const string &s1, const string &s2);

// case-independent (ci) string less_than
// returns true if s1 < s2
struct ci_less : binary_function<string, string, bool>
{

    // case-independent (ci) compare_less binary function
    struct compare_less
        : public binary_function<unsigned char, unsigned char, bool>
    {
        bool operator()(const unsigned char &c1, const unsigned char &c2) const
        {
            return tolower(c1) < tolower(c2);
        }
    }; // end of compare_less

    bool operator()(const string &s1, const string &s2) const
    {

        return lexicographical_compare(s1.begin(), s1.end(), // source range
                                       s2.begin(), s2.end(), // dest range
                                       compare_less()); // comparison
    }
}; // end of ci_less

// compare strings for less-than using the binary function above
// returns true if s1 < s2
bool ciStringLess(const string &s1, const string &s2);
bool strPrefix(const string &s1, // string to search
               const string &s2, // what to look for
               const bool no_case = false);
bool strPrefix(const string &s1, // string to search
               const string::size_type pos, // where in s1 to start
               const string &s2, // what to look for
               const bool no_case = false);
