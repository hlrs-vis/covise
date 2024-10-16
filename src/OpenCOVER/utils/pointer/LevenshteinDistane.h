#ifndef _LEVENSHTEINDISTANCE_H
#define _LEVENSHTEINDISTANCE_H

#include "export.h"
#include <string>

namespace opencover::utils::string {

/**
 * Computes the Levenshtein distance between two strings. 0 means the strings are equal. 
 * The higher the number, the more different chars are in the strings.
 * e.g. "kitten" and "sitting" have a Levenshtein distance of 3.
 * Source: http://www.blackbeltcoder.com/Articles/algorithms/approximate-string-comparisons-using-levenshtein-distance
 *
 * @param s1 The first string.
 * @param s2 The second string.
 * @param ignoreCase Flag indicating whether to ignore case sensitivity (default: false).
 * @return The Levenshtein distance between the two strings.
 */


size_t SRTING_EXPORT computeLevensteinDistance(const std::string &s1, const std::string &s2, bool ignoreCase = false);

} //opencover::utils::string
#endif // _LEVENSHTEINDISTANCE_H