#include "LevenshteinDistane.h"
#include <vector>

// Case-sensitive char comparer
bool helper_cmpChar(char a, char b)
{
    return (a == b);
}

// Case-insensitive char comparer
bool helper_cmpCharIgnoreCase(char a, char b)
{
    return (std::tolower(a) == std::tolower(b));
}

size_t opencover::utils::string::computeLevensteinDistance(const std::string &s1, const std::string &s2, bool ignoreCase)
{
    const auto &len1 = s1.size(), len2 = s2.size();

    // allocate distance matrix
    std::vector<std::vector<size_t>> d(len1 + 1, std::vector<size_t>(len2 + 1));

    auto isEqual = [&](char a, char b) {
        return (ignoreCase) ? helper_cmpCharIgnoreCase(a, b) : helper_cmpChar(a, b);
    };

    d[0][0] = 0;
    // compute distance
    for (int i = 1; i <= len1; ++i)
        d[i][0] = i;
    for (int j = 1; j <= len2; ++j)
        d[0][j] = j;

    for (int i = 1; i <= len1; ++i) {
        for (int j = 1; j <= len2; ++j) {
            if (isEqual(s1[i - 1], s2[j - 1]))
                d[i][j] = d[i - 1][j - 1];
            else
                d[i][j] = std::min({d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1});
        }
    }

    return d[len1][len2];
}