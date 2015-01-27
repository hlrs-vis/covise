/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef HFT_STRING_H_
#define HFT_STRING_H_

#include <string>
//Reine Methoden-Bibliothek für std::string Klasse
std::string::size_type HfT_replace_char_in_string(std::string *str, char oldc, char newc);

std::string HfT_double_to_string(double d);

double HfT_string_to_double(std::string str);

std::string HfT_int_to_string(int d);

int HfT_string_to_int(std::string str);

std::string HfT_replace_string_in_string(std::string instr, std::string repstr, std::string newstr);

std::string HfT_cut_LastBlanks_in_string(std::string instr);
std::string HfT_cut_LastDigits_in_string(std::string str, unsigned int round);

#endif /* HFT_STRING_H_ */
