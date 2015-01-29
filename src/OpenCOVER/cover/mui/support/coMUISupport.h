// no class; just some support-functions
#ifndef COMUISUPPORT_H
#define COMUISUPPORT_H

#include <iostream>

namespace coMUISupport{

// returns Integers from a string fin the order, they appeared
int readIntFromString(const std::string String, int pos);
int readIntFromStringGetArraySize(const std::string String);

}

#endif
