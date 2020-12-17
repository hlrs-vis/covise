#ifndef TEST_TRY_PRINT_H
#define TEST_TRY_PRINT_H

#include <util/tryPrint.h>

#include <iostream>


namespace test{

struct Printable{
};

std::ostream &operator<<(std::ostream &os, const test::Printable &p);

struct NotPrintable{
};

void test_tryPrint();

}//test


#endif //!TEST_TRY_PRINT_H