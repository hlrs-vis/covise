#include "tryPrint.h"

#include <string>

namespace test
{

    std::ostream &operator<<(std::ostream &os, const test::Printable &p)
    {
        os << "printable";
        return os;
    }

    void test_tryPrint()
    {
        static_assert(covise::HasPrintMethod<Printable>::value, "HasPrintMethod failed to find stream operator");
        static_assert(!covise::HasPrintMethod<NotPrintable>::value, "HasPrintMethod found stream operator where none should exist");
        static_assert(covise::HasPrintMethod<const char *>::value, "HasPrintMethod failed to find stream operator for const char*");
        static_assert(covise::HasPrintMethod<std::string>::value, "HasPrintMethod failed to find stream operator for std::string");

        covise::tryPrint(int{5});
        std::cerr << std::endl;
        covise::tryPrint(float{5.5f});
        std::cerr << std::endl;
        covise::tryPrint(std::string{"hello world"});
        std::cerr << std::endl;

        const char *c = "hello world2";
        covise::tryPrint(c);
        std::cerr << std::endl;
        covise::tryPrint("hello world3");
        std::cerr << std::endl;
        covise::tryPrint(Printable{});
        std::cerr << std::endl;
        covise::tryPrint(NotPrintable{});
        std::cerr << std::endl;
    }
} // namespace test
