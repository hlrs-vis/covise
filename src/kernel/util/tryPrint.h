#ifndef COVISE_TRY_PRINT_H
#define COVISE_TRY_PRINT_H
#include <type_traits>
#include <iostream>
namespace covise
{

    template <class T>
    static auto hasPrintMethod(int)
        -> std::integral_constant<bool, std::is_reference<decltype(operator<<(std::cerr, T()))>::value>;

    template <class>
    static auto hasPrintMethod(...) -> std::false_type;

    template <typename T>
    struct HasPrintMethod : decltype(hasPrintMethod<T>(0))
    {
    };

    template <class T>
    typename std::enable_if<HasPrintMethod<T>::value>::type
    tryPrintWithError(std::ostream &os, const T &t, const char *className, const char *errorMsg)
    {
        os << t;
    }

    template <class T>
    typename std::enable_if<!HasPrintMethod<T>::value && std::is_arithmetic<T>::value>::type
    tryPrintWithError(std::ostream &os, const T &t, const char *className, const char *errorMsg)
    {
        os << t;
    }

    // same as above
    template <class T>
    typename std::enable_if<!HasPrintMethod<T>::value && !std::is_arithmetic<T>::value>::type
    tryPrintWithError(std::ostream &os, const T &t, const char *className, const char *errorMsg)
    {
        if (className && errorMsg)
        {
            os << className << errorMsg;
        }
    }


    template<>
    inline void tryPrintWithError<char const*>(std::ostream& os, const char* const& t, const char* className, const char* errorMsg)
    {
        if (t)
        {
            os << t;
        }
        else
        {
            os << "nullptr";
        }
    }

    template<>
    inline void tryPrintWithError<char *>(std::ostream& os, char* const& t, const char* className, const char* errorMsg)
    {
        if (t)
        {
            os << t;
        }
        else
        {
            os << "nullptr";
        }
    }


    template <class T>
    void tryPrint(const T &t)
    {
        tryPrintWithError(std::cerr, t, nullptr, nullptr);
    }

} // namespace covise

#endif // !COVISE_TRY_PRINT_H