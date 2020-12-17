#ifndef COVISE_TOKEN_BUFFER_UTIL_H
#define COVISE_TOKEN_BUFFER_UTIL_H

#include <type_traits>
#include <tuple>
namespace covise {


namespace detail {

//helper to construct messages out of tokenbuffer
template<typename T>
T get(TokenBuffer &tb){
	T t{};
	tb >> t;
	return t;
}
//to get the c++14 syntax of std::get<T>(tuple)
template<size_t Pos, typename T, typename U, typename...Args>
struct Finder
{
	constexpr inline size_t operator()() const {
		return Finder<Pos + 1, T, Args...>{}();
	}
};

template<size_t Pos, typename T, typename...Args>
struct Finder<Pos, T, T, Args...>
{
	constexpr inline size_t operator()() const {
		return Pos;
	}
};

template<typename T, typename...Args>
const T& get(const std::tuple<Args...>& t) {
	constexpr size_t pos = Finder < size_t{}, T, Args... > {}();
	return std::get<pos>(t);
}

template<typename T, typename...Args>
T& get(std::tuple<Args...>& t) {
	constexpr size_t pos = Finder < size_t{}, T, Args... > {}();
	return std::get<pos>(t);
}
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

template<typename T>
constexpr bool isTBClass() { return false; }

template<typename STDTuple>
void readTokenBufferSingle(STDTuple& tuple, covise::TokenBuffer& tb) {
	return;
}

template<typename STDTuple, typename T, typename...Args>
void readTokenBufferSingle(STDTuple& tuple, covise::TokenBuffer& tb) {
	static_assert(detail::isTBClass<T>(), "type is not created with CREATE_TB_CLASS");
	T& val = detail::get<T>(tuple);
	tb >> val.value;
	readTokenBufferSingle<STDTuple, Args...>(tuple, tb);
}

template<typename Stream, size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
print(Stream& s, const std::tuple<Tp...>& t)
{ }

template<typename Stream, std::size_t I = 0, typename... Tp>
inline typename std::enable_if < I < sizeof...(Tp), void>::type
	print(Stream& s, const std::tuple<Tp...>& t)
{
	s << std::get<I>(t);
	print<Stream, I + 1, Tp...>(s, t);
}

} //detail



template<typename...Args>
struct TbReadVal {
	std::tuple<Args...> value;
	template<typename T>
	const T& get() const {
		return detail::get<T>(value);
	}


};

template<typename...Args>
TbReadVal<Args...> readTokenBuffer(covise::TokenBuffer& tb) {
	TbReadVal<Args...> retval;
	detail::readTokenBufferSingle< std::tuple<Args...>, Args...>(retval.value, tb);
	return retval;
}

template<typename Stream, typename...Args>
Stream& operator<<(Stream& s, const TbReadVal<Args...>& dt)
{
	detail::print(s, dt.value);
	return s;
}

#define CREATE_TB_CLASS(type, name)\
    struct name{\
        type value;\
        operator type()const {return value;}\
	};\
	template<typename Stream>\
	Stream &operator<<(Stream &s, const name& t){\
		s << #name << ": " << t.value << ", ";\
		return s;\
	}\
	namespace detail{\
	template<>\
	constexpr bool isTBClass<name>() { return true; }\
	}
}

#endif
