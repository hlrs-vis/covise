#ifndef COVISE_TOKEN_BUFFER_UTIL_H
#define COVISE_TOKEN_BUFFER_UTIL_H

namespace covise {

template<typename Stream, std::size_t I = 0, typename... Tp>
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



template<typename...Args>
struct TbReadVal {
	std::tuple<Args...> value;
	template<typename T>
	const T& get() const {
		return std::get<T>(value);
	}


};
template<typename Stream, typename...Args>
Stream& operator<<(Stream& s, const TbReadVal<Args...>& dt)
{
	print(s, dt.value);
	return s;
}

template<typename...Args>
TbReadVal<Args...> readTokenBuffer(covise::TokenBuffer& tb) {
	TbReadVal<Args...> retval;
	detail::readTokenBufferSingle< std::tuple<Args...>, Args...>(retval.value, tb);
	return retval;
}
namespace detail {
template<typename STDTuple, typename T, typename...Args>
void readTokenBufferSingle(STDTuple& tuple, covise::TokenBuffer& tb) {
	static_assert(isTBClass<T>(), "type is not created with CREATE_TB_CLASS");
	T& val = std::get<T>(tuple);
	tb >> val.value;
	readTokenBufferSingle<STDTuple, Args...>(tuple, tb);
}

template<typename STDTuple>
void readTokenBufferSingle(STDTuple& tuple, covise::TokenBuffer& tb) {
	return;
}

template<typename T>
constexpr bool isTBClass() { return false; }
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