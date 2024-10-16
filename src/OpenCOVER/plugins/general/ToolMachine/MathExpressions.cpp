#include "MathExpressions.h"
#include <exprtk.hpp>
#include <cassert>
#include <boost/algorithm/string.hpp>

bool istReserved(const std::string& symbol)
{
   return exprtk::details::is_reserved_word(symbol) || exprtk::details::is_reserved_symbol(symbol);
}

std::vector<std::string> getSymbols(const std::string &expression_string)
{
   std::vector<std::string> symbols;
   exprtk::collect_variables(expression_string, symbols);
   return symbols;
}

MathExpressionObserver::MathExpressionObserver(opencover::opcua::Client *client)
: m_client(client){}

MathExpressionObserver::ObserverHandle::~ObserverHandle()
{
    for(const auto &v : valueNames)
    {
        auto it = std::find_if(observer->m_opcuaHandles.begin(), observer->m_opcuaHandles.end(),
                [&v](const auto &node) {
                    return boost::iequals(v, node.first);
                });
        assert(it != observer->m_opcuaHandles.end());
        it->second.count--; 
        if(it->second.count == 0)
        {
            observer->m_opcuaHandles.erase(it);
        }
    }
}

MathExpressionObserver::ObserverHandle::ptr MathExpressionObserver::observe(const std::string &expression)
{
    auto symbols = getSymbols(expression);
    auto availableNodes = m_client->allAvailableScalars();
    for(auto s : symbols)
    {
        auto serverSymbol = std::find_if(availableNodes.begin(), availableNodes.end(),
                        [&s](const std::string &node) {
                            return boost::iequals(s, node);
                        });
        if(*serverSymbol == opencover::opcua::NoNodeName)
            return nullptr;

        if(serverSymbol == availableNodes.end())
        {
            std::cerr << "MathExpressionObserver: could not find opcua node " << s << std::endl;
            return nullptr;
        }
        s = *serverSymbol;
        auto opcuaHandle = m_opcuaHandles.find(s);
        if(opcuaHandle != m_opcuaHandles.end())
        {
            opcuaHandle->second.count++;
        } else{
            opcuaHandle = m_opcuaHandles.emplace(s, RefCountObserveHandle{1, m_client->observeNode(s)}).first;
            m_symbolTable.add_variable(s, opcuaHandle->second.value); 
        }
    }
    auto newHandle = std::make_unique<ObserverHandle>();
    newHandle->valueNames = {symbols.begin(), symbols.end()};
    newHandle->observer = this;
    newHandle->m_expression.register_symbol_table(m_symbolTable);
    if(!m_parser.compile(expression, newHandle->m_expression))
    {
        std::cerr << "MathExpressionObserver: could not compile expression " << expression << std::endl;
        std::cerr << m_parser.error() << std::endl;
        return nullptr;
    }
    return newHandle;
}

void MathExpressionObserver::update()
{
    for(auto &h : m_opcuaHandles)
    {
        h.second.value = m_client->getNumericScalar(h.second.handle);
    }
}