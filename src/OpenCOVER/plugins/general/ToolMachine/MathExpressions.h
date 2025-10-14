#ifndef  TOOLMACHNINE_MATH_EXPRESSIONS_H
#define TOOLMACHNINE_MATH_EXPRESSIONS_H

#include <exprtk.hpp>
#include <DataClient/DataClient.h>
class MathExpressionObserver{
public:
    struct ObserverHandle{
        bool updated = false;
        std::vector<std::string> valueNames;

        MathExpressionObserver *observer = nullptr;
        ~ObserverHandle();
        typedef std::unique_ptr<ObserverHandle> ptr;
        exprtk::expression<double> m_expression;
        double value() const {return m_expression.value();}
    };
    
    MathExpressionObserver(opencover::dataclient::Client *client);
    [[nodiscard]] ObserverHandle::ptr observe(const std::string &expression);
    void update(); //call before using ObserverHandle::value
private:
    opencover::dataclient::Client *m_client;
    struct RefCountObserveHandle{
        int count = 0;
        opencover::dataclient::ObserverHandle handle;
        double value = 0;
    };
    std::map<std::string, RefCountObserveHandle> m_opcuaHandles;
    exprtk::symbol_table<double> m_symbolTable;
    exprtk::parser<double> m_parser;

};

#endif // TOOLMACHNINE_MATH_EXPRESSIONS_H