#ifndef READENSIGHT_CASEPARSERDRIVER_H
#define READENSIGHT_CASEPARSERDRIVER_H

#include "CaseFile.h"
#include "CaseParser.h"
#include <string>
#include <fstream>

class CaseLexer;
class DataItem;
class TimeSet;
class CaseParserDriver;

int ensightlex(ensight::parser::value_type *, ensight::location *, CaseParserDriver &drv);

class CaseParserDriver {
    friend class ensight::parser;
    friend int ensightlex(ensight::parser::value_type *, ensight::location *, CaseParserDriver &drv);

public:
    CaseParserDriver(const std::string &sFileName);
    virtual ~CaseParserDriver();

    CaseFile getCaseObj();
    bool isOpen();
    void setVerbose(bool enable);
    bool parse();

private:
    std::ifstream *inputFile_ = nullptr;
    CaseLexer *lexer_ = nullptr;
    CaseFile caseFile_;
    DataItem *actIt_ = nullptr;
    TimeSet *actTs_ = nullptr;
    bool isOpen_ = false;
    bool verbose_ = false;
    int tsStart_ = 0;

    ensight::location location;
};
#endif
