#ifndef VISTLE_READENSIGHT_CASEPARSERDRIVER_H
#define VISTLE_READENSIGHT_CASEPARSERDRIVER_H

#include "CaseFile.h"
#include "CaseParser.h"
#include <string>
#include <fstream>

class CaseLexer;
class DataItem;
class TimeSet;
class CaseParserDriver;

int ensightlex(ensight::parser::value_type *, CaseParserDriver &drv);

class CaseParserDriver {
    friend class ensight::parser;
    friend int ensightlex(ensight::parser::value_type *, CaseParserDriver &drv);

public:
    CaseParserDriver(const std::string &sFileName);
    virtual ~CaseParserDriver();

    CaseFile getCaseObj();
    bool isOpen();
    void setVerbose(bool enable);
    bool parse();
    std::string lastError() const;
    int lineNumber() const;
    void setLineNumber(int l);
    void setText(const std::string &text); // as controlled by lexer, for error reporting
    void setState(int state); // as controlled by lexer start conditions, for error reporting

private:
    std::ifstream *inputFile_ = nullptr;
    CaseLexer *lexer_ = nullptr;
    CaseFile caseFile_;
    DataItem *actIt_ = nullptr;
    TimeSet *actTs_ = nullptr;
    bool isOpen_ = false;
    bool verbose_ = false;
    int tsStart_ = 0;
    std::string lastError_;
    std::string text_;
    int state_ = 0;

    int lineno_ = 0;
};
#endif
