#include "CaseParserDriver.h"
#include "CaseLexer.h"
#include "CaseFile.h"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

void ensight::parser::error(const std::string &msg)
{
    std::stringstream str;
    str << "parse error at line " << driver.lineno_ << ", text=\"" << driver.text_ << "\", state=" << driver.state_
        << ": " << msg;
    driver.lastError_ = str.str();
    std::cerr << driver.lastError_ << std::endl;
}

CaseParserDriver::CaseParserDriver(const std::string &sFileName)
{
    isOpen_ = false;

    inputFile_ = new std::ifstream(sFileName.c_str());
    if (!inputFile_->is_open()) {
        delete inputFile_;
        inputFile_ = nullptr;
        std::stringstream str;
        str << "could not open " << sFileName << " for reading";
        lastError_ = str.str();
        std::cerr << lastError_ << std::endl;
        return;
    }

    inputFile_->peek(); // try to exclude directories
    if (inputFile_->fail()) {
        delete inputFile_;
        inputFile_ = nullptr;
        std::stringstream str;
        str << "could not open " << sFileName << " for reading - fail";
        lastError_ = str.str();
        std::cerr << lastError_ << std::endl;
        return;
    }

    isOpen_ = true;

    lexer_ = new CaseLexer(inputFile_);
    lexer_->set_debug(1);
    //lexer_->set_debug( 0 );
}

CaseParserDriver::~CaseParserDriver()
{
    delete lexer_;
    lexer_ = nullptr;

    delete inputFile_;
    inputFile_ = nullptr;
}

std::string CaseParserDriver::lastError() const
{
    return lastError_;
}

void CaseParserDriver::setVerbose(bool enable)
{
    verbose_ = enable;
}

bool CaseParserDriver::parse()
{
    lexer_->set_debug(verbose_ ? 1 : 0);
    ensight::parser parse(*this);
    parse.set_debug_level(verbose_);
    return parse() == 0;
}

int CaseParserDriver::lineNumber() const
{
    return lineno_;
}

void CaseParserDriver::setLineNumber(int l)
{
    lineno_ = l;
}

void CaseParserDriver::setText(const std::string &text)
{
    text_ = text;
}

void CaseParserDriver::setState(int state)
{
    state_ = state;
}

int ensightlex(ensight::parser::value_type *yylval, CaseParserDriver &driver)
{
    //std::cerr << "lexing at driver " << driver.location << " / ptr " << *loc << std::endl;
    //ensightlineno   = driver.lexer_->yylineno;

    return driver.lexer_->scan(yylval, driver);
}

CaseFile CaseParserDriver::getCaseObj()
{
    return caseFile_;
}

bool CaseParserDriver::isOpen()
{
    return isOpen_;
}
