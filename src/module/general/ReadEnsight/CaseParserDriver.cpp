#include "CaseParserDriver.h"
#include "CaseLexer.h"
#include "CaseFile.h"

#include <string>
#include <iostream>
#include <fstream>

void ensight::parser::error(const ensight::location &location, const std::string &msg)
{
    std::cerr << "parse error at " << location << ": " << msg << std::endl;
}

CaseParserDriver::CaseParserDriver(const std::string &sFileName)
{
    isOpen_ = false;

    location.initialize(&sFileName);

    inputFile_ = new std::ifstream(sFileName.c_str());
    if (!inputFile_->is_open()) {
        fprintf(stderr, "could not open %s for reading", sFileName.c_str());
        delete inputFile_;
        inputFile_ = nullptr;
        return;
    }

    inputFile_->peek(); // try to exclude directories
    if (inputFile_->fail()) {
        fprintf(stderr, "could not open %s for reading - fail", sFileName.c_str());
        delete inputFile_;
        inputFile_ = nullptr;
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

int ensightlex(ensight::parser::value_type *yylval, ensight::location *, CaseParserDriver &driver)
{
    return (driver.lexer_->scan(yylval));
}

CaseFile CaseParserDriver::getCaseObj()
{
    return caseFile_;
}

bool CaseParserDriver::isOpen()
{
    return isOpen_;
}
