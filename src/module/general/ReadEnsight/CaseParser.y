/*
    Parser for EnSight case files
    */

%require "3.8"
%language "c++"

%define api.prefix {ensight}
%define api.value.type {struct Token}

%define parse.trace
%define parse.error detailed
%define parse.lac full

%header


%code provides {
    typedef ensight::parser CaseParser;
    // Give Bison token Type a readable name
    typedef ensight::parser::value_type CaseTokenType;

    #define YY_DECL int CaseLexer::scan(CaseTokenType *pToken, CaseParserDriver &driver)
}


// the includes
%code requires
{
#include "CaseFile.h"
#include "DataItem.h"

#include <string>
#include <iostream>
#include <fstream>

class CaseLexer;
class CaseParserDriver;

#ifndef NDEBUG
#define YYDEBUG 1
#endif

struct Token {
    std::string String;
    long Int = 0;
    double Double = 0.0;
};

}

%code top
{
#include "CaseParserDriver.h"
#include "CaseLexer.h"

#define CERR std::cerr << "CaseParser::parse, line " << driver.lineno_ << ": "
}

%param { CaseParserDriver &driver }



// work with untyped tokens

%token <String> FORMAT_SEC TYPE ENSIGHT ENSIGHT_GOLD
%token <String> GEOMETRY_SEC MODEL MATCH
%token <String> CONSTANT PER_CASE PER_CASE_FILE PER_PART
%token <String> MEASURED CH_CO_ONLY CH_GEOM_PER_PART
%token <String> SCRIPTS_SEC METADATA PYTHON
%token <String> QUERY_SEC XY_DATA
%token <String> VARIABLE_SEC
%token <String> SCALAR VECTOR COMPLEX PER_NODE PER_ELEMENT PER_M_NODE PER_M_ELEMENT
%token <String> TENSOR_SYMM TENSOR_ASYMM
%token <String> TIME_SEC TIME_SET NUM_OF_STEPS
%token <String> FN_ST_NUM FN_INCR
%token <String> TIME_VAL
%token <String> IDENTIFIER TEXT QUOTED
%token <String> FILE_SEC FILE_SET FN_NUMS
%token <String> DUMMY_YNERRS_CONSUMER
%token <Int> INTEGER
%token <Double> DOUBLE

%type <Double> number
%type <String> ts_hdr ts_opts ts_fn_start ts_fn_incr ts_fnum_sec ts_fn_nums ts_tval_secc ts_tvals
%type <String> ts_spec
%type <String> string

%%

//

ecase: section
| ecase section

section: sect_key spec

sect_key: FORMAT_SEC
| GEOMETRY_SEC
| VARIABLE_SEC
| TIME_SEC
| FILE_SEC
| SCRIPTS_SEC
| QUERY_SEC

spec: spec_line
| spec spec_line


spec_line: type_spec
| model_spec
| variable_spec
| ts_spec
| fs_spec
| const_spec
| scripts_spec
| query_spec


ts_spec: ts_hdr ts_opts
{
    CERR << "ts_hdr ts_opts " << $1 << $2 << std::endl;
}
| ts_spec ts_hdr ts_opts
{
    CERR << "ts_spec ts_hdr ts_opts " << $1 << $2 << std::endl;
}

ts_hdr: TIME_SET INTEGER NUM_OF_STEPS INTEGER
{
    int ts( $2 );
    int ns( $4 );
    std::cerr << " DEFINITION TIMESET " << ts << "  STEPS " << ns;
    driver.actTs_ = new TimeSet( ts, ns );
}
| TIME_SET INTEGER IDENTIFIER NUM_OF_STEPS INTEGER
{
    int ts( $2 );
    int ns( $5 );
    std::cerr << "DEFINITION TIMESET " << ts << "  STEPS " << ns;
    driver.actTs_ = new TimeSet( ts, ns );
}

ts_opts: ts_fnum_sec ts_tval_secc
| ts_tval_secc
| ts_fnum_sec
| ts_fn_start ts_fn_incr ts_tval_secc


ts_fn_start: FN_ST_NUM INTEGER
{
    int fs($2);
    driver.tsStart_ = fs;
    std::cerr << " FILENAME START " << fs;
}

ts_fn_incr: FN_INCR INTEGER
{
    int incr( $2 );
    std::cerr << "FILENAME INCREMENT " << incr;
    if (driver.actTs_ != nullptr) {
        int i=driver.tsStart_;
        for (int j=0; j<driver.actTs_->getNumTs(); ++j) {
            driver.actTs_->addFileNr(i);
            i += incr;
        }
        if ( driver.actTs_->getNumTs() == driver.actTs_->size() ) {
            // the time-set is full
            driver.caseFile_.addTimeSet( driver.actTs_ );
            driver.actTs_ = nullptr;
        }
    }
}

ts_fnum_sec: FN_NUMS ts_fn_nums
{
    std::cerr << " FILENAME NUMBERS ";
}

ts_fn_nums: ts_fn_nums INTEGER
{
    std::cerr << " INTEGER ";
    int nf($2);
    if (driver.actTs_ != nullptr) {
        driver.actTs_->addFileNr(nf);
        if ( driver.actTs_->getNumTs() == driver.actTs_->size() ) {
            // the time-set is full
            driver.caseFile_.addTimeSet( driver.actTs_ );
            driver.actTs_ = nullptr;
        }
    }
}
| INTEGER
{
    std::cerr << " INTEGER ";
    int nf($1);
    //	     fprintf(stderr, " %d ", nf);
    if (driver.actTs_ != nullptr) {
        driver.actTs_->addFileNr(nf);
        if ( driver.actTs_->getNumTs() == driver.actTs_->size() ) {
            // the time-set is full
            driver.caseFile_.addTimeSet( driver.actTs_ );
            driver.actTs_ = nullptr;
        }
    }
}

ts_tval_secc: TIME_VAL ts_tvals
{
    std::cerr << " TIME_VAL ts_tvals ";
    if ( driver.actTs_ != nullptr ) {
    }
}

ts_tvals: ts_tvals number
{
    std::cerr << " ts_tvals DOUBLE or INTEGER ";
    float rt( $2 );
    if (driver.actTs_ != nullptr) {
        driver.actTs_->addRealTimeVal(rt);
    } else {
        driver.actTs_ = driver.caseFile_.getLastTimeSet();
        driver.actTs_->addRealTimeVal(rt);
    }
}
| number
{
    std::cerr << " number DOUBLE or INTEGER ";
    float rt($1);
    if (driver.actTs_ != nullptr) {
        driver.actTs_->addRealTimeVal(rt);
    } else {
        driver.actTs_ = driver.caseFile_.getLastTimeSet();
        driver.actTs_->addRealTimeVal(rt);
    }
}


fs_spec: fs_hdr
| fs_spec fs_hdr

fs_hdr: FILE_SET INTEGER NUM_OF_STEPS INTEGER
{
    int ts( $2 );
    int ns( $4 );
    std::cerr << "DEFINITION FILESET " << ts << "  STEPS " << ns << std::endl;
    //driver.actTs_ = new TimeSet( ts, ns );
}
| FILE_SET INTEGER IDENTIFIER NUM_OF_STEPS INTEGER
{
    int ts( $2 );
    int ns( $5 );
    std::cerr << "DEFINITION FILESET " << ts << "  STEPS " << ns;
    //driver.actTs_ = new TimeSet( ts, ns );
}


type_spec: TYPE ENSIGHT
{
    //	fprintf(stderr,"  ENSIGHT VERSION 6 found\n");
    driver.caseFile_.setVersion(CaseFile::v6);
}
| TYPE ENSIGHT ENSIGHT_GOLD
{
    //	fprintf(stderr,"  ENSIGHT GOLD found\n");
    driver.caseFile_.setVersion(CaseFile::gold);
}


coord_change_spec:
| CH_CO_ONLY
{
    driver.caseFile_.setConnectivityFileIndex(0);
}
| CH_CO_ONLY INTEGER
{
    driver.caseFile_.setConnectivityFileIndex($2);
}

model_spec: MODEL string coord_change_spec
{
    std::string ensight_geofile($2);
    //std::cerr << "  ENSIGHT MODEL " << ensight_geofile << found\n";
    driver.caseFile_.setGeoFileNm( ensight_geofile );
}
| MODEL INTEGER string coord_change_spec
{
    std::string ensight_geofile($3);
    driver.caseFile_.setGeoFileNm( ensight_geofile );
    int ts($2);
    driver.caseFile_.setGeoTsIdx(ts);
    std::cerr << "  ENSIGHT MODEL " << ensight_geofile << " TIMESET " << ts << " found\n";
}
| MODEL INTEGER INTEGER string coord_change_spec
{
    std::string ensight_geofile($4);
    driver.caseFile_.setGeoFileNm( ensight_geofile );
    int ts($2);
    driver.caseFile_.setGeoTsIdx(ts);
    //std::cerr << "  ENSIGHT MODEL <" << ensight_geofile << "> TIMESET <" << s << "> found\n";
}
// we may find lines like: model bla_geo.**** in this case we set the timeset to 1
| MODEL string coord_change_spec
{
    std::string ensight_geofile($2);
    driver.caseFile_.setGeoFileNm( ensight_geofile );
    driver.caseFile_.setGeoTsIdx(1);
    std::cerr << "  ENSIGHT MODEL " << ensight_geofile << " TIMESET 1 found\n";
}
| MEASURED string coord_change_spec
{
    std::string ensight_geofile($2);
    driver.caseFile_.setMGeoFileNm( ensight_geofile );
}
| MEASURED INTEGER string coord_change_spec
{
    std::string ensight_geofile($3);
    driver.caseFile_.setMGeoFileNm( ensight_geofile );
    int ts($2);
    driver.caseFile_.setGeoTsIdx(ts);
    std::cerr << "  ENSIGHT MEASURED " << ensight_geofile <<" TIMESET " << ts << " found\n";
}
| MEASURED INTEGER INTEGER string coord_change_spec
{
    std::string ensight_geofile($4);
    driver.caseFile_.setMGeoFileNm( ensight_geofile );
    int ts($2);
    driver.caseFile_.setGeoTsIdx(ts);
}
/*
// we may find lines like: model bla_geo.**** in this case we set the timeset to 1
| MEASURED string coord_change_spec
{
    std::string ensight_geofile($2);
    driver.caseFile_.setMGeoFileNm( ensight_geofile );
    driver.caseFile_.setGeoTsIdx(1);
}
*/


variable_spec: var_pre INTEGER INTEGER IDENTIFIER string
{
    long ts($2);
    //long fs($3); // FIXME
    std::string desc = $4;
    std::string fname = $5;
    if ( driver.actIt_ != nullptr ) {
        driver.actIt_->setDesc( desc );

        driver.actIt_->setFileName(fname);
        driver.actIt_->setTimeSet(ts);

        driver.caseFile_.addDataItem( *driver.actIt_ );
    } else {
        CERR << "try to add nullptr DataItem" << std::endl;
    }

    delete driver.actIt_;
    driver.actIt_ = nullptr;
}
| var_pre INTEGER IDENTIFIER string
{
    long ts($2);
    std::string desc = $3;
    std::string fname = $4;

    if ( driver.actIt_ != nullptr ) {
        driver.actIt_->setDesc( desc );
        driver.actIt_->setFileName(fname);
        driver.actIt_->setTimeSet(ts);

        driver.caseFile_.addDataItem( *driver.actIt_ );
    } else {
        CERR << "try to add nullptr DataItem" << std::endl;
    }

    delete driver.actIt_;
    driver.actIt_ = nullptr;
}
| var_pre IDENTIFIER string
{
    std::cerr << " var_pre IDENTIFIER QUOTED " << $2 << " " << $3 << std::endl;
    std::string desc = $2;
    std::string fname = $3;
    if ( driver.actIt_ != nullptr ) {
        driver.actIt_->setDesc( desc );
        driver.actIt_->setFileName(fname);

        driver.caseFile_.addDataItem( *driver.actIt_ );
    } else {
        CERR << "try to add nullptr DataItem" << std::endl;
    }

    delete driver.actIt_;
    driver.actIt_ = nullptr;
}
| var_pre IDENTIFIER DOUBLE
{
    std::cerr << " var_pre IDENTIFIER DOUBLE " << $3 << "\n";
}

var_pre :  var_type var_rela
{
    std::cerr << " var_type var_rela\n";
}

var_type: SCALAR
{
    std::cerr << "     ENSIGHT SCALAR VARIABLE ";
    driver.actIt_ = new DataItem;
    driver.actIt_->setType( DataItem::scalar );
}
| COMPLEX SCALAR
{
    std::cerr << "     ENSIGHT COMPLEX SCALAR VARIABLE - ignored ";
    driver.actIt_ = new DataItem;
    driver.actIt_->setType( DataItem::scalar );
}
| VECTOR
{
    std::cerr << "     ENSIGHT VECTOR VARIABLE ";
    driver.actIt_ = new DataItem;
    driver.actIt_->setType( DataItem::vector );
}
| COMPLEX VECTOR
{
    std::cerr << "     ENSIGHT COMPLEX VECTOR VARIABLE - ignored ";
    driver.actIt_ = new DataItem;
    driver.actIt_->setType( DataItem::vector );
}
| TENSOR_SYMM
{
    std::cerr << "     ENSIGHT SYMMETRIC TENSOR VARIABLE ";
    driver.actIt_ = new DataItem;
    driver.actIt_->setType( DataItem::tensor );
}
| TENSOR_ASYMM
{
    std::cerr << "     ENSIGHT ASYMMETRIC TENSOR VARIABLE - ignored ";
    driver.actIt_ = new DataItem;
    driver.actIt_->setType( DataItem::tensor );
}


var_rela: PER_ELEMENT
{
    std::cerr << " PER ELEMENT DATA";
    if (driver.actIt_) {
        driver.actIt_->setMapping(DataItem::PerElement);
        driver.actIt_->setMeasured(false);
    }
}
| PER_NODE
{
    std::cerr << " PER NODE DATA";
    if (driver.actIt_) {
        driver.actIt_->setMapping(DataItem::PerNode);
        driver.actIt_->setMeasured(false);
    }
}
| PER_M_NODE
{
    std::cerr << " PER NODE DATA";
    if (driver.actIt_) {
        driver.actIt_->setMapping(DataItem::PerNode);
        driver.actIt_->setMeasured(true);
    }
}
| PER_M_ELEMENT
{
    std::cerr << " PER NODE DATA";
    if (driver.actIt_) {
        driver.actIt_->setMapping(DataItem::PerElement);
        driver.actIt_->setMeasured(true);
    }
}

const_rela: PER_CASE
{
    std::cerr << " PER CASE ";
    //driver.actIt_->setMapping(DataItem::PerCase);
}
| PER_CASE_FILE
{
    std::cerr << " PER CASE FILE ";
    //driver.actIt_->setMapping(DataItem::PerCaseFile);
}
| PER_PART
{
    std::cerr << " PER PART ";
    //driver.actIt_->setMapping(DataItem::PerPart);
}

const_spec: CONSTANT const_rela IDENTIFIER number
| CONSTANT const_rela INTEGER IDENTIFIER number

scripts_spec: METADATA string
| PYTHON string

query_spec: XY_DATA string

DUMMY_YNERRS_CONSUMER {
    // just avoid a compiler warning about unused yynerrs_
    (void)yynerrs_;
}

string: IDENTIFIER
| TEXT
| QUOTED

number: INTEGER { $$ = $1; /* make flex warning go away */ }
| DOUBLE

%%
// end of rule definition
