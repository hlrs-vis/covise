/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
 **                                                           (C)2013 RUS  **
 **                                                                        **
 ** Description: Read FOAM data format                                     **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** History:                                                               **
 ** May   13	    C.Kopf  	    V1.0                                   **
 *\**************************************************************************/

//Includes copied from vistle ReadFOAM.cpp
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <cctype>

#include <cstdlib>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_no_skip.hpp>
#include <boost/spirit/include/support_multi_pass.hpp>

#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/io.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/iterator.hpp>
#include <boost/fusion/include/iterator.hpp>
#include <boost/fusion/container/vector/vector_fwd.hpp>
#include <boost/fusion/include/vector_fwd.hpp>
#include <boost/fusion/include/std_pair.hpp>

#include <boost/filesystem.hpp>

#include <boost/algorithm/string/predicate.hpp>

#include <boost/mpl/same_as.hpp>

#include "foamtoolbox.h"
#include "byteswap.h"

const size_t MaxHeaderLines = 100;

namespace bi = boost::iostreams;
namespace bs = boost::spirit;
namespace bf = boost::filesystem;

template <typename Alloc = std::allocator<char> >
struct basic_gzip_decompressor;
typedef basic_gzip_decompressor<> gzip_decompressor;

namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;

BOOST_FUSION_ADAPT_STRUCT(
    HeaderInfo,
    (std::string, version)(std::string, format)(std::string, fieldclass)(std::string, location)(std::string, object)(std::string, dimensions)(std::string, internalField)(index_t, lines))

BOOST_FUSION_ADAPT_STRUCT(
    DimensionInfo,
    (index_t, points)(index_t, cells)(index_t, faces)(index_t, internalFaces))

static bool is_directory(const bf::path &p)
{
    try
    {
        return bf::is_directory(p);
    }
    catch (const bf::filesystem_error &e)
    {
        std::cerr << "is_directory: " << e.what() << std::endl;
        return false;
    }
}

class FilteringStreamDeleter
{
public:
    FilteringStreamDeleter(bi::filtering_istream *f, std::ifstream *s)
        : filtered(f)
        , stream(s)
    {
    }

    void operator()(std::istream *f)
    {
        assert(static_cast<bi::filtering_istream *>(f) == filtered);
        delete filtered;
        delete stream;
    }

    bi::filtering_istream *filtered;
    std::ifstream *stream;
};

boost::shared_ptr<std::istream> getStreamForFile(const std::string &filename)
{

    std::ifstream *s = new std::ifstream(filename.c_str(), std::ios_base::in | std::ios_base::binary);
    if (!s->is_open())
    {
        std::cerr << "failed to open " << filename << std::endl;
        return boost::shared_ptr<std::istream>();
    }

    bi::filtering_istream *fi = new bi::filtering_istream;
    bf::path p(filename);
    if (p.extension().string() == ".gz")
    {
        fi->push(bi::gzip_decompressor());
    }
    fi->push(*s);
    return boost::shared_ptr<std::istream>(fi, FilteringStreamDeleter(fi, s));
}

boost::shared_ptr<std::istream> getStreamForFile(const std::string &dir, const std::string &basename)
{

    bf::path zipped(dir + "/" + basename + ".gz");
    if (bf::exists(zipped) && !::is_directory(zipped))
        return getStreamForFile(zipped.string());
    else
        return getStreamForFile(dir + "/" + basename);
}

bool isTimeDir(const std::string &dir)
{

    //std::cerr << "checking timedir: " << dir << std::endl;

    if (dir == ".")
        return false;
#if 1
    if (dir == "0")
        return false;
#endif

    int numdots = 0;
    for (size_t i = 0; i < dir.length(); ++i)
    {
        if (dir[i] == '.')
        {
            ++numdots;
            if (numdots > 1)
                return false;
        }
        else if (!isdigit(dir[i]) && dir[i]!='e' && dir[i]!='-')
            return false;
    }

    return true;
}

bool isProcessorDir(const std::string &dir)
{

    if (dir.find("processor") != 0)
    {

        return false;
    }

    for (size_t i = strlen("processor"); i < dir.length(); ++i)
    {

        if (!isdigit(dir[i]))
            return false;
    }

    return true;
}

bool checkMeshDirectory(CaseInfo &info, const std::string &meshdir, bool time)
{

    //std::cerr << "checking meshdir " << meshdir << std::endl;
    bf::path p(meshdir);
    if (!::is_directory(p))
    {
        std::cerr << meshdir << " is not a directory" << std::endl;
        return false;
    }

    bool havePoints = false;
    std::map<std::string, std::string> meshfiles;
    for (bf::directory_iterator it(p);
         it != bf::directory_iterator();
         ++it)
    {
        bf::path ent(*it);
        std::string stem = ent.stem().string();
        std::string ext = ent.extension().string();
        if (stem == "points" || stem == "faces" || stem == "owner" || stem == "neighbour")
        {
            if (::is_directory(*it) || (!ext.empty() && ext != ".gz"))
            {
                std::cerr << "ignoring " << *it << std::endl;
            }
            else
            {
                meshfiles[stem] = bf::path(*it).string();
                if (stem == "points")
                    havePoints = true;
            }
        }
    }
    // 
    if (meshfiles.size() == 4)
    {
        if (time)
        {
            info.varyingGrid = true;
            info.varyingCoords = true;
        }
        return true;
    }
    if (meshfiles.size() == 1 && time && havePoints)
    {
        info.varyingGrid = false;
        info.varyingCoords = true;
        return true;
    }
	// this usually never occurs:
    if (meshfiles.size() == 3 && time && !havePoints)
    {
        info.varyingGrid = true;
        info.varyingCoords = false;
        return true;
    }

    std::cerr << "did not find all of points, faces, owner and neighbour files" << std::endl;
    return false;
}

bool checkLagrangianDirectory(CaseInfo &info, std::string lagdir, bool time)
{
    if (info.lagrangiandir.empty())
    {
        info.lagrangiandir = "dsmc";
    }
    lagdir = lagdir + "/" + info.lagrangiandir;
    //std::cerr << "checking lagdir " << lagdir << std::endl;
    bf::path p(lagdir);
    if (!::is_directory(p))
    {
        std::cerr << lagdir << " is not a directory" << std::endl;
        return false;
    }

    bool havePositions = false;
    std::map<std::string, std::string> meshfiles;
    for (bf::directory_iterator it(p);
         it != bf::directory_iterator();
         ++it)
    {
        bf::path ent(*it);
        std::string stem = ent.stem().string();
        std::string ext = ent.extension().string();
        if (::is_directory(*it) || (!ext.empty() && ext != ".gz"))
        {
            if (stem == "positions")
            {
                std::cerr << "ignoring " << *it << std::endl;
            }
        }
        else
        {
            meshfiles[stem] = bf::path(*it).string();
            if (stem == "positions")
                havePositions = true;
            else
                ++info.particleFields[stem];
        }
    }

    if (havePositions)
        info.hasParticles = true;

    if (!havePositions)
    {
        std::cerr << "did not find positions in " << lagdir << std::endl;
        return false;
    }

    return true;
}

bool checkSubDirectory(CaseInfo &info, const std::string &timedir, bool time)
{

    bf::path dir(timedir);
    if (!bf::exists(dir))
    {
        std::cerr << "timestep directory " << timedir << " does not exist" << std::endl;
        return false;
    }
    if (!::is_directory(timedir))
    {
        std::cerr << "timestep directory " << timedir << " is not a directory" << std::endl;
        return false;
    }

    for (bf::directory_iterator it(dir);
         it != bf::directory_iterator();
         ++it)
    {
        bf::path p(*it);
        if (::is_directory(*it))
        {
            std::string name = p.filename().string();
            if (name == "polyMesh")
            {
                if (!checkMeshDirectory(info, p.string(), time))
                    return false;
            }
            if (name == "lagrangian")
            {
                if (!checkLagrangianDirectory(info, p.string(), time))
                    return false;
            }
        }
        else
        {
            std::string stem = p.stem().string();
            if (!boost::algorithm::ends_with(stem, "Dict") && !boost::algorithm::ends_with(stem, "Properties"))
            {
                if (time)
                {
                    ++info.varyingFields[stem];
                    //std::cerr << "counting timedir : " << timedir << std::endl;
                    //std::cerr << "added varying field :" << stem << std::endl;
                }
                else
                    ++info.constantFields[stem];
            }
        }
    }

    return true;
}


bool checkPolyMeshDirContent(CaseInfo &info)
{
	// start out with 
	std::stringstream s;
    if (info.numblocks>0)
		s << "/processor0" << "/";
    std::string basedir = info.casedir;
	basedir += s.str(); 
	std::string fullMeshDir = info.constantdir;

	for (std::map<double, std::string>::iterator it = info.timedirs.begin(); it != info.timedirs.end(); ++it)
	{
		std::string currentTimeDir= basedir + it->second;
		checkSubDirectory(info, currentTimeDir, true);
		if (info.varyingGrid)
		{
			fullMeshDir = it->second;
		}
		info.completeMeshDirs[it->first]=fullMeshDir;
        //std::cerr << "Full Mesh for timestep " << it->second << " found at time = " << fullMeshDir << std::endl;
	}

    return true;
}



bool checkCaseDirectory(CaseInfo &info, const std::string &casedir, bool compare, bool exact)
{

    std::cerr << "reading casedir: " << casedir << std::endl;

    bf::path dir(casedir);
    if (!bf::exists(dir))
    {
        std::cerr << "case directory " << casedir << " does not exist" << std::endl;
        return false;
    }
    if (!::is_directory(casedir))
    {
        std::cerr << "case directory " << casedir << " is not a directory" << std::endl;
        return false;
    }

    int num_processors = 0;
    for (bf::directory_iterator it(dir);
         it != bf::directory_iterator();
         ++it)
    {
        if (::is_directory(*it))
        {
            if (isProcessorDir(bf::basename(it->path())))
                ++num_processors;
        }
    }

    if (!compare && num_processors > 0)
    {
        info.numblocks = num_processors;
    }

    if (compare && num_processors > 0)
    {
        std::cerr << "found processor subdirectory in processor directory" << std::endl;
        return false;
    }

    if (num_processors > 0)
    {
        bool result = checkCaseDirectory(info, casedir + "/processor0", false, exact);
        if (!result)
        {
            std::cerr << "failed to read case directory for processor 0" << std::endl;
            return false;
        }

        if (exact)
        {
            for (int i = 1; i < num_processors; ++i)
            {
                std::stringstream s;
                s << casedir << "/processor" << i;
                bool result = checkCaseDirectory(info, s.str(), true, exact);
                if (!result)
                    return false;
            }
        }

        return true;
    }

    index_t num_timesteps = 0;
    for (bf::directory_iterator it(dir);
         it != bf::directory_iterator();
         ++it)
    {
        if (::is_directory(*it))
        {
            std::string bn = it->path().filename().string();
            if (isTimeDir(bn))
            {
                double t = atof(bn.c_str());
                //std::cerr << bn << "is a time directory, " << t  << std::endl;
                //if (t >= mintime && t <= maxtime)
                {
                    ++num_timesteps;
                    if (compare)
                    {
                        if (info.timedirs.find(t) == info.timedirs.end())
                        {
                            --num_timesteps;
                            //                     std::cerr << "timestep " << bn << " not available on all processors" << std::endl;
                            //                     return false;
                        }
                    }
                    else
                    {
                        info.timedirs[t] = bn;
                    }
                }
            }
            else
            {
                if (bn == "constant")
                {
                    info.constantdir = bn;
                }
                else if (bn == "0" && info.constantdir.empty())
                {
                    info.constantdir = bn;
                }
            }
        }
    }

    bool varyingChecked = false, constantChecked = false;
    for (bf::directory_iterator it(dir);
         it != bf::directory_iterator();
         ++it)
    {
        if (::is_directory(*it))
        {
            std::string bn = it->path().filename().string();
			//std::cerr << "directory :" << bn << std::endl;
            if (isTimeDir(bn) && !varyingChecked)
            {
                double t = atof(bn.c_str());
                if (info.timedirs.find(t) != info.timedirs.end())
                {
                    bool result = checkSubDirectory(info, it->path().string(), true);
                    if (!result)
                        return false;
                    varyingChecked = true;
                }
            }
            else if (bn == info.constantdir)
            {
                bool result = checkSubDirectory(info, it->path().string(), false);
                if (!result)
                    return false;
                constantChecked = true;
            }

            if (!exact && constantChecked && varyingChecked)
                return true;
        }
    }

    if (compare && num_timesteps != info.timedirs.size())
    {
        std::cerr << "not all timesteps available on all processors" << std::endl;
        return false;
    }

    return true;
}

bool checkFields(std::map<std::string, int> &fields, int nRequired, bool exact)
{
    bool ignored = false;
    for (std::map<std::string, int>::iterator it = fields.begin(), next;
         it != fields.end();
         it = next)
    {
        next = it;
        ++next;
        std::cerr << "  " << it->first << ": " << it->second;
        if (exact && it->second != nRequired)
        {
            ignored = true;
            std::cerr << " (ignored)";
            fields.erase(it->first);
        }
    }
    std::cerr << std::endl;
    return !ignored;
}

CaseInfo getCaseInfo(const std::string &casedir, bool exact)
{

    CaseInfo info;
    info.casedir = casedir;
    info.valid = checkCaseDirectory(info, casedir, false, exact);

    std::cerr << " "
              << "casedir: " << casedir << " " << std::endl
              << "Number of processors: " << info.numblocks << std::endl
              << "Number of time directories found: " << info.timedirs.size() << std::endl
              << "Number of fields: " << info.constantFields.size() + info.varyingFields.size() << std::endl;

    int np = info.numblocks > 0 ? info.numblocks : 1;
    std::cerr << "  constant:";
    checkFields(info.constantFields, np, exact);

    std::cerr << "  varying: ";
    checkFields(info.varyingFields, np * info.timedirs.size(), exact);

    if (info.hasParticles)
    {
        std::cerr << "  lagrangian from " << info.lagrangiandir << ": ";
        checkFields(info.particleFields, np * info.timedirs.size(), exact);
    }
    else
    {
        std::cerr << "  no lagrangian data" << std::endl;
    }

    return info;
}

//Boost Spirit parser definitions

std::string getFoamHeader(std::istream &stream)
{
    size_t internalFieldLine = MaxHeaderLines; 
    std::string header;
    for (size_t i = 0; (i < MaxHeaderLines && i< internalFieldLine+2); ++i)
    {
        int c = stream.peek();
        if (c == '(')
           break;
        std::string line;
        std::getline(stream, line);
        if (boost::algorithm::starts_with(line, "internalField"))
        {
            //std::cerr << "Current line is internalField, we need one line more...  "<< std::endl;
            internalFieldLine=i;
        }
        header.append(line);
        header.append("\n");
    }
    return header;
}

std::string getFieldHeader(std::istream &stream)
{
    std::string header;
    for (size_t i = 0; i < MaxHeaderLines; ++i)
    {
        int c = stream.peek();
        if (c == '(')
           break;
        std::string line;
        std::getline(stream, line);
        if (boost::algorithm::starts_with(line, "("))
        {
            break;
        }
        header.append(line);
        header.append("\n");
    }
    return header;
}

//Skipper - skipping unimportant data when parsing FOAM headers
template <typename Iterator>
struct headerSkipper : qi::grammar<Iterator>
{

    headerSkipper()
        : headerSkipper::base_type(start)
    {

        start = ascii::space
                | "/*" >> *(ascii::char_ - "*/") >> "*/"
                | "//" >> *(ascii::char_ - qi::eol) >> qi::eol
                | "FoamFile" >> *ascii::space >> '{' >> *(ascii::char_ - qi::eol) >> qi::eol
                | "note" >> *(ascii::char_ - qi::eol) >> qi::eol
                | '}';
    }

    qi::rule<Iterator> start;
};

template <typename Iterator>
struct FileHeaderParser : qi::grammar<Iterator, HeaderInfo(), headerSkipper<Iterator> >
{

    FileHeaderParser()
        : FileHeaderParser::base_type(start)
    {

        start = version
                >> format
                >> fieldclass
                >> location
                >> object
                >> -dimensions
                >> -(internalField|boundaryField)
                >> -lines;

        version = "version" >> +(ascii::char_ - ';') >> ';';
        format = "format" >> +(ascii::char_ - ';') >> ';';
        fieldclass = "class" >> +(ascii::char_ - ';') >> ';';
        location = "location" >> qi::lit('"') >> +(ascii::char_ - '"') >> '"' >> ';';
        object = "object" >> +(ascii::char_ - ';') >> ';';
        dimensions = "dimensions" >> qi::lexeme['[' >> +(ascii::char_ - ']') >> ']' >> ';'];
        nonUniformField= "nonuniform">> qi::lexeme[+(ascii::char_ - qi::eol) >> qi::eol];
        //uniformField= "uniform">> qi::lexeme[+(ascii::char_ - qi::eol) >> qi::eol];
        uniformField= "uniform">> +(ascii::char_ - ';') >> ';';
        internalField = "internalField" >> (uniformField|nonUniformField);
        boundaryField = "boundaryField" >> qi::lexeme[+(ascii::char_ - qi::eol) >> qi::eol];
        lines = qi::int_;
    }

    qi::rule<Iterator, HeaderInfo(), headerSkipper<Iterator> > start;
    qi::rule<Iterator, std::string(), headerSkipper<Iterator> > version;
    qi::rule<Iterator, std::string(), headerSkipper<Iterator> > format;
    qi::rule<Iterator, std::string(), headerSkipper<Iterator> > fieldclass;
    qi::rule<Iterator, std::string(), headerSkipper<Iterator> > location;
    qi::rule<Iterator, std::string(), headerSkipper<Iterator> > object;
    qi::rule<Iterator, std::string(), headerSkipper<Iterator> > dimensions;
    qi::rule<Iterator, std::string(), headerSkipper<Iterator> > internalField;
    qi::rule<Iterator, std::string(), headerSkipper<Iterator> > boundaryField;
    qi::rule<Iterator, std::string(), headerSkipper<Iterator> > uniformField;
    qi::rule<Iterator, std::string(), headerSkipper<Iterator> > nonUniformField;
    qi::rule<Iterator, int(), headerSkipper<Iterator> > lines;
};

template <typename Iterator>
struct FieldHeaderParser : qi::grammar<Iterator, HeaderInfo(), headerSkipper<Iterator> >
{
    FieldHeaderParser()
        : FieldHeaderParser::base_type(start)
    {

        start = -internalField
                >> -boundaryField
                >> lines;

        dimensions = "dimensions" >> qi::lexeme['[' >> +(ascii::char_ - ']') >> ']' >> ';'];
        fieldType = internalField | boundaryField;
        internalField = "internalField" >> qi::lexeme[+(ascii::char_ - qi::eol) >> qi::eol];
        boundaryField = "boundaryField" >> qi::lexeme[+(ascii::char_ - qi::eol) >> qi::eol];
        lines = qi::int_;
    }

    qi::rule<Iterator, HeaderInfo(), headerSkipper<Iterator> > start;
    qi::rule<Iterator, std::string(), headerSkipper<Iterator> > dimensions;
    qi::rule<Iterator, std::string(), headerSkipper<Iterator> > fieldType;
    qi::rule<Iterator, std::string(), headerSkipper<Iterator> > internalField;
    qi::rule<Iterator, std::string(), headerSkipper<Iterator> > boundaryField;
    qi::rule<Iterator, int(), headerSkipper<Iterator> > lines;
};

HeaderInfo readFoamHeader(std::istream &stream)
{
    struct FileHeaderParser<std::string::iterator> headerParser;
    struct FieldHeaderParser<std::string::iterator> fieldHeaderParser;
    struct headerSkipper<std::string::iterator> headerSkipper;

    HeaderInfo info;
    info.header = getFoamHeader(stream);
    std::string fileheader = info.header;

    info.valid = qi::phrase_parse(fileheader.begin(), fileheader.end(),
                                  headerParser, headerSkipper, info);

    if (!info.valid)
    {
        std::cerr << "parsing FOAM file header failed" << std::endl;

        std::cerr << "================================================" << std::endl;
        std::cerr << info.header << std::endl;
        std::cerr << "================================================" << std::endl;
    }
#if 0
    else
    {
        std::string fieldheader = getFieldHeader(stream);
        std::cerr << "field header" << fieldheader << "<<< end" << std::endl;
        info.valid = qi::phrase_parse(fieldheader.begin(), fieldheader.end(),
                                     fieldHeaderParser, headerSkipper, info);
        if (!info.valid)
        {
            std::cerr << "parsing FOAM field header failed" << std::endl;

            std::cerr << "================================================" << std::endl;
            std::cerr << fieldheader << std::endl;
            std::cerr << "================================================" << std::endl;
        }
    }
#endif

    return info;
}

//Skipper - skipping everything but the dimensions in the owners file Header
template <typename Iterator>
struct dimSkipper : qi::grammar<Iterator>
{

    dimSkipper()
        : dimSkipper::base_type(start)
    {

        start = ascii::space
                | "/*" >> *(ascii::char_ - "*/") >> "*/"
                | "//" >> *(ascii::char_ - qi::eol) >> qi::eol
                | "version" >> *(ascii::char_ - qi::eol) >> qi::eol
                | "location" >> *(ascii::char_)
                | ~ascii::char_("0-9");
    }

    qi::rule<Iterator> start;
};

//Parser that Reads the size of the domain (nCells, nPoints, nFaces, nInternalFaces) from the owners header (use together with dimSkipper)
template <typename Iterator>
struct dimParser : qi::grammar<Iterator, DimensionInfo(),
                               dimSkipper<Iterator> >
{

    dimParser()
        : dimParser::base_type(start)
    {

        start = term >> term >> term >> term;

        term = qi::int_;
    }

    qi::rule<Iterator, DimensionInfo(), dimSkipper<Iterator> > start;
    qi::rule<Iterator, index_t(), dimSkipper<Iterator> > term;
};

//Skipper - skipping FOAM Headers and unimportant data when parsing Boundary files
template <typename Iterator>
struct skipper : qi::grammar<Iterator>
{

    skipper()
        : skipper::base_type(start)
    {

        start = ascii::space
                | "/*" >> *(ascii::char_ - "*/") >> "*/"
                | "//" >> *(ascii::char_ - qi::eol) >> qi::eol
                | "FoamFile" >> *ascii::space >> '{' >> *(ascii::char_ - '}') >> '}'
                | "dimensions" >> *(ascii::char_ - qi::eol) >> qi::eol
                | "internalField" >> *(ascii::char_ - qi::eol) >> qi::eol;
    }

    qi::rule<Iterator> start;
};

template <typename Iterator>
struct BoundaryParser
    : qi::grammar<Iterator, std::vector<std::pair<std::string, std::map<std::string, std::string> > >(),
                  skipper<Iterator> >
{
    typedef std::string name_type;
    typedef std::string value_type;
    typedef std::string key_type;
    typedef std::map<key_type, value_type> boundary_type;
    typedef std::vector<std::pair<name_type, boundary_type> > boundary_list;

    BoundaryParser()
        : BoundaryParser::base_type(start)
    {
        start = qi::omit[qi::int_] >> '(' >> +mapmap >> ')';
        mapmap = +(qi::char_ - '{') >> '{' >> entrymap >> '}';
        entrymap = +pair;
        pair = qi::lexeme[+qi::char_("a-zA-Z")] >> +(qi::char_ - ';') >> ';';
    }

    qi::rule<Iterator, boundary_list(), skipper<Iterator> > start;
    qi::rule<Iterator, std::pair<name_type, boundary_type>(), skipper<Iterator> > mapmap;

    qi::rule<Iterator, boundary_type(), skipper<Iterator> > entrymap;
    qi::rule<Iterator, std::pair<key_type, value_type>(), skipper<Iterator> > pair;
};

Boundaries loadBoundary(const std::string &meshdir)
{

    Boundaries bounds;
    boost::shared_ptr<std::istream> stream = getStreamForFile(meshdir, "boundary");
    if (!stream)
        return bounds;

    typedef std::istreambuf_iterator<char> base_iterator_type;
    typedef bs::multi_pass<base_iterator_type> forward_iterator_type;
    forward_iterator_type fwd_begin = bs::make_default_multi_pass(base_iterator_type(*stream));
    forward_iterator_type fwd_end;

    struct skipper<forward_iterator_type> skipper;
    typedef BoundaryParser<forward_iterator_type> Parser;
    Parser p;

    Parser::boundary_list boundaries;
    bounds.valid = qi::phrase_parse(fwd_begin, fwd_end, p, skipper, boundaries);

    index_t index = 0;
    for (Parser::boundary_list::iterator top = boundaries.begin();
            top != boundaries.end();
            ++top)
    {

        std::string name = top->first;
#if 0
      std::cout << name << ":" << std::endl;
      std::map<std::string, std::string>::iterator i;
      for (i = top->second.begin(); i != top->second.end(); i ++) {
         std::cout << "    " << i->first << " => " << i->second << std::endl;
      }
#endif
        const Parser::boundary_type &cur = top->second;
        Parser::boundary_type::const_iterator nFaces = cur.find("nFaces");
        Parser::boundary_type::const_iterator startFace = cur.find("startFace");
        Parser::boundary_type::const_iterator type = cur.find("type");
        Parser::boundary_type::const_iterator myProc = cur.find("myProcNo");
        Parser::boundary_type::const_iterator neighbor = cur.find("neighbProcNo");
        if (nFaces != cur.end() && startFace != cur.end() && type != cur.end())
        {
            std::string t = type->second;
            index_t n = atol(nFaces->second.c_str());
            index_t s = atol(startFace->second.c_str());
            Boundary b(name, s, n, t, index);
            if (myProc != cur.end())
                b.myProc = atol(myProc->second.c_str());
            if (neighbor != cur.end())
                b.neighborProc = atol(neighbor->second.c_str());
            bounds.addBoundary(b);
            ++index;
        }
    }

    return bounds;
}

namespace
{

const endianness foam_endian = little_endian;

typedef double FoamFloat;
typedef uint32_t FoamIndex;

template <typename T>
struct on_disk;

template<>
struct on_disk<float>
{
    typedef FoamFloat type;
};

template<>
struct on_disk<double>
{
    typedef FoamFloat type;
};

template<>
struct on_disk<int>
{
    typedef FoamIndex type;
};

template<>
struct on_disk<unsigned>
{
    typedef FoamIndex type;
};

template<>
struct on_disk<long>
{
    typedef FoamIndex type;
};

template<>
struct on_disk<unsigned long>
{
    typedef FoamIndex type;
};

}

// requires 
#define expect(c) \
    do \
    { \
        char paren = 0; \
        stream.read(&paren, 1); \
        if (paren != c) { \
            std::cerr << __FILE__ << ":" << __LINE__ << ": expected '" << char(c) << "', got '" << char(paren) << "'" << std::endl; \
            return false; \
        } \
    } \
    while(false)


template <typename T>
std::istream &operator>>(std::istream &stream, std::vector<T> &vec)
{

    size_t n;
    stream >> n;
    stream.ignore(std::numeric_limits<std::streamsize>::max(), '(');
    vec.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        T val;
        stream >> val;
        vec.push_back(val);
    }
    stream.ignore(std::numeric_limits<std::streamsize>::max(), ')');
    return stream;
}


template <typename D>
bool readArrayChunkBinary(std::istream &stream, D *buf, const size_t num)
{
    stream.read(reinterpret_cast<char *>(buf), sizeof(buf[0])*num);
    if (!stream.good())
        return false;
    if (foam_endian != host_endian)
    {
       for (size_t i=0; i<num; ++i)
       {
          buf[i] = byte_swap<foam_endian, host_endian, D>(buf[i]);
       }
    }
    return true;
}

static const size_t bufsiz = 16384;

template <typename T, typename D>
bool readVectorArrayBinary(std::istream &stream, T *x, T *y, T *z, const size_t lines)
{
    std::vector<D> buf(3*bufsiz);
    for (size_t i=0; i<lines; i+=bufsiz)
    {
        const size_t nread = i+bufsiz <= lines ? bufsiz : lines-i;
        if (!readArrayChunkBinary(stream, &buf[0], 3*nread))
            return false;
        for (size_t j=0; j<nread; ++j)
        {
            x[i+j] = buf[j*3+0];
            y[i+j] = buf[j*3+1];
            z[i+j] = buf[j*3+2];
        }
    }
    return stream.good();
}

template <typename T>
bool readVectorArrayAscii(std::istream &stream, T *x, T *y, T *z, const size_t lines)
{
    expect('\n');
    for (size_t i = 0; i < lines; ++i)
    {
        stream.ignore(std::numeric_limits<std::streamsize>::max(), '(');
        stream >> x[i] >> y[i] >> z[i];
        stream.ignore(std::numeric_limits<std::streamsize>::max(), ')');
    }
    expect('\n');

    return stream.good();
}

template <>
bool readVectorArrayAscii(std::istream &stream, float *x, float *y, float *z, const size_t lines)
{
    expect('\n');
    for (size_t i = 0; i < lines; ++i)
    {
        stream.ignore(std::numeric_limits<std::streamsize>::max(), '(');
        double vx, vy, vz;
        stream >> vx >> vy >> vz;
        x[i] = vx;
        y[i] = vy;
        z[i] = vz;
        stream.ignore(std::numeric_limits<std::streamsize>::max(), ')');
    }
    expect('\n');

    return stream.good();
}

template <typename T>
bool readVectorArray(const HeaderInfo &info, std::istream &stream, T *x, T *y, T *z, const size_t lines)
{
    bool ok = true;
    expect('(');
    if (info.format == "binary")
    {
        ok = readVectorArrayBinary<T, typename on_disk<T>::type>(stream, x, y, z, lines);
    }
    else
    {
        ok = readVectorArrayAscii<T>(stream, x, y, z, lines);
    }
    expect(')');

    return ok && stream.good();
}

template <typename T>
bool readArrayBinary(std::istream &stream, T *p, const size_t lines)
{
    typedef typename on_disk<T>::type D;
    if (boost::is_same<T, D>::value)
    {
       return readArrayChunkBinary(stream, p, lines);
    }
    else
    {
       std::vector<D> buf(bufsiz);
       for (size_t i=0; i<lines; i+=bufsiz)
       {
          const size_t nread = i+bufsiz <= lines ? bufsiz : lines-i;
          if (!readArrayChunkBinary(stream, &buf[0], nread))
             return false;
          for (size_t j=0; j<nread; ++j)
          {
             p[i+j] = buf[j];
          }
       }
    }
    return true;
}

template <typename T>
bool readListBinary(std::istream &stream, std::vector<T> &vec)
{

    size_t n;
    stream >> n;
    stream.ignore(std::numeric_limits<std::streamsize>::max(), '(');
    vec.resize(n);
    readArrayBinary(stream, &vec[0], n);
    stream.ignore(std::numeric_limits<std::streamsize>::max(), ')');
    return stream.good();
}

template <typename T>
bool readArrayAscii(std::istream &stream, T *p, const size_t lines)
{
    for (size_t i = 0; i < lines; ++i)
    {
        stream >> p[i];
        if (!stream.good())
        {
           std::cerr << "readArrayAscii: failure at element " << i << " of " << lines << std::endl;
           return false;
        }
    }
    expect('\n');
    return stream.good();
}

template <>
bool readArrayAscii(std::istream &stream, float *p, const size_t lines)
{
    for (size_t i = 0; i < lines; ++i)
    {
        double val;
        stream >> val;
        p[i] = val;
        if (!stream.good())
        {
           std::cerr << "readArrayAscii: failure at element " << i << " of " << lines << std::endl;
           return false;
        }
    }
    expect('\n');
    return stream.good();
}

template <typename T>
bool readIndexListArrayBinary(std::istream &stream, std::vector<T> *p, const size_t lines)
{
    for (size_t i = 0; i < lines; ++i)
    {
        readListBinary(stream, p[i]);
        if (!stream.good())
        {
           std::cerr << "readIndexListArrayBinary: failure at element " << i << " of " << lines << std::endl;
           return false;
        }
    }
    expect('\n');
    return stream.good();
}

template <typename T>
bool readArray(const HeaderInfo &info, std::istream &stream, T *p, const size_t lines)
{
    expect('(');
    if (info.format == "binary")
    {
        if (!readArrayBinary(stream, p, lines))
           return false;
    }
    else
    {
        if(!readArrayAscii(stream, p, lines))
           return false;
    }
    expect(')');
    return true;
}

bool readIndexArray(const HeaderInfo &info, std::istream &stream, index_t *p, const size_t lines)
{
    if (!stream.good()) {
       std::cerr << "readIndexArray: stream not good initially" << std::endl;
       return false;
    }
    assert(stream.good());
    return readArray<index_t>(info, stream, p, lines);
}

template <typename T>
bool readIndexCompactListArrayBinary(std::istream &stream, std::vector<T> *p, const size_t lines)
{
    expect('(');
    std::vector<FoamIndex> faceIndex(lines);
    if (!readArrayBinary<FoamIndex>(stream, &faceIndex[0], lines))
    {
        std::cerr << "readIndexCompactListArrayBinary: readArrayBinary<FoamIndex> for index array failed" << std::endl;
        return false;
    }
    expect(')');
    expect('\n');

    std::string line;
    std::getline(stream, line);
    const size_t totalIndices = atol(line.c_str());
    if (faceIndex[lines-1] != totalIndices)
    {
        std::cerr << "readIndexCompactListArrayBinary: expecting last faceIndex[" << lines-1 << "] == " << totalIndices << std::endl;
        return false;
    }

    expect('(');
    for (size_t i=0; i<lines-1; ++i)
    {
       size_t n = faceIndex[i+1] - faceIndex[i];
       p[i].resize(n);
       if (!readArrayBinary<index_t>(stream, &p[i][0], n))
       {
           std::cerr << "readIndexCompactListArrayBinary: readArrayBinary<index_t> failed to read index list " << i << std::endl;
           return false;
       }
    }
    expect(')');
    p[lines-1].resize(0);

    return true;
}

bool readIndexListArray(const HeaderInfo &info, std::istream &stream, std::vector<index_t> *p, const size_t lines)
{
    if (!stream.good()) {
       std::cerr << "readIndexListArray: stream not good initially" << std::endl;
       return false;
    }
   assert(stream.good());
   if (info.format == "binary")
   {
      if (info.fieldclass == "faceCompactList")
      {
         return readIndexCompactListArrayBinary<index_t>(stream, p, lines);
      }
      else if (info.fieldclass == "faceList")
      {
          expect('(');
          if(!readIndexListArrayBinary<index_t>(stream, p, lines))
              return false;
          expect(')');
      }
      else
      {
          std::cerr << "readIndexListArray: unsupported class '" << info.fieldclass << "'" << std::endl;
          return false;
      }
   }
   else
   {
      expect('(');
      if(!readArrayAscii<std::vector<index_t> >(stream, p, lines))
         return false;
      expect(')');
   }
   return true;
}

bool readFloatArray(const HeaderInfo &info, std::istream &stream, scalar_t *p, const size_t lines)
{
    if (!stream.good()) {
       std::cerr << "readFloatArray: stream not good initially" << std::endl;
       return false;
    }
    assert(stream.good());
    return readArray<scalar_t>(info, stream, p, lines);
}

bool readFloatVectorArray(const HeaderInfo &info, std::istream &stream, scalar_t *x, scalar_t *y, scalar_t *z, const size_t lines)
{
    if (!stream.good()) {
       std::cerr << "readFloatVectorArray: stream not good initially" << std::endl;
       return false;
    }
    assert(stream.good());
    return readVectorArray(info, stream, x, y, z, lines);
}

DimensionInfo parseDimensions(std::string header)
{
    struct dimParser<std::string::iterator> dimParser;
    struct dimSkipper<std::string::iterator> dimSkipper;
    DimensionInfo info;
    info.valid = qi::phrase_parse(header.begin(), header.end(), dimParser, dimSkipper, info);
    return info;
}

index_t findVertexAlongEdge(const index_t point,
                            const index_t homeface, //the vertex we are looking for is certainly not on this face
                            const std::vector<index_t> &cellfaces,
                            const std::vector<std::vector<index_t> > &faces)
{
    //find the other 2 faces that include the vertex with index "point"
    index_t pointfaces[2];
    int idx = 0;
    for (index_t i = 0; i < cellfaces.size(); i++)
    {
        if (cellfaces[i] != homeface)
        {
            const std::vector<index_t> &face = faces[cellfaces[i]];
            for (index_t j = 0; j < face.size(); j++)
            {
                if (face[j] == point)
                {
                    pointfaces[idx++] = cellfaces[i];
                    break;
                }
            }
            if (idx >= 2)
                break;
        }
    }

    //use these faces to find the vertex that is along the edge formed by these two faces
    const std::vector<index_t> &a = faces[pointfaces[0]];
    const std::vector<index_t> &b = faces[pointfaces[1]];
    for (index_t i = 0; i < a.size(); i++)
    {
        if (a[i] == point)
            continue;
        for (index_t j = 0; j < b.size(); j++)
        {
            if (a[i] == b[j])
            {
                return a[i];
            }
        }
    }

    return -1;
}

bool isPointingInwards(index_t face,
                       index_t cell,
                       index_t ninternalFaces,
                       const std::vector<index_t> &owners,
                       const std::vector<index_t> &neighbors)
{
    // OpenFOAM uses the right Hand Rule for numbering its faces.
    // check if the normal vector of a face is pointing into the current cell or the neighbouring cell
    // (in OpenFOAM it always points into the neighbouring cell with the higher index)
    if (face >= ninternalFaces)
    {
        // if face is bigger than the number of internal faces
        // then face is a boundary-face and normal vector goes out of the domain by default
        return false;
    }
    else
    {
        index_t o = owners[face];
        index_t n = neighbors[face];
        index_t j = o==cell ? n : o;
        // cell is the index of current cell and j is index of other cell sharing the same face
        // if index of cell is higher than index of the "next door" cell
        // then normal vector points inwards else outwards
        if (cell > j)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
}

vertex_set getVerticesForCell(
    const std::vector<index_t> &cellfaces,
    const std::vector<std::vector<index_t> > &faces)
{

    vertex_set cellvertices;
    for (index_t i = 0; i < cellfaces.size(); i++)
    {
        for (index_t j = 0; j < faces[cellfaces[i]].size(); j++)
        {
            cellvertices.insert(faces[cellfaces[i]][j]);
        }
    }
    return cellvertices;
}

template <typename F, typename I>
bool readParticleArrayAscii(std::istream &stream, F *x, F *y, F *z, I *cell, const size_t lines)
{
    expect('\n');
    for (size_t i = 0; i < lines; ++i)
    {
        F vx, vy, vz;
        I vc;
        stream.ignore(std::numeric_limits<std::streamsize>::max(), '(');
        stream >> vx >> vy >> vz;
        stream.ignore(std::numeric_limits<std::streamsize>::max(), ')');
        stream >> vc;
        if (x) x[i] = vx;
        if (y) y[i] = vy;
        if (z) z[i] = vz;
        if (cell) cell[i] = vc;
    }
    expect('\n');

    return stream.good();
}

template <typename F, typename I>
bool readParticleArrayBinary(std::istream &stream, F *x, F *y, F *z, I *cell, const size_t lines)
{
    typedef typename on_disk<F>::type Float;
    typedef typename on_disk<I>::type Index;
    std::vector<Float> fbuf(3);
    std::vector<Index> ibuf(4);
    expect('\n');
    for (size_t i=0; i<lines; ++i)
    {
        expect('(');
        if (!readArrayChunkBinary(stream, &fbuf[0], fbuf.size()))
            return false;
        if (!readArrayChunkBinary(stream, &ibuf[0], ibuf.size()))
            return false;
        if (x) x[i] = fbuf[0];
        if (y) y[i] = fbuf[1];
        if (z) z[i] = fbuf[2];
        if (cell) cell[i] = ibuf[0];
        expect(')');
        expect('\n');
    }

    return stream.good();
}

bool readParticleArray(const HeaderInfo &info, std::istream &stream, scalar_t *x, scalar_t *y, scalar_t *z, index_t *cell, const size_t lines)
{
    if (!stream.good()) {
        std::cerr << "readParticleArray: stream not good initially" << std::endl;
        return false;
    }
    assert(stream.good());
    expect('(');
    if (info.format == "binary")
    {
        if(!readParticleArrayBinary<scalar_t, index_t>(stream, x, y, z, cell, lines))
            return false;
    }
    else
    {
        if(!readParticleArrayAscii<scalar_t, index_t>(stream, x, y, z, cell, lines))
            return false;
    }
    expect(')');
    return stream.good();
}
