
#include <exception>
#include <string>
#include "coExport.h"
namespace covise{
class UTILEXPORT exception: public std::exception {

   public:
   exception(const std::string &what = "covise error");
   virtual ~exception() throw();

   virtual const char* what() const throw();
   virtual const char* info() const throw();

   protected:
   std::string m_info;

   private:
   std::string m_what;
};

} //covise