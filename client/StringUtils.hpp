/* 
 * Author: Johannes M Dieterich
 */

#ifndef STRINGUTILS_HPP
#define	STRINGUTILS_HPP

#include <string>
#include <vector>
using namespace std;

class StringUtils {
public:
    static void trim(string &s);
    static vector<string> split(string const &input);
};

#endif	/* STRINGUTILS_HPP */

