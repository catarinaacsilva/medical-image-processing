/**
 * @file lib_fs
 * @brief File System library
 *
 * Functions used to read the contents of a filesystem.
 * Namely read all the directories and files from a path.
 *
 * @author $Author: Catarina Silva $
 * @version $Revision: 1.0 $
 * @date $Date: 2020/01/05 $
 */

#ifndef FS_H
#define FS_H

#include <filesystem>
#include <vector>

namespace fs = std::filesystem;

std::vector<fs::path> get_directories(const fs::path&);

std::vector<std::filesystem::path> get_files(const fs::path&);

#endif
