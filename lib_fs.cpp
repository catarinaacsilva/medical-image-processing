#include "lib_fs.h"

#include<algorithm>

std::vector<fs::path> get_directories(const fs::path& p)
{
  std::vector<fs::path> rv;
  for(auto& p : fs::directory_iterator(p)){
    if (p.is_directory()) {
      rv.push_back(p.path());
    }
  }
  std::sort(std::begin(rv), std::end(rv));
  return rv;
}

std::vector<std::filesystem::path> get_files(const fs::path& p)
{
  std::vector<std::filesystem::path> rv;
  for(auto& p : std::filesystem::directory_iterator(p)) {
    rv.push_back(p.path());
  }
  std::sort(std::begin(rv), std::end(rv));
  return rv;
}
