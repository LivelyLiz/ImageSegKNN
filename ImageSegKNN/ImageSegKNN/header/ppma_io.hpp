#pragma once
#include <string>

char ch_cap ( char ch );
int i4_max ( int i1, int i2 );

bool ppma_check_data ( int xsize, int ysize, int maxrgb, int *rarray,
       int *garray, int *barray );
bool ppma_example ( int xsize, int ysize, int *rarray, int *garray, int *barray );

bool ppma_read ( std::string file_in_name, int &xsize, int &ysize, int &maxrgb,
       int **rarrary, int **garray, int **barray );
bool ppma_read_data ( std::ifstream &file_in, int xsize, int ysize, int *rarray,
       int *garray, int *barray );
bool ppma_read_header ( std::ifstream &file_in, int &xsize, int &ysize, int &maxrgb );
bool ppma_read_test ( std::string file_in_name );

bool ppma_write ( std::string file_out_name, int xsize, int ysize, int *rarray, 
      int *garray, int *barray );
bool ppma_write_data ( std::ofstream &file_out, int xsize, int ysize, int *rarray,
       int *garray, int *barray );
bool ppma_write_header(std::ofstream &file_out, std::string file_out_name, int xsize,
       int ysize, int maxrgb );
bool ppma_write_test ( std::string file_out_name );

bool s_eqi ( std::string s1, std::string s2 );
int s_len_trim ( std::string s );
void s_word_extract_first ( std::string s, std::string &s1, std::string &s2 );
