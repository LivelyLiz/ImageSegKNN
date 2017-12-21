/// @file
////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Copyright (C) 2017/18      Christian Lessig, Otto-von-Guericke Universitaet Magdeburg
///
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  module     : tutorial 9
///
///  author     : lessig@isg.cs.ovgu.de
///
///  project    : GPU Programming
///
///  description: image
///
////////////////////////////////////////////////////////////////////////////////////////////////////

// includes, project
#include "header/image.h"

// includes, system
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cmath>

////////////////////////////////////////////////////////////////////////////////////////////////////
// constructor, user
////////////////////////////////////////////////////////////////////////////////////////////////////
Image::Image( int num_rows, int num_cols) :
  n_rows( num_rows),
  n_cols( num_cols),
  data( num_rows * num_cols, 0.0)
{ }

////////////////////////////////////////////////////////////////////////////////////////////////////
// constructor, user
////////////////////////////////////////////////////////////////////////////////////////////////////
Image::Image( const std::string& fname) :
  n_rows( 0),
  n_cols( 0),
  data( 0)
{
  read( fname);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//! constructor, copy
Image::Image( const Image& other) :
  n_rows( other.n_rows),
  n_cols( other.n_cols),
  data( other.data)
{ }

////////////////////////////////////////////////////////////////////////////////////////////////////
// set pixel
////////////////////////////////////////////////////////////////////////////////////////////////////
const float&
Image::operator()( const int& i_row, const int& i_col) const {

  assert( i_row < n_rows);
  assert( i_col < n_cols);

  return data[i_row * n_cols + i_col];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// set pixel
////////////////////////////////////////////////////////////////////////////////////////////////////
float&
Image::operator()( const int& i_row, const int& i_col) {

  assert( i_row < n_rows);
  assert( i_col < n_cols);

  return data[i_row * n_cols + i_col];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// set pixel block
////////////////////////////////////////////////////////////////////////////////////////////////////
void
Image::setBlock( int rows_start, int rows_end, int cols_start, int cols_end, const Image& im) {

  int i_row_src = 0;
  int i_col_src = 0;
  for( int i_row = rows_start; i_row < rows_end; ++i_row) {
    i_col_src = 0;
    for( int i_col = cols_start; i_col < cols_end; ++i_col) {
      data[i_row * n_cols + i_col] = im( i_row_src, i_col_src);
      ++i_col_src;
    }
    ++i_row_src;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// constructor, user
////////////////////////////////////////////////////////////////////////////////////////////////////
void
Image::write( const std::string& fname) const {

  // open file and check
  std::fstream file( fname, std::ios::out);
  if( ! file.good()) {
    std::cerr << "Image::write() : Failed to open \"" << fname << "\"" << std::endl;
    return;
  }

  // write header
  file << "P2\n";
  file << "# Simple example image\n";
  file << n_rows << " " << n_cols << '\n';
  file << 255 << '\n';

  // write image data (y axis is flipped in image format)
  for( int i_row = n_rows-1; i_row >= 0; --i_row) {
    for( int i_col = 0; i_col < n_cols; ++i_col) {
      // clamp if necessary
      file << std::min( 255, static_cast<int>( std::round( data[i_row * n_cols + i_col]))) << " ";
    }
    file << '\n';
  }

  if( ! file.good()) {
    std::cerr << "Image::write() : Failed to write '" << fname << "'" << std::endl;
    return;
  }
  file.close();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// constructor, user
////////////////////////////////////////////////////////////////////////////////////////////////////
void
Image::read( const std::string& fname) {

  // open file and check
  std::fstream file( fname, std::ios::in);
  if( ! file.good()) {
    std::cerr << "Image::read() : Failed to open \"" << fname << "\"" << std::endl;
    return;
  }

  try {

    std::string line;
    std::getline( file, line);
    if( "P2" != line) {
      std::cerr << "Image::read() : Cannot read file. Incorrect format." << std::endl;
      return;
    }

    // skip comment
    std::getline( file, line);

    std::string line_size;
    std::getline( file, line_size);
    std::stringstream sstr_line_size( line_size);
    sstr_line_size >> n_rows;
    sstr_line_size >> n_cols;

    std::string line_max_val;
    std::getline( file, line_max_val);
    std::stringstream sstr_max_val( line_max_val);
    int max_val = 0;
    sstr_max_val >> max_val;
    if( 255 != max_val) {
      std::cerr <<  "Image::read() : incorrect image format :: max_val = " << max_val << std::endl;
      return;
    }

  #ifdef DEBUG
    std::cout << "Image::read() : n_rows = " << n_rows << std::endl;
    std::cout << "Image::read() : n_cols = " << n_cols << std::endl;
    std::cout << "Image::read() : max_val = " << max_val << std::endl;
  #endif

    data.resize( n_rows * n_cols);

    // read image data (y axis is flipped in image format)
    int k = 0;
    for( int i_row = n_rows-1; i_row >= 0; --i_row) {

      std::string str_row;
      if( ! std::getline( file, str_row)) {
        if( file.eof()) {
          std::cerr << "Image::read() : Failed to read '" << fname << "' :: "
                    << i_row << " at k = " << k << std::endl;
          return;
        }
      }
      std::stringstream sstr_row( str_row);

      float val = 0.0;
      for( int i_col = 0; i_col < n_cols; ++i_col) {

        sstr_row >> val;
        data[i_row * n_cols + i_col] = val;
      }
    }
  }
  catch( const std::exception& excep) {
    std::cerr << "Image::read() : exeception occurred : " << excep.what() << std::endl;
  }

  if( ! file.good()) {
    std::cerr << "Image::read() : Failed to read '" << fname << "'" << std::endl;
    return;
  }
  file.close();
}
