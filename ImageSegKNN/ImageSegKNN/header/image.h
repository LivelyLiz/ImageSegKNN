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

#ifndef _IMAGE_H_
#define _IMAGE_H_

// includes, system
#include <vector>

// Simple Image description
class Image {

public:

  //! constructor, user
  Image( int num_rows, int num_cols);

  //! constructor, user
  Image( const std::string& fname);

  //! constructor, copy
  Image( const Image& other);

  // destructor
  ~Image() { };

public:

  const float& operator()( const int& i_row, const int& i_col) const;

  float& operator()( const int& i_row, const int& i_col);

public:

  void read( const std::string& fname);

  void write( const std::string& fname) const;

  void setBlock( int rows_start, int rows_end, int cols_start, int cols_end, const Image& im);

public:

  int n_rows;
  int n_cols;

public:

  std::vector< float > data;

private:

  // constructor, default
  Image();

  // assignment operator
  Image& operator=( const Image&);

};

#endif // _GEOMETRY_H_
