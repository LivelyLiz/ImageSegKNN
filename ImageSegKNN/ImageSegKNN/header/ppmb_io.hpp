char ch_cap ( char ch );

int i4_max ( int i1, int i2 );

bool ppmb_check_data ( int xsize, int ysize, int maxrgb, unsigned char *r,
  unsigned char *g, unsigned char *b );

bool ppmb_example ( int xsize, int ysize, unsigned char *r, 
  unsigned char *g, unsigned char *b );

bool ppmb_read (std::string file_in_name, int &xsize, int &ysize, int &maxrgb,
  unsigned char **r, unsigned char **g, unsigned char **b );
bool ppmb_read_data (std::ifstream &file_in, int xsize, int ysize, unsigned char *r,
  unsigned char *g, unsigned char *b );
bool ppmb_read_header (std::ifstream &file_in, int &xsize, int &ysize, int &maxrgb );
bool ppmb_read_test (std::string file_in_name );

bool ppmb_write (std::string file_out_name, int xsize, int ysize, unsigned char *r,
  unsigned char *g, unsigned char *b );
bool ppmb_write_data (std::ofstream &file_out, int xsize, int ysize, unsigned char *r,
  unsigned char *g, unsigned char *b );
bool ppmb_write_header (std::ofstream &file_out, int xsize, int ysize, int maxrgb );
bool ppmb_write_test (std::string file_out_name );

bool s_eqi (std::string s1, std::string s2 );
int s_len_trim (std::string s );
void s_word_extract_first (std::string s, std::string &s1, std::string &s2 );
