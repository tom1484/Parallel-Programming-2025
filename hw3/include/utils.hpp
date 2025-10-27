#ifndef UTILS_HPP
#define UTILS_HPP

#include <iomanip>
#include <iostream>

// Simple progress bar that only show percentage
class ProgressBar {
   private:
    int total;
    int initial;
    bool oneline;

   public:
    ProgressBar(int total, int initial = 0) : total(total), initial(initial), oneline(true) {}
    ProgressBar(int total, bool oneline, int initial = 0) : total(total), initial(initial), oneline(oneline) {}
    void update(int current);
    void done();
};

void write_png(const char* filename, unsigned char* raw_image, unsigned width, unsigned height);
void print_device_info();

#endif