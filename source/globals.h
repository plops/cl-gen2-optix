#ifndef GLOBALS_H

#define GLOBALS_H

#include <GLFW/glfw3.h>

#include <complex>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
template <typename T, int MaxLen> class FixedDequeTM : public std::deque<T> {
  // https://stackoverflow.com/questions/56334492/c-create-fixed-size-queue
public:
  std::mutex mutex;
  void push_back(const T &val) {
    if ((MaxLen) == (this->size())) {
      this->pop_front();
    };
    std::deque<T>::push_back(val);
  }
};

#include <chrono>
struct State {
  typeof(std::chrono::high_resolution_clock::now().time_since_epoch().count())
      _start_time;
  GLuint _fontTex;
  bool _framebufferResized;
  GLFWwindow *_window;
  char const *_filename;
};
typedef struct State State;

#endif
