
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
State state = {};
void mainLoop() {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("mainLoop") << (" ")
      << (std::endl) << (std::flush);
  while (!(glfwWindowShouldClose(state._window))) {
    glfwPollEvents();
    drawFrame();
    render();
    download_pixels(state._pixels.data());
    static GLuint fb_texture = 0;
    if ((0) == (fb_texture)) {
      glGenTextures(1, &fb_texture);
    };
    glBindTexture(GL_TEXTURE_2D, fb_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, state.launch_params.fbSize_x,
                 state.launch_params.fbSize_y, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 state._pixels.data());
    glDisable(GL_LIGHTING);
    glColor3f(1, 1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, fb_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, state.launch_params.fbSize_x,
               state.launch_params.fbSize_y);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho((0.0e+0f), static_cast<float>(state.launch_params.fbSize_x),
            (0.0e+0f), static_cast<float>(state.launch_params.fbSize_y),
            (-1.e+0f), (1.e+0f));
    glBegin(GL_QUADS);
    glTexCoord2f((0.0e+0f), (0.0e+0f));
    glVertex3f(0, 0, (0.0e+0f));
    glTexCoord2f((0.0e+0f), (1.e+0f));
    glVertex3f(0, static_cast<float>(state.launch_params.fbSize_y), (0.0e+0f));
    glTexCoord2f((1.e+0f), (1.e+0f));
    glVertex3f(static_cast<float>(state.launch_params.fbSize_x),
               static_cast<float>(state.launch_params.fbSize_y), (0.0e+0f));
    glTexCoord2f((1.e+0f), (0.0e+0f));
    glVertex3f(static_cast<float>(state.launch_params.fbSize_x), 0, (0.0e+0f));
    glEnd();
    drawGui();
    glfwSwapBuffers(state._window);
  }

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("exit mainLoop")
      << (" ") << (std::endl) << (std::flush);
}
void run() {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start run") << (" ")
      << (std::endl) << (std::flush);
  initWindow();
  initGui();
  initDraw();
  initOptix();
  mainLoop();

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("finish run") << (" ")
      << (std::endl) << (std::flush);
};
int main() {
  state._start_time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start main") << (" ")
      << (std::endl) << (std::flush);
  state._filename = "bla.txt";
  run();

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start cleanups")
      << (" ") << (std::endl) << (std::flush);
  cleanupOptix();
  cleanupDraw();
  cleanupGui();
  cleanupWindow();

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("end main") << (" ")
      << (std::endl) << (std::flush);
  return 0;
};