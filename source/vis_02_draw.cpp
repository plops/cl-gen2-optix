
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
#include <algorithm>
void uploadTex(const void *image, int w, int h) {
  glGenTextures(1, &(state._fontTex));
  glBindTexture(GL_TEXTURE_2D, state._fontTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               image);
}
void initDraw() { glClearColor(0, 0, 0, 1); }
void cleanupDraw() { glDeleteTextures(1, &(state._fontTex)); }
void drawFrame() { glClear(((GL_COLOR_BUFFER_BIT) | (GL_DEPTH_BUFFER_BIT))); };