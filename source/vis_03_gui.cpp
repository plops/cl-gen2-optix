
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
// https://youtu.be/nVaQuNXueFw?t=317
// https://blog.conan.io/2019/06/26/An-introduction-to-the-Dear-ImGui-library.html
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl2.h"
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
void initGui() {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("initGui") << (" ")
      << (std::endl) << (std::flush);
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui_ImplGlfw_InitForOpenGL(state._window, true);
  ImGui_ImplOpenGL2_Init();
  ImGui::StyleColorsDark();
}
void cleanupGui() {
  ImGui_ImplOpenGL2_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}
void drawGui() {
  ImGui_ImplOpenGL2_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  ImGui::Begin("camera");
  {
    auto goal = (0.0e+0f);
    auto current = goal;
    ImGui::SliderFloat("cam-rotation", &current, (0.0e+0f), (3.6e+2f));
    if (!((goal) == (current))) {
      goal = current;
    };
    auto m = glm::rotate(glm::mat4((1.e+0f)), glm::radians(current),
                         glm::vec3((0.0e+0f), (0.0e+0f), (1.e+0f)));
    camera_t camera = {glm::vec3(((m) * (glm::vec4((-1.e+1f), (2.e+0f),
                                                   (-1.2e+1f), (1.e+0f))))),
                       glm::vec3((0.0e+0f), (0.0e+0f), (0.0e+0f)),
                       glm::vec3((0.0e+0f), (1.e+0f), (0.0e+0f))};
    set_camera(camera);
  };
  ImGui::End();
  auto b = true;
  ImGui::ShowDemoWindow(&b);
  ImGui::Render();
  ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
};