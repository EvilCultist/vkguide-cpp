
#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/vector_float3.hpp"
#include <SDL_events.h>
#include <vk_types.h>

class Camera {
  // NOTE: everything in radians please
public:
  float inv_mouse_sensitivity = 200.f;
  float speed = 0.5f;

  glm::mat4 getViewMatrix();
  // glm::mat4 getRotationMatrix();
  //
  glm::vec3 getPos();
  glm::vec3 getVel();
  //
  // float getPitch();
  // float getYaw();

  Camera(glm::vec3 startPos, glm::mat4 startRot, float speed);
  Camera(glm::vec3 startPos, float pitch, float yaw, float speed);
  Camera(glm::vec3 startPos, float pitch, float yaw);
  Camera();

  // void setPitch(float newPitch);
  // void setYaw(float newYaw);
  //
  // void setVelocity(glm::vec3 setVelocity);
  // void setPos(glm::vec3 setPos);

  void processSDLEvent(SDL_Event &e);

  void update();

private:
  bool dirty = true;
  glm::vec3 pos = glm::vec3(0.f);
  glm::vec3 velocity = glm::vec3(0.f);

  glm::mat4 view = glm::mat4(1.f);
  glm::mat4 rot = glm::mat4(1.f);

  void generateMatrices();
  float pitch{0.f};
  float yaw{0.f};
};
