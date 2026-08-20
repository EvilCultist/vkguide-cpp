#include "camera.h"
#include "glm/ext/matrix_transform.hpp"
#include "glm/ext/vector_float4.hpp"
#include "glm/matrix.hpp"
#include <SDL_events.h>
#include <SDL_keycode.h>

// whenever possible matrix initialization is strongly discouraged.
Camera::Camera(glm::vec3 startPos, glm::mat4 startRot, float speed) {
  this->pos = startPos;
  this->speed = speed;

  glm::vec3 angles = glm::eulerAngles(glm::quat_cast(startRot));
  this->pitch = angles.x;
  this->yaw = angles.y;

  this->generateMatrices();
}

// Camera::Camera(glm::vec3 startPos, float pitch, float yaw, float speed) {
//   this->pos = startPos;
//   this->pitch = pitch;
//   this->yaw = yaw;
//   this->speed = speed;
//
//   this->generateMatrices();
// }
//
// Camera::Camera(glm::vec3 startPos, float pitch, float yaw) {
//   this->pos = startPos;
//   this->pitch = pitch;
//   this->yaw = yaw;
//
//   this->generateMatrices();
// }

Camera::Camera(glm::vec3 startPos, float pitch, float yaw, float speed)
    : pos(startPos), pitch(pitch), yaw(yaw), speed(speed) {
  generateMatrices();
}

Camera::Camera(glm::vec3 startPos, float pitch, float yaw)
    : Camera(glm::vec3(0.0f), 0.0f, 0.0f, 0.5f) {}
Camera::Camera() : Camera(glm::vec3(0.0f), 0.0f, 0.0f) {}

glm::mat4 Camera::getViewMatrix() {
  if (this->dirty) {
    this->generateMatrices();
  }

  return this->view;
}

// glm::mat4 Camera::getRotationMatrix() { return this->rot; }
//
glm::vec3 Camera::getPos() { return this->pos; }
glm::vec3 Camera::getVel() { return this->velocity; }
//
// float Camera::getPitch() { return this->pitch; }
// float Camera::getYaw() { return this->yaw; }

void Camera::processSDLEvent(SDL_Event &e) {
  if (e.type == SDL_KEYDOWN) {
    if (e.key.keysym.sym == SDLK_w) {
      this->velocity.z = -1;
    }
    if (e.key.keysym.sym == SDLK_s) {
      this->velocity.z = 1;
    }
    if (e.key.keysym.sym == SDLK_a) {
      this->velocity.x = -1;
    }
    if (e.key.keysym.sym == SDLK_d) {
      this->velocity.x = 1;
    }
  }

  if (e.type == SDL_KEYUP) {
    if (e.key.keysym.sym == SDLK_w) {
      this->velocity.z = 0;
    }
    if (e.key.keysym.sym == SDLK_s) {
      this->velocity.z = 0;
    }
    if (e.key.keysym.sym == SDLK_a) {
      this->velocity.x = 0;
    }
    if (e.key.keysym.sym == SDLK_d) {
      this->velocity.x = 0;
    }
  }

  if (e.type == SDL_MOUSEMOTION) {
    dirty = true;
    this->yaw += (float)e.motion.xrel / this->inv_mouse_sensitivity;
    this->pitch -= (float)e.motion.yrel / this->inv_mouse_sensitivity;
  }

  return;
}

void Camera::update() {
  if (this->dirty) {
    this->generateMatrices();
  }
  glm::mat4 cameraRotation = this->rot;
  this->pos +=
      glm::vec3(cameraRotation * glm::vec4(this->velocity * this->speed, 0.0f));

  dirty = true;
}

void Camera::generateMatrices() {
  this->rot = glm::rotate(glm::mat4(1.f), pitch, glm::vec3(1.f, 0.f, 0.f));
  this->rot = glm::rotate(this->rot, yaw, glm::vec3(0.f, -1.f, 0.f));

  this->view = glm::translate(glm::mat4(1.f), this->pos);
  this->view = glm::inverse(this->view * this->rot);

  this->dirty = false;
}
