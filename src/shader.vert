#version 450

layout(location = 0) in vec2 inPosition;

layout(push_constant) uniform constants {
    mat4 rotation;
} Constants;

void main() {
    gl_Position = Constants.rotation * vec4(inPosition / 2, 0., 1.);
}
