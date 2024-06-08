#version 450

layout(location = 0) in vec2 inPosition;

void main() {
    gl_Position = vec4(inPosition / 2, 0., 1.);
}
