add_library(ImGui STATIC
    ImGui/imgui.cpp
    ImGui/imgui_demo.cpp
    ImGui/imgui_draw.cpp
    ImGui/imgui_impl_glfw.cpp
    ImGui/imgui_impl_opengl3.cpp
    ImGui/imgui_tables.cpp
    ImGui/imgui_widgets.cpp
)

target_include_directories(ImGui PUBLIC
    ImGui
)