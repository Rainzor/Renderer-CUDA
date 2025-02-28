#include "main.h"
#include "preview.h"
#include <cstring>
// #include "core/pathtrace.h"
#include "core/integrator.h"

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static bool reset = true;
static bool first = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
Integrator* integrator;
int iteration;

int width;
int height;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
	startTimeString = currentTimeString();

	if (argc < 2) {
		printf("Usage: %s SCENEFILE.txt\n", argv[0]);
		return 1;
	}

	const char* sceneFile = argv[1];

	// Load scene file
	scene = new Scene(sceneFile);

	integrator = new Integrator;

	//Create Instance for ImGUIData
	guiData = new GuiDataContainer();

	// Set up camera stuff from loaded path tracer settings
	iteration = 0;
	reset = true;
	renderState = &scene->state;
	Camera& cam = renderState->camera;
	width = cam.resolution.x;
	height = cam.resolution.y;

	glm::vec3 view = cam.view;
	glm::vec3 up = cam.up;
	glm::vec3 right = glm::cross(view, up);
	up = glm::cross(right, view);

	cameraPosition = cam.position;

	// compute phi (horizontal) and theta (vertical) relative 3D axis
	// so, (0 0 1) is forward, (0 1 0) is up
	glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
	glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
	phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
	theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
	ogLookAt = cam.lookAt;
	zoom = glm::length(cam.position - ogLookAt);

	// Initialize CUDA and GL components
	init();

	// Initialize ImGui Data
	InitImguiData(guiData);
	// InitDataContainer(guiData);

	// GLFW main loop
	mainLoop();

	return 0;
}

void freeData() {
	delete guiData;
	//resourceFree();
	//pathtraceFree();
	integrator->resourceFree();
	integrator->pathtraceFree();
	delete integrator;
	delete scene;
}

void saveImage() {
	float samples = iteration;
	// output image file
	image img(width, height);

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int index = x + (y * width);
			glm::vec3 pix = renderState->image[index];
			img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
		}
	}

	std::string filename = renderState->imageName;
	std::ostringstream ss;
	ss << filename << "." << startTimeString << "." << samples << "samp";
	filename = ss.str();

	// CHECKITOUT
	img.savePNG(filename);
	//img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {
	if (camchanged) {
		iteration = 0;
		Camera& cam = renderState->camera;
		cameraPosition.x = zoom * sin(phi) * sin(theta);
		cameraPosition.y = zoom * cos(theta);
		cameraPosition.z = zoom * cos(phi) * sin(theta);

		cam.view = -glm::normalize(cameraPosition);
		glm::vec3 v = cam.view;
		glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
		glm::vec3 r = glm::cross(v, u);
		cam.up = glm::cross(r, v);
		cam.right = r;

		cam.position = cameraPosition;
		cameraPosition += cam.lookAt;
		cam.position = cameraPosition;
		camchanged = false;
		reset = true;
	}

	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	if (reset) {
		//pathtraceFree();
		integrator->pathtraceFree();
		//pathtraceInit(scene);
		integrator->pathtraceInit(scene);
		if (first) {
			integrator->resourceInit(scene);
			//resourceInit(scene);
			first = false;
		}
		reset = false;
	}

	if (iteration < renderState->iterations) {
		uchar4* pbo_dptr = NULL;
		iteration++;
		// Map the buffer object into CUDA's address space
		// So we can write to pbo directly from the GPU
		cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

		// execute the kernel
		int frame = 0;
		integrator->render(pbo_dptr, frame, iteration, guiData);

		// unmap buffer object
		cudaGLUnmapBufferObject(pbo);
	}
	else {
		saveImage();
		freeData();
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
	}
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        Camera& cam = renderState->camera;

        switch (key) {
            case GLFW_KEY_ESCAPE:
                saveImage();
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
            case GLFW_KEY_F5:
                saveImage();
                break;
            case GLFW_KEY_SPACE:
                camchanged = true;
                renderState = &scene->state;
                cam.lookAt = ogLookAt;
                break;

            // Camera movement controls
			case GLFW_KEY_W: // Move camera forward
				cam.position += cam.view * 0.1f;
				cam.lookAt += cam.view * 0.1f;
				camchanged = true;
				break;
			case GLFW_KEY_S: // Move camera backward
				cam.position -= cam.view * 0.1f;
				cam.lookAt -= cam.view * 0.1f;
				camchanged = true;
				break;
			case GLFW_KEY_A: // Move camera left
				cam.position -= cam.right * 0.1f;
				cam.lookAt -= cam.right * 0.1f;
				camchanged = true;
				break;
			case GLFW_KEY_D: // Move camera right
				cam.position += cam.right * 0.1f;
				cam.lookAt += cam.right * 0.1f;
				camchanged = true;
				break;
			case GLFW_KEY_Q: // Move camera down
				cam.position -= cam.up * 0.1f;
				cam.lookAt -= cam.up * 0.1f;
				camchanged = true;
				break;
			case GLFW_KEY_E: // Move camera up
				cam.position += cam.up * 0.1f;
				cam.lookAt += cam.up * 0.1f;
				camchanged = true;
				break;
			case GLFW_KEY_UP: // Move camera forward
				cam.position += cam.view * 0.1f;
				cam.lookAt += cam.view * 0.1f;
				camchanged = true;
				break;
			case GLFW_KEY_DOWN: // Move camera backward
				cam.position -= cam.view * 0.1f;
				cam.lookAt -= cam.view * 0.1f;
				camchanged = true;
				break;
			case GLFW_KEY_LEFT: // Move camera left
				cam.position -= cam.right * 0.1f;
				cam.lookAt -= cam.right * 0.1f;
				camchanged = true;
				break;
			case GLFW_KEY_RIGHT: // Move camera right	
				cam.position += cam.right * 0.1f;
				cam.lookAt += cam.right * 0.1f;
				camchanged = true;
				break;
			case GLFW_KEY_PAGE_UP: // Move camera up
				cam.position += cam.up * 0.1f;
				cam.lookAt += cam.up * 0.1f;
				camchanged = true;
				break;
			case GLFW_KEY_PAGE_DOWN: // Move camera down
				cam.position -= cam.up * 0.1f;
				cam.lookAt -= cam.up * 0.1f;
				camchanged = true;
				break;
			case GLFW_KEY_J: // Rotate camera left
				cam.lookAt += cam.right * 0.1f;
				camchanged = true;
				break;
			case GLFW_KEY_L: // Rotate camera right
				cam.lookAt -= cam.right * 0.1f;
				camchanged = true;
				break;
			case GLFW_KEY_I: // Rotate camera up
				cam.lookAt += cam.up * 0.1f;
				camchanged = true;
				break;
			case GLFW_KEY_K: // Rotate camera down
				cam.lookAt -= cam.up * 0.1f;
				camchanged = true;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (MouseOverImGuiWindow())
	{
		return;
	}
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
	middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
	if (leftMousePressed) {
		// compute new camera parameters
		phi -= (xpos - lastX) / width;
		theta -= (ypos - lastY) / height;
		theta = std::fmax(0.001f, std::fmin(theta, PI));
		camchanged = true;
	}
	else if (rightMousePressed) {
		zoom += (ypos - lastY) / height;
		zoom = std::fmax(0.1f, zoom);
		camchanged = true;
	}
	else if (middleMousePressed) {
		renderState = &scene->state;
		Camera& cam = renderState->camera;
		glm::vec3 forward = cam.view;
		forward.y = 0.0f;
		forward = glm::normalize(forward);
		glm::vec3 right = cam.right;
		right.y = 0.0f;
		right = glm::normalize(right);

		cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
		cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
		camchanged = true;
	}
	lastX = xpos;
	lastY = ypos;
}
