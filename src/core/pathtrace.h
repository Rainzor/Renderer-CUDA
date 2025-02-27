#pragma once

#include <vector>
#include "scene.h"
#include "utils/utilities.h"
#include "utils/common.h"

// void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration, GuiDataContainer* guiData);
void resourceInit(Scene* scene);
void resourceFree();