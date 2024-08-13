#pragma once

#include "OpenCVApplication.h"
#include "stdafx.h"
class SeventhLaboratory
{
public:
    SeventhLaboratory();
    ~SeventhLaboratory();
public:
    Mat computePCA(const char* path, int red_dim);
    void imageUsingPCA2(char* path, int k);
    void imageUsingPCA3(char* path, int k);

};

