/** Copyright (C) 2013 David Braam - Released under terms of the AGPLv3 License */
#include <string.h>
//#include <strings.h>
#include <stdio.h>

#include "modelFile.h"
#include "../utils/logoutput.h"
#include "../utils/string.h"

FILE* binaryMeshBlob = nullptr;

/* Custom fgets function to support Mac line-ends in Ascii STL files. OpenSCAD produces this when used on Mac */
void* fgets_(char* ptr, size_t len, FILE* f)
{
    while(len && fread(ptr, 1, 1, f) > 0)
    {
        if (*ptr == '\n' || *ptr == '\r')
        {
            *ptr = '\0';
            return ptr;
        }
        ptr++;
        len--;
    }
    return nullptr;
}

SimpleModel* loadModelSTL_ascii(SimpleModel *m,const char* filename, FMatrix3x3& matrix)
{
    m->volumes.push_back(SimpleVolume());
    SimpleVolume* vol = &m->volumes[m->volumes.size()-1];
    FILE* f = fopen(filename, "rt");
    char buffer[1024];
    FPoint3 vertex;
    int n = 0;
    Point3 v0(0,0,0), v1(0,0,0), v2(0,0,0);
    while(fgets_(buffer, sizeof(buffer), f))
    {
        if (sscanf(buffer, " vertex %lf %lf %lf", &vertex.x, &vertex.y, &vertex.z) == 3)
        {
            n++;
            switch(n)
            {
            case 1:
                v0 = matrix.apply(vertex);
                break;
            case 2:
                v1 = matrix.apply(vertex);
                break;
            case 3:
                v2 = matrix.apply(vertex);
                vol->addFace(v0, v1, v2);
                n = 0;
                break;
            }
        }
    }
    fclose(f);
    return m;
}

SimpleModel* loadModelSTL_binary(SimpleModel *m,const char* filename, FMatrix3x3& matrix)
{
    FILE* f = fopen(filename, "rb");
    char buffer[80];
    uint32_t faceCount;
    //Skip the header
    if (fread(buffer, 80, 1, f) != 1)
    {
        fclose(f);
        return nullptr;
    }
    //Read the face count
    if (fread(&faceCount, sizeof(uint32_t), 1, f) != 1)
    {
        fclose(f);
        return nullptr;
    }
    //For each face read:
    //float(x,y,z) = normal, float(X,Y,Z)*3 = vertexes, uint16_t = flags
    m->volumes.push_back(SimpleVolume());
    SimpleVolume* vol = &m->volumes[m->volumes.size()-1];
	if(vol == nullptr)
	{
		fclose(f);
		return nullptr;
	}

    for(unsigned int i=0;i<faceCount;i++)
    {
        if (fread(buffer, sizeof(float) * 3, 1, f) != 1)
        {
            fclose(f);
            return nullptr;
        }
        float v[9];
        if (fread(v, sizeof(float) * 9, 1, f) != 1)
        {
            fclose(f);
            return nullptr;
        }
        Point3 v0 = matrix.apply(FPoint3(v[0], v[1], v[2]));
        Point3 v1 = matrix.apply(FPoint3(v[3], v[4], v[5]));
        Point3 v2 = matrix.apply(FPoint3(v[6], v[7], v[8]));
        vol->addFace(v0, v1, v2);
        if (fread(buffer, sizeof(uint16_t), 1, f) != 1)
        {
            fclose(f);
            return nullptr;
        }
    }
    fclose(f);
    return m;
}

SimpleModel* loadModelSTL(SimpleModel *m,const char* filename, FMatrix3x3& matrix)
{
    FILE* f = fopen(filename, "r");
    char buffer[6];
    if (f == nullptr)
        return nullptr;

    if (fread(buffer, 5, 1, f) != 1)
    {
        fclose(f);
        return nullptr;
    }
    fclose(f);

    buffer[5] = '\0';
    if (stringcasecompare(buffer, "solid") == 0)
    {
        SimpleModel* asciiModel = loadModelSTL_ascii(m, filename, matrix);
        if (!asciiModel)
            return nullptr;

        // This logic is used to handle the case where the file starts with
        // "solid" but is a binary file.
        if (m->volumes[m->volumes.size()-1].faces.size() < 1)
        {
            m->volumes.erase(m->volumes.end() - 1);
            return loadModelSTL_binary(m, filename, matrix);
        }
        return asciiModel;
    }
    return loadModelSTL_binary(m, filename, matrix);
}

SimpleModel* loadModelFromFile(SimpleModel *m,const char* filename, FMatrix3x3& matrix)
{
    const char* ext = strrchr(filename, '.');
    if (ext && stringcasecompare(ext, ".stl") == 0)
    {
        return loadModelSTL(m,filename, matrix);
    }
    if (filename[0] == '#' && binaryMeshBlob != nullptr)
    {
        while(*filename == '#')
        {
            filename++;

            m->volumes.push_back(SimpleVolume());
            SimpleVolume* vol = &m->volumes[m->volumes.size()-1];
            int32_t n, pNr = 0;
            if (fread(&n, 1, sizeof(int32_t), binaryMeshBlob) < 1)
                return nullptr;
            cura::log("Reading mesh from binary blob with %i vertexes\n", n);
            Point3 v[3];
            while(n)
            {
                float f[3];
                if (fread(f, 3, sizeof(float), binaryMeshBlob) < 1)
                    return nullptr;
                FPoint3 fp(f[0], f[1], f[2]);
                v[pNr++] = matrix.apply(fp);
                if (pNr == 3)
                {
                    vol->addFace(v[0], v[1], v[2]);
                    pNr = 0;
                }
                n--;
            }
        }
        return m;
    }
    return nullptr;
}
