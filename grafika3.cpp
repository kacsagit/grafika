//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2016-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiv?ve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
 
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
#include <vector>
 
#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>        // must be downloaded 
#include <GL/freeglut.h>    // must be downloaded unless you have an Apple
#endif
 
const unsigned int windowWidth = 600, windowHeight = 600;
 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...
 
// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 0;
 
void getErrorInfo(unsigned int handle) {
    int logLen;
    glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
    if (logLen > 0) {
        char * log = new char[logLen];
        int written;
        glGetShaderInfoLog(handle, logLen, &written, log);
        printf("Shader log:\n%s", log);
        delete log;
    }
}
 
// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
    int OK;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
    if (!OK) {
        printf("%s!\n", message);
        getErrorInfo(shader);
    }
}
 
// check if shader could be linked
void checkLinking(unsigned int program) {
    int OK;
    glGetProgramiv(program, GL_LINK_STATUS, &OK);
    if (!OK) {
        printf("Failed to link shader program!\n");
        getErrorInfo(program);
    }
}
 
 
const char *vertexSource = R"(
    #version 130
        
        uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
        uniform vec4  wLiPos[5];       // pos of light source 
        uniform vec4  wEye;         // pos of eye
        uniform int lightSize;
        
        in  vec4 vtxPos;            // pos in modeling space
        in  vec4 vtxNorm;           // normal in modeling space
        in  vec2 vtxUV;
 
        out vec2 texcoord;
        out vec4 wNormal;           // normal in world space
        out vec4 wView[5];             // view in world space
        out vec4 wLight[5];            // light dir in world space
 
                                                            
        void main() {
           gl_Position = vtxPos* MVP; // to NDC    
           texcoord = vtxUV;
           vec4 wPos = vtxPos * M;
           wNormal =  Minv * vtxNorm;
            for(int i=0;i<lightSize;i++){
                 wView[i]= wEye-wPos*wLiPos[i].w;
                 wLight[i]= wLiPos[i] - wPos ;
            }
        
        
        
        }
)";
 
 
// fragment shader in GLSL
const char *fragmentSource = R"(
    #version 130
 
        uniform sampler2D samplerUnit;
 
        uniform vec4 kd, ks, ka;// diffuse, specular, ambient ref
        uniform vec4 La[5], Le[5];    // ambient and point source rad
        uniform float shine;    // shininess for specular ref
        uniform int lightSize;
        
        in  vec4 wNormal;       // interpolated world sp normal
        in  vec4 wView[5];         // interpolated world sp view
        in  vec4 wLight[5];        // interpolated world sp illum dir
        in  vec2 texcoord;
 
        out vec4 fragmentColor; // output goes to frame buffer
        
        void main() {
                
            for(int i=0;i<lightSize;i++){
                     vec4 N = normalize(wNormal);
                    vec4 V = normalize(wView[i]);
                   vec4 L = normalize(wLight[i]);
                   vec4 H = normalize(L + V);
                   float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
                    fragmentColor+=texture(samplerUnit, texcoord)*La[i]+(texture(samplerUnit, texcoord)*cost+ks*pow(cosd,shine))*Le[i]/pow(length(wLight[i]),2); 
            }
        }
 
 
 
        )";
 
 
 
// row-major matrix 4x4
struct mat4 {
    float m[4][4];
public:
    mat4() {
        m[0][0] = 1; m[0][1] = 0; m[0][2] = 0; m[0][3] = 0;
        m[1][0] = 0; m[1][1] = 1; m[1][2] = 0; m[1][3] = 0;
        m[2][0] = 0; m[2][1] = 0; m[2][2] = 1; m[2][3] = 0;
        m[3][0] = 0; m[3][1] = 0; m[3][2] = 0; m[3][3] = 1;
    }
    mat4(float m00, float m01, float m02, float m03,
        float m10, float m11, float m12, float m13,
        float m20, float m21, float m22, float m23,
        float m30, float m31, float m32, float m33) {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
        m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
    }
 
    mat4 operator*(const mat4& right) {
        mat4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
            }
        }
        return result;
    }
    operator float*() { return &m[0][0]; }
 
 
 
    void SetUniform(unsigned shaderProg, char * name) {
        int loc = glGetUniformLocation(shaderProg, name);
        glUniformMatrix4fv(loc, 1, GL_TRUE, &m[0][0]);
    }
};
 
mat4 Translate(float tx, float ty, float tz) {
    return mat4(1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        tx, ty, tz, 1);
}
 
 
mat4 Rotate(float angle, float wx, float wy, float wz) {
    float c = cosf(angle);
    float s = sinf(angle);
 
    return mat4( wx * wx * (1 - c)+ c, wx * wy * (1 - c) - wz * s, wx * wz * (1 - c) + wy * s, 0,
        wy * wx * (1 - c) + wz * s,  wy * wy * (1 - c)+ c, wy * wz * (1 - c) - wx * s, 0,
        wz * wx * (1 - c) - wy * s, wz * wy * (1 - c) + wx * s, wz * wz * (1 - c)+ c, 0,
        0, 0, 0, 1);
}
mat4 Scale(float sx, float sy, float sz) {
    return mat4(
        sx, 0, 0, 0,
        0, sy, 0, 0,
        0, 0, sz, 0,
        0, 0, 0, 1
        );
}
 
 
 
// 3D point in homogeneous coordinates
struct vec4 {
    float v[4];
 
 
    vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
        v[0] = x; v[1] = y; v[2] = z; v[3] = w;
    }
 
 
    vec4 operator+(const vec4& vec) const {
        return vec4(v[0] + vec.v[0], v[1] + vec.v[1], v[2] + vec.v[2]);
    }
 
    vec4 operator-(const vec4& vec) const {
        return vec4(v[0] - vec.v[0], v[1] - vec.v[1], v[2] - vec.v[2]);
    }
 
    vec4 operator*(const vec4& vec) const {
        return vec4(v[0] * vec.v[0], v[1] * vec.v[1], v[2] * vec.v[2]);
    }
    vec4 operator/(const vec4& vec) const {
        return vec4(v[0] / vec.v[0], v[1] / vec.v[1], v[2] / vec.v[2]);
    }
    vec4 operator+(const float& vec) const {
        return vec4(v[0] + vec, v[1] + vec, v[2] + vec);
    }
    vec4 operator-(const float& vec) const {
        return vec4(v[0] - vec, v[1] - vec, v[2] - vec);
    }
 
    vec4 operator*(const float& vec) const {
        return vec4(v[0] * vec, v[1] * vec, v[2] * vec);
    }
    vec4 operator/(const float& vec) const {
        return vec4(v[0] / vec, v[1] / vec, v[2] / vec);
    }
 
    float length() const {
        return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }
 
 
    float dot(const vec4& vec) const {
        return v[0] * vec.v[0] + v[1] * vec.v[1] + v[2] * vec.v[2];
    }
    vec4 cross(const vec4& vec) const {
        return vec4(v[1] * vec.v[2] - v[2] * vec.v[1], v[2] * vec.v[0] - v[0] * vec.v[2], v[0] * vec.v[1] - v[1] * vec.v[0]);
    }
 
    vec4 normalize() const {
        float l = this->length();
        if (l < 0.0001)
            return vec4();
        return *this / l;
 
    }
    friend vec4 operator+(float f, const vec4& vec) {
        return vec + f;
    }
    friend vec4 operator*(float f, const vec4& vec) { return vec*f; }
    friend float dot(const vec4& vec, const vec4& vec2) {
        return vec.dot(vec2);
    }
    friend vec4 cross(const vec4& vec, const vec4& vec2) {
        return vec.cross(vec2);
    }
 
    vec4 operator*(const mat4& mat) {
        vec4 result;
        for (int j = 0; j < 4; j++) {
            result.v[j] = 0;
            for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
        }
        return result;
    }
 
 
 
    void SetUniform(unsigned shaderProg, char * name) {
        int loc = glGetUniformLocation(shaderProg, name);
        glUniform4fv(loc, 1, &v[0]);
    }
 
};
 
void SetUniform(unsigned shaderProg, char * name, float shine) {
    int loc = glGetUniformLocation(shaderProg, name);
    glUniform1f(loc, shine);
}
 
void SetUniform(unsigned shaderProg, char * name, int shine) {
    int loc = glGetUniformLocation(shaderProg, name);
    glUniform1i(loc, shine);
}
 
 
 
 
 
 
// handle of the shader program
unsigned int shaderProgram;
 
 
 
struct Camera {
    vec4  wEye, wLookat, wVup;
    float fov, asp, fp, bp;
public:
    Camera() {
        wEye = vec4(10.4, 0, 0);
        wLookat = vec4(0, 0, 7);
        wVup = vec4(0, 1, 0);
        fov = M_PI_4;
        asp = windowWidth / windowHeight;
        fp = 1.f;
        bp = 100.f;
 
    }
 
 
 
 
    mat4 V() { // view matrix
        vec4 w = (wEye - wLookat).normalize();
        vec4 u = cross(wVup, w).normalize();
        vec4 v = cross(w, u);
        return Translate(-wEye.v[0], -wEye.v[1], -wEye.v[2]) *
            mat4(u.v[0], v.v[0], w.v[0], 0.0f,
                u.v[1], v.v[1], w.v[1], 0.0f,
                u.v[2], v.v[2], w.v[2], 0.0f,
                0.0f, 0.0f, 0.0f, 1.0f);
    }
    mat4 P() { // projection matrix
        float sy = 1 / tanf(fov / 2.0f);
        return mat4(sy / asp, 0.0f, 0.0f, 0.0f,
            0.0f, sy, 0.0f, 0.0f,
            0.0f, 0.0f, -(fp + bp) / (bp - fp), -1.0f,
            0.0f, 0.0f, -2 * fp*bp / (bp - fp), 0.0f);
    }
 
    void animate(float time, vec4 pos) {
    //    pos.v[1] = 0;
        wLookat = pos;
    }
};
Camera camera;
 
vec4* LoadImage(char* fname, int width, int height) {
    vec4* ary = new vec4[width*height];
    bool b = true;
    int db = 0;
    if (fname == "gomb") {
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++) {
                if ((j % 100 == i % 100)) {
                    b = !b;
                }
                if (b)
                    ary[height*i + j] = vec4(0.8, 0.8, 0.8);
                else
                    ary[height*i + j] = vec4(0.1, 0.5, 0.1);
            }
    }
    else {
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++) {
                if((int)(i/15+j/15l)%2){
                    ary[height*i + j] = vec4(0.8, 0.8, 0.1);
                    db++;
                }
                else {
                    ary[height*i + j] = vec4(0.1, 0.5, 0.8);
                    db = 0;
                }
 
            }
    }
    return ary;
 
 
 
 
 
}
 
struct Texture {
    unsigned int textureId;
    void drawTexture(char* fname = "ranf") {
        glGenTextures(1, &textureId);
        glBindTexture(GL_TEXTURE_2D, textureId);    // binding
        int width = 600, height = 600;
        vec4 *image = LoadImage(fname, width, height);
 
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,
            0, GL_RGBA, GL_FLOAT, image); //Texture -> OpenGL
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
};
 
struct Geometry {
    unsigned int vao, nVtx;
    Texture texture;
    Geometry() {
 
    }
    void Draw() {
 
 
        camera.wEye.SetUniform(shaderProgram, "wEye");
        int samplerUnit = GL_TEXTURE0; // GL_TEXTURE1, ?
        int location = glGetUniformLocation(shaderProgram, "samplerUnit");
        glUniform1i(location, 0);
        glActiveTexture(samplerUnit);
        glBindTexture(GL_TEXTURE_2D, texture.textureId);
 
 
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, nVtx);
 
    }
    virtual void animate(float time, vec4* pos, mat4* rotateM, mat4* rotateMinv){
        //*pos = vec4(0, 0, 0) + 10 * sin(t*M_PI / 10);
    }
 
};
 
struct VertexData {
    vec4 position, normal;
    float u, v;
};
 
struct ParamSurface : Geometry {
    ParamSurface() : Geometry() {}
    virtual VertexData GenVertexData(float u, float v) = 0;
    virtual void Create(int N, int M) {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        nVtx = N * M * 6;
        unsigned int vbo;
        glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
 
        VertexData *vtxData = new VertexData[nVtx], *pVtx = vtxData;
        for (int i = 0; i < N; i++) for (int j = 0; j < M; j++) {
            *pVtx++ = GenVertexData((float)i / N, (float)j / M);
            *pVtx++ = GenVertexData((float)(i + 1) / N, (float)j / M);
            *pVtx++ = GenVertexData((float)i / N, (float)(j + 1) / M);
            *pVtx++ = GenVertexData((float)(i + 1) / N, (float)j / M);
            *pVtx++ = GenVertexData((float)(i + 1) / N, (float)(j + 1) / M);
            *pVtx++ = GenVertexData((float)i / N, (float)(j + 1) / M);
        }
 
 
        int stride = sizeof(VertexData), svec4 = sizeof(vec4);
        glBufferData(GL_ARRAY_BUFFER, nVtx * stride, vtxData, GL_STATIC_DRAW);
 
        glEnableVertexAttribArray(0);  // AttribArray 0 = POSITION
        glEnableVertexAttribArray(1);  // AttribArray 1 = NORMAL
        glEnableVertexAttribArray(2);  // AttribArray 2 = UV
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)svec4);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(2 * svec4));
 
 
    }
};
 
struct Torus : public ParamSurface {
    vec4 center;
    float r;
    float R;
 
    Torus(vec4 ce, float R, float r) : center(ce), R(R), r(r), ParamSurface() {
    }
    void Create(int N, int M) {
        texture.drawTexture();
        ParamSurface::Create(N, M);
    }
 
    VertexData GenVertexData(float u, float v) {
        VertexData vd;
        vd.position = vec4(
            (R + r * cosf(u * 2.f * M_PI)) * cosf(v * 2.f * M_PI),
            r * sinf(u * 2.f * M_PI),
            (R + r * cosf(u * 2.f * M_PI)) * sinf(v * 2.f * M_PI)
            );
        vec4 centerSphere = vec4(cosf(v * 2.f * M_PI) * R, 0, sinf(v * 2.f * M_PI) * R);
        vd.normal = (centerSphere - vd.position).normalize();
        vd.position = vd.position + center;
        vd.u = u;
        vd.v = v;
        return vd;
    }
 
    float distence(vec4 point) {
        vec4 point1(point.v[0], 0, point.v[2]);
        vec4 p1normal = (point1 - center).normalize();
        vec4 circleCenter = center + p1normal * R;
        return r - (point - circleCenter).length();
 
    }
 
    vec4 normalvec(vec4 point) {
        vec4 point1(point.v[0], 0, point.v[2]);
        vec4 p1normal = (point1 - center).normalize();
        vec4 circleCenter = center + p1normal * R;
        return (point - circleCenter).normalize();
 
 
    }
 
};
 
Torus t(vec4(), 7, 3.5);
struct Sphere : public ParamSurface {
    vec4 center;
    float radius;
    float deltatime;
    float passedtime;
    vec4 speed;
 
    Sphere(vec4 c, float r) : center(c), radius(r), ParamSurface() {
        passedtime = 0;
        deltatime = 0;
        speed = vec4(0.8, 0.3, 0.1).normalize();
        
    
    }
    void Create(int N, int M) {
        texture.drawTexture("gomb");
        ParamSurface::Create(N, M);
 
    }
 
    VertexData GenVertexData(float u, float v) {
        VertexData vd;
        vd.normal = vec4(cosf(u * 2 * M_PI) * sinf(v*M_PI),
            sinf(u * 2 * M_PI) * sinf(v*M_PI),
            cosf(v*M_PI));
        vd.position = vd.normal * radius + center;
        vd.u = u;
        vd.v = v;
        return vd;
    }
    void animate(float time, vec4* pos, mat4* rotateM, mat4* rotateMinv) {
        deltatime = time - passedtime;
        //*pos = vec4(0, 0, 0) + 10 * sin(time*M_PI / 10);
        float v = time / 3.f;
        float u = v * 3.f;
        float radi = t.r - radius;
        *pos = vec4(
            (t.R + radi * cosf(u * 2.f * M_PI)) * cosf(v * 2.f * M_PI),
            radi * sinf(u * 2.f * M_PI),
            (t.R + radi * cosf(u * 2.f * M_PI)) * sinf(v * 2.f * M_PI)
            );
 
        vec4 sebu = vec4(
            -2.f * M_PI*radi*sinf(2.f * M_PI*u)*cosf(2.f * M_PI*v),
            2.f * M_PI*radi*cosf(u * 2.f * M_PI),
            -2.f * M_PI*radi*sinf(2.f * M_PI*u)*sinf(2.f * M_PI*v)
            );
 
        vec4 sebv = vec4(
            -2.f *radi* M_PI*sinf(2.f * M_PI*v)*cosf(2.f * M_PI*u),
            0,
            2.f * radi*M_PI*cosf(2.f * M_PI*v)*cosf(2.f * M_PI*u));
 
    
        vec4 seb = sebu * 3.f/3.f + sebv * 1.f / 3.f;
 
        vec4 omega = seb / (-1 * seb.normalize()*radius);
        float angle = omega.length()*deltatime;
        vec4 rotAxis = cross(t.GenVertexData(u, v).normal, seb.normalize()).normalize();
        *rotateM = *rotateM*Rotate(-angle, rotAxis.v[0], rotAxis.v[1], rotAxis.v[2]);
        *rotateMinv = *rotateMinv*Rotate(-angle, rotAxis.v[0], rotAxis.v[1], rotAxis.v[2]);
 
        passedtime = time;
        camera.animate(time, *pos);
 
        
    }
};
 
 
struct Material {
    float shiness;
    vec4 kd;
    vec4 ks;
    vec4 ka;
    Material() {
        kd = vec4(1, 1, 1);
        ks = vec4(1, 1, 1);
        ka = vec4(1, 1, 1);
        shiness = 10;
    }
    void setUni() {
        kd.SetUniform(shaderProgram, "kd");
        ks.SetUniform(shaderProgram, "ks");
        ka.SetUniform(shaderProgram, "ka");
        SetUniform(shaderProgram, "shine", shiness);
    }
 
};
 
 
 
struct Object {
    vec4 scale, pos, rotAxis;
    float angle;
    Geometry* geometry;
    Material material;
    mat4 rotateM;
    mat4 rotateMinv;
 
    Object() {
        scale = vec4(1, 1, 1);
        pos = vec4(0, 0, 0);
        rotAxis = vec4(0, 0, 0);
        angle = 0;
        geometry = new Geometry();
        //rotateM = Rotate(angle, rotAxis.v[0], rotAxis.v[1], rotAxis.v[2]);
        //rotateMinv = Rotate(-angle, rotAxis.v[0], rotAxis.v[1], rotAxis.v[2]);
 
    }
 
    void setGeom(Geometry *g) {
        geometry = g;
    }
 
    void setUni() {
        material.setUni();
    }
    void animate(float time) {
        geometry->animate(time, &pos, &rotateM,&rotateMinv);
 
    }
 
    void draw() {
        mat4 M = Scale(scale.v[0], scale.v[1], scale.v[2]) *
            rotateM*
            Translate(pos.v[0], pos.v[1], pos.v[2]);
        mat4 Minv = Translate(-pos.v[0], -pos.v[1], -pos.v[2]) *
            rotateMinv*
            Scale(1 / scale.v[0], 1 / scale.v[1], 1 / scale.v[2]);
        mat4 MVP = M * camera.V() * camera.P();
        M.SetUniform(shaderProgram, "M");
        Minv.SetUniform(shaderProgram, "Minv");
        MVP.SetUniform(shaderProgram, "MVP");
        setUni();
        geometry->Draw();
    }
 
 
};
 
 
 
struct Light {
    vec4 La, Le, wLightPos;
    vec4 speed;
    float deltatime;
    float passedtime;
 
    Light() {
        //    wEye = vec4(9.4, 0, 0);
        //    wLookat = vec4(0, 0, 7);
        La = vec4(0.1, 0.1, 0.1);
        Le = vec4(0, 10, 10);
        wLightPos = vec4(1.2, 0, 8);
        speed = vec4(-10, 7, 10);
        deltatime = 0;
        passedtime = 0;
    }
 
    void set(vec4 a, vec4 e, vec4 LightPos, vec4 speede) {
        La = a;
        Le = e;
        wLightPos = LightPos;
        speed = speede;
    }
 
    void setUni() {
 
    }
 
    void animate(float time) {
        //wLightPos = vec4(1.2, 0, 9+sin(time));
 
        deltatime = time - passedtime;
        wLightPos = wLightPos + deltatime*speed;
        if (fabs(t.distence(wLightPos))>t.r) {
            speed = ((speed)-t.normalvec(wLightPos) * dot(t.normalvec(wLightPos), (speed)) * 2.0f);
            wLightPos = wLightPos + deltatime*speed;
        }
        speed = speed + deltatime*vec4(0, -9.8, 0);
        passedtime = time;
 
    }
 
 
};
 
struct Scene {
    Light light[10];
    int lightSize;
    Object object[5];
    int objectSize;
    Camera camera;
 
    Scene() { lightSize = 0; objectSize = 0; }
    void addLight(Light l) {
        if (lightSize<5)
            light[lightSize++] = l;
    }
 
    void addObject(Object o) {
        if (objectSize<5)
            object[objectSize++] = o;
 
    }
 
    void render() {
        for (int i = 0; i < lightSize; i++)
            light[i].setUni();
        for (int i = 0; i < objectSize; i++)
            object[i].draw();
        setUni();
    }
 
    void setUni() {
        SetUniform(shaderProgram, "lightSize", lightSize);
        vec4 la[10];
        vec4 le[10];
        vec4 wLightPos[10];
        for (int i = 0; i < lightSize; i++) {
            la[i] = light[i].La;
            le[i] = light[i].Le;
            wLightPos[i] = light[i].wLightPos;
        }
 
        glUniform4fv(glGetUniformLocation(shaderProgram, "La"), 10, &la[0].v[0]);
        glUniform4fv(glGetUniformLocation(shaderProgram, "Le"), 10, &le[0].v[0]);
        glUniform4fv(glGetUniformLocation(shaderProgram, "wLiPos"), 10, &wLightPos[0].v[0]);
    }
 
    void animate(float time) {
        for (int i = 0; i < objectSize; i++)
            object[i].animate(time);
        for (int i = 0; i < objectSize; i++)
            light[i].animate(time);
    }
 
 
 
};
 
Scene sc;
 
 
Sphere s(vec4(), 1.2);
Light l;
Light l2;
Object o1;
Object o2;
 
// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    l2.set(vec4(0.1, 0.1, 0.1), vec4(10, 10, 0), vec4(8, 0, 1.2), vec4(10, 10, 10));
    sc.addLight(l);
    sc.addLight(l2);
    s.Create(50, 50); // tessellation level
    t.Create(350, 350);
    o1.setGeom(&s);
    o2.setGeom(&t);
    sc.addObject(o1);
    sc.addObject(o2);
 
    // Create vertex shader from string
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    if (!vertexShader) {
        printf("Error in vertex shader creation\n");
        exit(1);
    }
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);
    checkShader(vertexShader, "Vertex shader error");
 
    // Create fragment shader from string
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    if (!fragmentShader) {
        printf("Error in fragment shader creation\n");
        exit(1);
    }
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);
    checkShader(fragmentShader, "Fragment shader error");
 
    // Attach shaders to a single program
    shaderProgram = glCreateProgram();
    if (!shaderProgram) {
        printf("Error in shader program creation\n");
        exit(1);
    }
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
 
    // Connect Attrib Arrays to input variables of the vertex shader
    glBindAttribLocation(shaderProgram, 0, "vtxPosition"); // vertexPosition gets values from Attrib Array 0
    glBindAttribLocation(shaderProgram, 1, "vtxNorm");
    glBindAttribLocation(shaderProgram, 2, "vtxUV");
 
    // Connect the fragmentColor to the frame buffer memory
    glBindFragDataLocation(shaderProgram, 0, "fragmentColor");    // fragmentColor goes to the frame buffer memory
 
                                                                // program packaging
    glLinkProgram(shaderProgram);
    checkLinking(shaderProgram);
    // make this program run
    glUseProgram(shaderProgram);
 
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
}
 
void onExit() {
    glDeleteProgram(shaderProgram);
    printf("exit");
}
 
// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0, 0, 0, 0);                            // background color 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
 
    sc.render();
 
    glutSwapBuffers();                                    // exchange the two buffersani
}
 
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}
 
// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
 
}
 
// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
        float cX = 2.0f * pX / windowWidth - 1;    // flip y axis
        float cY = 1.0f - 2.0f * pY / windowHeight;
 
        glutPostRedisplay();     // redraw
    }
}
 
// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}
 
 
 
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
    float sec = time / 1000.0f;                // convert msec to sec
 
    sc.animate(sec);
 
    glutPostRedisplay();                    // redraw the scene
}
 
// Idaig modosithatod...
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
int main(int argc, char * argv[]) {
    glutInit(&argc, argv);
#if !defined(__APPLE__)
    glutInitContextVersion(majorVersion, minorVersion);
#endif
    glutInitWindowSize(windowWidth, windowHeight);                // Application window is initially of resolution 600x600
    glutInitWindowPosition(100, 100);                            // Relative location of the application window
#if defined(__APPLE__)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
    glutCreateWindow(argv[0]);
 
#if !defined(__APPLE__)
    glewExperimental = true;    // magic
    glewInit();
#endif
 
    printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
    printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
    printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
    glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
    glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
    printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
    printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
 
    onInitialization();
 
    glutDisplayFunc(onDisplay);                // Register event handlers
    glutMouseFunc(onMouse);
    glutIdleFunc(onIdle);
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    glutMotionFunc(onMouseMotion);
 
 
 
    glutMainLoop();
    onExit();
    return 1;
}