//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2016-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
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
 
const unsigned int windowWittth = 600, windowHeight = 600;
 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...
 
 
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
 
 
void checkShader(unsigned int shader, char * message) {
    int OK;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
    if (!OK) {
        printf("%s!\n", message);
        getErrorInfo(shader);
    }
}
 
 
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
    precision highp float;
 
    uniform mat4 MVP;   
    uniform float dopler;       
 
    in vec2 vertexPosition;        
    in vec3 vertexColor;       
    out vec3 color;                
 
    void main() {
        if (dopler<0)
        color = vec3(vertexColor.x-dopler/2,vertexColor.y+dopler/10,vertexColor.z );  
        if (dopler>=0)
        color = vec3(vertexColor.x,vertexColor.y ,vertexColor.z+dopler);     
                                                   
        gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP;         
    }
)";
 
const char *fragmentSource = R"(
    #version 130
    precision highp float;
 
    in vec3 color;                
    out vec4 fragmentColor;        
 
    void main() {
        fragmentColor = vec4(color, 1); 
    }
)";
 
 
 
struct mat4 {
    float m[4][4];
public:
    mat4() {}
    mat4(float m00, float m01, float m02, float m03,
        float m10, float m11, float m12, float m13,
        float m20, float m21, float m22, float m23,
        float m30, float m31, float m32, float m33) {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m02; m[1][3] = m13;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m02; m[2][3] = m23;
        m[3][0] = m30; m[3][1] = m31; m[3][2] = m02; m[3][3] = m33;
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
};
 
 
 
struct vec4 {
    float v[4];
 
 
    vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
        v[0] = x; v[1] = y; v[2] = z; v[3] = w;
    }
 
    vec4 operator*(const mat4& mat) {
        vec4 result;
        for (int j = 0; j < 4; j++) {
            result.v[j] = 0;
            for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
        }
        return result;
    }
    vec4 operator-(const vec4& vec) const {
        return vec4(v[0] - vec.v[0], v[1] - vec.v[1], v[2] - vec.v[2], v[3] - vec.v[3]);
    }
 
 
    vec4 operator*(const vec4& vec) const {
        return vec4(v[0] * vec.v[0], v[1] * vec.v[1], v[2] * vec.v[2], v[3] * vec.v[3]);
    }
    vec4 operator/(const vec4& vec) const {
        return vec4(v[0] / vec.v[0], v[1] / vec.v[1], v[2] / vec.v[2], v[3] / vec.v[3]);
    }
 
    float length() {
        return sqrtf(v[0] * v[0] + v[1] * v[1]);
    }
 
    vec4 operator+(const vec4& vec) const {
        return vec4(v[0] + vec.v[0], v[1] + vec.v[1], v[2] + vec.v[1], v[3] + vec.v[3]);
    }
    vec4 operator*(float t) const {
        return vec4(v[0] * t, v[1] * t, v[2] * t, v[3] * t);
    }
    vec4 operator/(float t) const {
        return vec4(v[0] / t, v[1] / t, v[2] / t, v[3] / t);
    }
};
 
 
struct Camera {
    float wCx, wCy;
    float wWx, wWy;
    int stati;
public:
    Camera() {
        wWx = 20;
        wWy = 20;
        wCx = 0;
        wCy = 0;
        stati = 1;
 
        Animate(0);
    }
    int getStati() { return stati; }
    void setStati(int b) { stati = b; }
    void setzoom(float wx, float wy) { wWx = wx; wWy = wy; }
    void setCenter(float cx, float cy) { wCx = cx; wCy = cy; }
 
    mat4 V() {
        return mat4(1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            -wCx, -wCy, 0, 1);
    }
 
    mat4 P() {
        return mat4(2 / wWx, 0, 0, 0,
            0, 2 / wWy, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1);
    }
 
    mat4 Vinv() {
        return mat4(1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            wCx, wCy, 0, 1);
    }
 
    mat4 Pinv() { // inverse projection matrix
        return mat4(wWx / 2, 0, 0, 0,
            0, wWy / 2, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1);
    }
 
    void Animate(float t) {
        wCx = wCx;// 10 * cosf(t);
        wCy = wCy;
        wWx = wWx;
        wWy = wWy;
    }
};
 
 
Camera camera;
 
 
unsigned int shaderProgram;
 
class Triangle {
    unsigned int vao;
    float sx, sy, sx1,sy1;
    float cx, cy;
    float wTx, wTy;
    float mass;
    vec4 v;
    float last;
    float dopler;
public:
    Triangle() {
        Animate(0);
        last = 0;
        dopler = 0;
 
    }
    float getwTx() { return wTx; }
    float getwTy() { return wTy; }
 
 
    void Create(float x, float y, float m, float c1, float c2, float c3, float size,vec4 velocity) {
        cx = x; cy = y;
        wTx = x; wTy = y;
        mass = m;
        v = velocity;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        unsigned int vbo[2];
        glGenBuffers(2, &vbo[0]);
 
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
 
        float vertexCoords[6 * 7 * 2 ];
 
        int j = 0;
        for (int i = 0; i<7; i++) {
            float angle = (2.0f *(float) M_PI / 7.0f)*i;
            vertexCoords[j++] = 0;
            vertexCoords[j++] = 0;
            vertexCoords[j++] = 0 + size / 2.0f * cosf(angle);
            vertexCoords[j++] = 0 + size / 2.0f * sinf(angle);
            angle = (2 * (float)M_PI / (float)7)*(i + 1);
            vertexCoords[j++] = 0 + size / 2.0f * cosf(angle);
            vertexCoords[j++] = 0 + size / 2.0f * sinf(angle);
        }
        for (int i = 0; i<7; i++) {
            float angle = (2.0f *(float)M_PI / 7.0f)*i + (2.0f * (float)M_PI / 7.0f) / 2.0f;
            float angle2 = (2 * (float)M_PI / 7.0f)*i;
            vertexCoords[j++] = 0 + size * cosf(angle);
            vertexCoords[j++] = 0 + size * sinf(angle);
            vertexCoords[j++] = 0 + size / 2.0f * cosf(angle2);
            vertexCoords[j++] = 0 + size / 2.0f * sinf(angle2);
            angle2 = (2 * (float)M_PI / 7.0f)*(i + 1);
            vertexCoords[j++] = 0 + size / 2.0f * cosf(angle2);
            vertexCoords[j++] = 0 + size / 2.0f * sinf(angle2);
        }
 
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
 
        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
        static float vertexColors[6*7*2*3];
        j = 0;
        for (int i = 0; i < 7 * 2 * 3*2; i++) {
            vertexColors[j++] = c1;
            vertexColors[j++] = c2;
            vertexColors[j++] = c3;
 
        }
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);
 
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);
 
 
    }
    void setwxwy(float x, float y) { wTx = x; wTy = y; }
    vec4 getwxwy() { return vec4(wTx, wTy, 0, 0); }
 
    void setV(vec4 ve) {
        v = ve;
        
        
    }
    void Animate(float t) {
        sx = sinf(t);
        sy = cosf(t)+0.1f;
        sx1 = -sinf(t);
        sy1 = cosf(t)+0.1f;
    }
    void gravityvec(Triangle t, float time)
    {
        vec4 r = t.getwxwy()- getwxwy();
        float l = r.length();
        vec4 force = r*((mass*t.mass) / powf(l,2) *( 6.6730f / 10000000.0f))- v*0.005f;
        vec4 a = force / mass;
        v = v + a*(time);
        setwxwy(wTx +v.v[0]*(time), wTy + v.v[1]*(time));
    }
 
    void doppler() {
        if (v.length() != 0) {
            float direction =  getwxwy().length() - last;
            float leng = getwxwy().length();
            float alfa = atanf(wTx / wTy);
            float beta = atanf(v.v[1] / v.v[0]);
            float vo = leng*cosf(alfa + beta);
            float vl = leng*sinf(alfa + beta);
            if (direction < 0) {
                if (sqrtf(vo*vo + vl*vl) / 10 > 1)
                    dopler= 0;
                else
                    dopler=  1 - sqrtf(vo*vo + vl*vl) / 10;
            }
            if (direction > 0)
                dopler= -(sqrtf(vo*vo + vl*vl) / 10);
 
 
            last = getwxwy().length();
        }
    }
 
    void Draw() {
        mat4 M(sy, sx, 0, 0,
            sx1, sy1, 0, 0,
            0, 0, 1, 0,
            wTx, wTy, 0, 1);
 
        mat4 MVPTransform = M * camera.V() * camera.P();
 
 
        int location = glGetUniformLocation(shaderProgram, "MVP");
        if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform);
        else printf("uniform MVP cannot be set\n");
 
 
        
        int vertexColorLocation = glGetUniformLocation(shaderProgram, "dopler");
        if (vertexColorLocation >= 0) glUniform1f(vertexColorLocation, dopler);
        else printf("uniform dopler cannot be set\n");
 
    
        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, 7 * 6 * 2);
 
        
    }
};
 
Triangle triangle;
Triangle triangle2;
Triangle triangle3;
 
class LineStrip {
    GLuint vao, vbo;
    float  ts[20];
    vec4  vertexData[100];
    int    nVertices;
    float tension;
    vec4 v;
    int draw;
public:
    LineStrip() {
        nVertices = 0;
        tension = -0.8f;
        draw = 0;
        v = vec4(0, 0, 0, 0);
    }
 
    void Create() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
 
        GLuint vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
 
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
    }
 
 
    void AddPoint(float cX, float cY, float t) {
        if (nVertices >= 19) return;
        draw = 1;
        ts[nVertices] = t;
        vec4 wVertex = vec4(cX, cY, 0, 1) *camera.Pinv() * camera.Vinv();
 
 
        vertexData[nVertices] = wVertex;
        nVertices++;
 
        vertexData[nVertices] = vertexData[0];
        ts[nVertices] = ts[nVertices - 1] + 0.5f;
 
    }
    vec4 seb(int i) {
        
        if (i > 0 && i<nVertices) {
            v = ((vertexData[i] - vertexData[i - 1]) / (ts[i] - ts[i - 1]) + (vertexData[i + 1] - vertexData[i]) / (ts[i + 1] - ts[i]))* (1 - tension) / (float)2;
        }
        if (i == 0) {
            v = ((vertexData[i] - vertexData[nVertices - 1]) / (0.5f) + (vertexData[i + 1] - vertexData[i]) / (ts[i + 1] - ts[i]))* (1 - tension) / (float)2;
        }
        if (i == nVertices) {
            v = ((vertexData[0] - vertexData[i - 1]) / (ts[i] - ts[i - 1]) + (vertexData[1] - vertexData[0]) / (ts[1] - ts[0]))* (1 - tension) / (float)2;
        }
        return v;
    }
 
 
    vec4 trans(float t) {
 
        int i = nVertices;
        for (int j = 1; j <= nVertices; j++) {
            if (ts[j] > t) { i = j - 1; break; }
        }
 
        vec4 a0 = vertexData[i];
        vec4 a1 = seb(i);
        vec4 a2 = (vertexData[i + 1] - vertexData[i])*3.0f / powf(ts[i + 1] - ts[i], 2) - (seb(i + 1) + seb(i) * 2.0f) / (ts[i + 1] - ts[i]);
        vec4 a3 = (vertexData[i] - vertexData[i + 1])*2.0f / powf(ts[i + 1] - ts[i], 3) + (seb(i + 1) + seb(i)) / powf(ts[i + 1] - ts[i], 2);
        return (a0 + a1 *(t - ts[i]) + a2 * powf(t - ts[i], 2) + a3 * powf(t - ts[i], 3));
 
 
    }
 
    void Draw() {
        if (nVertices > 1) {
            mat4 VPTransform = camera.V() * camera.P();
 
            int location = glGetUniformLocation(shaderProgram, "MVP");
            if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
            else printf("uniform MVP cannot be set\n");
 
 
            vec4  valami[1000];
            int db = 0;
            int res = 1000 - 1;
            for (int i = 0; i <= res; i++) {
                float t = ((float)i / res)*(ts[nVertices] - ts[0]) + ts[0];
                valami[db++] = trans(t);
 
            };
 
            float colorvalami[1000 * 5];
            int j = 0;
            for (int i = 0; i < db; i++) {
                colorvalami[j++] = valami[i].v[0];
                colorvalami[j++] = valami[i].v[1];
                colorvalami[j++] = 1;
                colorvalami[j++] = 1;
                colorvalami[j++] = 0;
 
            }
            int vertexColorLocation = glGetUniformLocation(shaderProgram, "dopler");
            if (vertexColorLocation >= 0) glUniform1f(vertexColorLocation,0);
            else printf("uniform dopler cannot be set\n");
 
 
            glBufferData(GL_ARRAY_BUFFER, j * sizeof(float), colorvalami, GL_DYNAMIC_DRAW);
 
            glBindVertexArray(vao);
            glDrawArrays(GL_LINE_STRIP, 0, db);
 
 
        }
 
 
    }
 
    void DrawStar(float sec) {
        if (draw == 0) return;
        float t = ts[0] +
            (((sec - ts[0]) / (ts[nVertices] - ts[0])) - floorf((sec - ts[0]) / (ts[nVertices] - ts[0])))
            * (ts[nVertices] - ts[0]);
        vec4 pos = trans(t);
        triangle.setwxwy(pos.v[0], pos.v[1]);
        triangle.setV(v);
    }
 
};
 
LineStrip lineStrip;
 
void onInitialization() {
    glViewport(0, 0, windowWittth, windowHeight);
 
 
    triangle.Create(0.0f, 0.0f, 15000000.0f, 0.6f, 0.7f, 0.0f, 1.0f,vec4(0.0f,0.0f,0.0f,0.0f));
    triangle2.Create(-2.0f, -8.0f, 0.02f, 0.4f, 0.4f, 0.0f, 0.7f,vec4(1.0f,0.1f,0.0f,0.0f));
    triangle3.Create(-4.0f, -6.0f, 0.05f, 0.3f, 0.3f, 0.0f, 0.5f,vec4(0.1f,1.0f,0.0f,0.0f));
    lineStrip.Create();
 
 
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    if (!vertexShader) {
        printf("Error in vertex shader creation\n");
        exit(1);
    }
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);
    checkShader(vertexShader, "Vertex shader error");
 
 
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    if (!fragmentShader) {
        printf("Error in fragment shader creation\n");
        exit(1);
    }
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);
    checkShader(fragmentShader, "Fragment shader error");
 
 
    shaderProgram = glCreateProgram();
    if (!shaderProgram) {
        printf("Error in shader program creation\n");
        exit(1);
    }
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
 
 
    glBindAttribLocation(shaderProgram, 0, "vertexPosition");
    glBindAttribLocation(shaderProgram, 1, "vertexColor");
 
    glBindFragDataLocation(shaderProgram, 0, "fragmentColor");
    glLinkProgram(shaderProgram);
    checkLinking(shaderProgram);
    glUseProgram(shaderProgram);
}
 
void onExit() {
    glDeleteProgram(shaderProgram);
    printf("exit");
}
 
void onDisplay() {
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    lineStrip.Draw();
    triangle.Draw();
    triangle2.Draw();
    triangle3.Draw();
 
    glutSwapBuffers();
}
 
 
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == ' ') {
        if (camera.getStati()) {
            camera.setStati(0);
            camera.setzoom(10, 10);
            camera.setCenter(triangle.getwTx(), triangle.getwTy());
        }
        else {
            camera.setStati(1);
            camera.setzoom(20, 20);
            camera.setCenter(0, 0);
        }
 
        glutPostRedisplay();
    }
}
 
 
void onKeyboardUp(unsigned char key, int pX, int pY) {
 
}
 
 
void onMouse(int button, int state, int pX, int pY) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        float cX = 2.0f * pX / windowWittth - 1;
        float cY = 1.0f - 2.0f * pY / windowHeight;
        lineStrip.AddPoint(cX, cY, glutGet(GLUT_ELAPSED_TIME) / 1000.0f);
        glutPostRedisplay();
    }
}
 
void onMouseMotion(int pX, int pY) {
}
 
float oldtime = 0;
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME);
    float sec = time / 1000.0f;
    triangle.doppler();
    triangle.Animate(sec);
    triangle2.doppler();
    triangle2.Animate(sec + 0.5f);
    triangle3.doppler();
    triangle3.Animate(sec + 0.3f);
    lineStrip.DrawStar(sec);
    float timepassed = sec - oldtime;
    triangle2.gravityvec(triangle, timepassed);
    triangle3.gravityvec(triangle, timepassed);
    oldtime =sec;
 
    if (!camera.getStati()) {
        camera.setCenter(triangle.getwTx(), triangle.getwTy());
    }
 
    glutPostRedisplay();
}
 
// Idaig modosithatod...
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
int main(int argc, char * argv[]) {
    glutInit(&argc, argv);
#if !defined(__APPLE__)
    glutInitContextVersion(majorVersion, minorVersion);
#endif
    glutInitWindowSize(windowWittth, windowHeight);                // Application window is initially of resolution 600x600
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