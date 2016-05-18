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
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, talalkozneten, stb.) erkezo minden egyeb
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
 
// vertex shader in GLSL
const char *vertexSource = R"(
    #version 130
    precision highp float;
 
            in vec2 vertexPosition;        // variable input from Attrib Array selected by glBindAttribLocation
    out vec2 texcoord;            // output attribute: texture coordinate
 
            void main() {
        texcoord = (vertexPosition + vec2(1, 1))/2;                            // -1,1 to 0,1
        gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1);         // transform to clipping space
    }
)";
 
// fragment shader in GLSL
const char *fragmentSource = R"(
    #version 130
    precision highp float;
 
            uniform sampler2D textureUnit;
    in  vec2 texcoord;            // talalkozpolated texture coordinates
    out vec4 fragmentColor;        // output that goes to the raster memory as told by glBindFragDataLocation
 
            void main() {
        fragmentColor = texture(textureUnit, texcoord); 
    }
)";
 
 
#define EPSILON 0.0001f
#define MAX(a,b) ((a) > (b) ? (a) : (b))
 
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
};
 
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
        if (l < EPSILON)
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
    //vik.wiki
    vec4 toneMap() const {
        // Filmic tonemap
        vec4 vec = vec4(MAX(v[0] - 0.004f, 0.0f), MAX(v[1] - 0.004f, 0.0f), MAX(v[2] - 0.004f, 0.0f));
        vec = (vec*(vec*6.2f + 0.5f)) / (vec*(vec*6.2f + 1.7f) + 0.06f);
        return vec4(powf(vec.v[0], 2.2f), powf(vec.v[1], 2.2f), powf(vec.v[2], 2.2f));
    }
 
    vec4 operator*(const mat4& mat) {
        vec4 result;
        for (int j = 0; j < 4; j++) {
            result.v[j] = 0;
            for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
        }
        return result;
    }
 
};
 
typedef vec4 Color;
 
 
 
// handle of the shader program
unsigned int shaderProgram;
static vec4 background[windowWidth * windowHeight];
 
 
struct Material;
struct Camera;
 
 
struct Ray {
    vec4 origin, direction;
    bool levegoben;
    Ray(vec4 origin = vec4(), vec4 direction = vec4(),bool levegoben=true)
        : origin(origin), direction(direction),levegoben(levegoben) { }
};
 
struct Talalkozas {
    Ray ray;
    vec4 position, normal;
    bool letezik;
    Talalkozas(Ray ray = vec4(), vec4 position = vec4(), vec4 normal = vec4(), bool letezik = false)
        :ray(ray), position(position), normal(normal), letezik(letezik) { }
};
 
struct Object {
    Material *mat;
    Object(Material* m) : mat(m) { }
    virtual ~Object() { }
    virtual Talalkozas sugarTalalkozas(Ray) = 0;
 
    mat4 transformMatrix, inverseMatrix;
 
    vec4 objectCoord(vec4 v)
    {
 
        return v * inverseMatrix;
    }
 
    vec4 normalCoord(vec4 v)
    {
        return v * transformMatrix;
    }
 
    Ray objectCoord(Ray r)
    {
        return Ray(objectCoord(r.origin), objectCoord(r.direction),r.levegoben);
    }
 
    Ray normalCoord(Ray r)
    {
        return Ray(normalCoord(r.origin), normalCoord(r.direction),r.levegoben);
    }
 
    Talalkozas normalCoord(Talalkozas talalkoz)
    {
        Talalkozas newTalalkoz;
        newTalalkoz.ray = normalCoord(talalkoz.ray);
        newTalalkoz.position = normalCoord(talalkoz.position);
        newTalalkoz.normal = (normalCoord(talalkoz.normal)).normalize();
        newTalalkoz.letezik = talalkoz.letezik;
        return newTalalkoz;
    }
 
    void translate(vec4 v) {
    
        mat4 matrix(1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            v.v[0], v.v[1],v.v[2], 1);
        
        transformMatrix = transformMatrix * matrix;
 
        mat4 inverzmatrix(1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            -v.v[0], -v.v[1], -v.v[2], 1);
        inverseMatrix = inverzmatrix * inverseMatrix;
 
 
    }
 
    
    
};
 
 
struct Ellipsoid : public Object {
    float a, b, c;
 
    Ellipsoid(Material* mat,float a,float b,float c): a(a),b(b),c(c), Object(mat) {    }
        
 
    Talalkozas sugarTalalkozas(Ray ray) {
 
        Ray ownRay =objectCoord(ray);
 
 
        vec4& o = ownRay.origin;
        vec4& d = ownRay.direction;
        Talalkozas hit;
 
 
        float ea = d.v[0] * d.v[0] * b*b * c*c + d.v[1] * d.v[1] * a*a * c*c + d.v[2] * d.v[2] * a*a * b*b;
        float eb = 2.0f * (o.v[0] * d.v[0] * b*b * c*c + o.v[1] * d.v[1] * a*a * c*c + o.v[2] * d.v[2] * a*a * b*b);
        float ec = b*b * c*c * o.v[0] * o.v[0] + a*a * c*c * o.v[1] * o.v[1] + a*a * b*b * o.v[2] * o.v[2] - a*a * b*b * c*c;
 
 
        float ed = eb*eb - 4 * ea*ec;
 
        if (ed < 0) {
            return hit;
        }
 
        float t1 = (-eb + sqrtf(ed)) / (2 * ea);
        float t2 = (-eb - sqrtf(ed)) / (2 * ea);
 
        float t = -1;
        if (t1 > 0 && t1 <= t2)
            t = t1;
        else if (t2 > 0)
            t = t2;
 
        if (t > EPSILON) {
            hit.position = ownRay.origin + ownRay.direction * t;
 
            float xn = 2*b*b*c*c*hit.position.v[0];
            float yn = 2*a*a*c*c*hit.position.v[1];
            float zn = 2*a*a*b*b*hit.position.v[2];
            vec4 normal= vec4(xn, yn,zn).normalize();
            hit.normal = normal;
            hit.letezik = true;
            hit.ray = ownRay;
            hit=normalCoord(hit);
        }
        return hit;
    }
};
 
 
struct Light {
    vec4 direction;
    Color color;
    Light(vec4 direction = vec4(), Color color = Color()) : direction(direction), color(color) {}
    virtual bool onLight(Talalkozas talalkozas) { 
        return false; }
    virtual Color getColorDiff( Color diffcolor, Talalkozas talalkoz) { return vec4(); }
    virtual Color getColorSpec( vec4 eye, Talalkozas talalkoz, Color tukorColor, float shininess) {
        return vec4();
    }
    virtual Color getColorRefl(Talalkozas talalkoz, Color tukorColor, float shininess, Color F0) {
        return vec4();
    }
    virtual Color getColorReflFull(Talalkozas talalkoz, Color tukorColor, float shininess, Color F0) {
        return vec4();
    }
    virtual ~Light() { }
};
 
struct Material {
    virtual ~Material() { }
    virtual Color getColor(Talalkozas, Light**, int, int recursio) = 0;
};
 
struct Scene {
    int objsize;
    Object* objects[50];
    int lightsize;
    Light* lights[20];
    vec4 hatter;
    Scene() : objsize(0), lightsize(0) ,hatter(vec4(0.6f, 0.7f, 0.9f)) { }
    void AddLight(Light *l) {
        lights[lightsize++] = l;
    }
    void AddObject(Object *o) {
        objects[objsize++] = o;
    }
 
 
    Talalkozas getElsoTalalkozas(Ray ray, int* index = NULL) {
        Talalkozas elsoTalalkozas;
        float elsoDist;
        int elsoIndex = -1;
 
        for (int i = 0; i < objsize; ++i) {
            Talalkozas talalkoz = objects[i]->sugarTalalkozas(ray);
            if (talalkoz.letezik) {
                float dist = (talalkoz.position - ray.origin).length();
                if (elsoIndex == -1) elsoDist = dist;
                if ( dist <= elsoDist) {
                    elsoTalalkozas = talalkoz;
                    elsoDist = dist;
                    elsoIndex = i;
                }
            }
        }
        if (index!=NULL) {
            *index = elsoIndex;
        }
        return elsoTalalkozas;
    }
 
    Color trace(Ray r, int recursio = 0) {
        if (recursio < 5) {
            int index = -1;
            Talalkozas talalkoz = getElsoTalalkozas(r, &index);
            if (index != -1)
                return objects[index]->mat->getColor(talalkoz, lights, lightsize, recursio);
        }
        return hatter;    
    }
 
    ~Scene() {
        for (int i = 0; i <= objsize; ++i) {
            delete objects[i];
        }
        for (int i = 0; i <= lightsize; ++i) {
            delete lights[i];
        }
    }
    
}; 
Scene scene;
 
 
struct Camera {
    vec4 eye, lookAt, right, up;
 
    //eloadasdiak
    Camera(float latoszog, vec4 _eye, vec4 _lookAt)
        : eye(_eye), lookAt(_lookAt)
    {
 
        vec4 plane_up = vec4(0, 1, 0);
        vec4 camfwd = lookAt - eye;
        float half = camfwd.length()*tanf((latoszog*M_PI / 180.f) / 2.f);
 
        right = half * cross(camfwd, plane_up).normalize();
        up = half * cross(right, camfwd).normalize();
    }
 
 
    void capturePixel(int x, int y) {
        float xpos = (x + 0.5f - windowWidth / 2) / (windowWidth / 2);
        float ypos= (y + 0.5f - windowHeight / 2) / (windowHeight / 2);
 
        vec4 talalkoas = lookAt + xpos* right + ypos* up;
 
        Ray r = { eye, (talalkoas - eye).normalize() };
 
        background[y*windowWidth + x] = scene.trace(r).toneMap();
 
    }
}; 
Camera camera(40.f, vec4(-40.f, 50.f, 0.f), vec4());
 
 
 
struct Ambient : Light {
    Ambient(vec4 direction, Color color) :Light(direction, color) {}
    bool onLight(Talalkozas talalkozas) {
        return true;
    }
    Color getColorDiff( Color diffcolor, Talalkozas talalkoz) {
        return  (this->color) * diffcolor;
    }
    Color getColorSpec( vec4 eye, Talalkozas talalkoz, Color tukorColor, float shininess) {
        return vec4();
    }
    Color getColorRefl( Talalkozas talalkoz, Color tukorColor, float shininess, Color F0) {
        return vec4();
    }
    Color getColorReflFull(Talalkozas talalkoz, Color tukorColor, float shininess,  Color F0) {
        return vec4();
    }
};
 
//eloadasdiak
struct Directional : Light {
 
    Directional(vec4 direction, Color color) :Light(direction, color) {}
    bool onLight(Talalkozas talalkozas) {
        Ray arnyek = Ray(talalkozas.position + EPSILON*talalkozas.normal, direction); 
        Talalkozas arnyekTalalkozas = scene.getElsoTalalkozas(arnyek);
        if (!arnyekTalalkozas.letezik)
            return true;
        return false;
    }
 
    Color getColorDiff( Color diffcolor, Talalkozas talalkoz) {
        float intensity = dot(talalkoz.normal, this->direction.normalize());
        return  intensity*(this->color)* diffcolor;
    }
 
    Color getColorSpec( vec4 eye, Talalkozas talalkoz, Color tukorColor, float shininess) {
        vec4 L = this->direction.normalize();
        vec4 V = (eye - talalkoz.position).normalize();
        vec4 H = (L + V).normalize();
        vec4 N = talalkoz.normal;
        float tukorerosseg = powf(dot(N, H), shininess);
        return tukorerosseg * this->color * tukorColor;
    }
    Color F(vec4 inDir, vec4 normal,Color F0) {
        float cosa = fabs(dot(normal, inDir));
        return F0 + (vec4(1.f, 1.f, 1.f) - F0) * powf(1 - cosa, 5);
    }
    Color getColorRefl( Talalkozas talalkoz, Color tukorColor, float shininess, Color F0) {
        vec4 L = this->direction.normalize();
        vec4 V = -1 * talalkoz.ray.direction;
        vec4 H = (L + V).normalize();
        vec4 N = talalkoz.normal;
        float tukorerosseg = powf(dot(H, N), shininess);
        return  tukorerosseg * F(talalkoz.ray.direction, talalkoz.normal, F0) * this->color * tukorColor;
        
    }
    Color getColorReflFull(Talalkozas talalkoz, Color tukorColor, float shininess, Color F0) {
        vec4 L = this->direction.normalize();
        vec4 V = -1 * talalkoz.ray.direction;
        vec4 H = (L + V).normalize();
        vec4 N = talalkoz.normal;
        float tukorerosseg = powf(dot(H, N), shininess);
        return tukorerosseg * this->color * tukorColor;
        
    }
 
};
 
 
 
struct DiffuseMaterial : public Material {
    Color diffcolor;
 
    DiffuseMaterial(Color color) : diffcolor(color) { }
 
    Color getColor(Talalkozas talalkoz, Light** lights, int lightsize, int rec_level = 0) {
        Color allcolor;
        for (int i = 0; (i < lightsize) ; ++i) {
            Light* light = lights[i];
 
            if (light->onLight(talalkoz)) {
                allcolor = allcolor+ light->getColorDiff(diffcolor, talalkoz);
            }
 
        }
 
        return allcolor;
    }
};
 
struct SpecularMaterial : DiffuseMaterial {
    Color tukorColor;
    float shininess;
 
    SpecularMaterial(Color diffusecolor, Color tukorColor, float shininess)
        : DiffuseMaterial(diffusecolor), tukorColor(tukorColor), shininess(shininess) { }
 
    Color getColor(Talalkozas talalkoz,Light** lights, int lightsize, int rec_level = 0) {
        Color allcolor = DiffuseMaterial::getColor(talalkoz, lights, lightsize);
 
        for (int i = 0; (i < lightsize); ++i) {
            Light* light = lights[i];
 
            if (light->onLight(talalkoz)) {
                allcolor = allcolor+light->getColorSpec(camera.eye,talalkoz,tukorColor,shininess);
            }
        }
 
        return allcolor;
    }
};
 
 
//eloadasdiak
struct ReflectiveMaterial : public Material {
    Color F0, tukorColor;
    float shininess;
 
    ReflectiveMaterial(Color n, Color k, Color tukorColor, float shininess)
        : F0(((n - 1)*(n - 1) + k*k) /
            ((n + 1)*(n + 1) + k*k))
        , tukorColor(tukorColor)
        , shininess(shininess)
    { }
 
 
    vec4 reflect(vec4 inDir, vec4 normal) {
        return inDir - normal * dot(normal, inDir) * 2.0f;
    }
 
    Color F(vec4 inDir, vec4 normal) {
        float cosa = fabs(dot(normal, inDir));
        return F0 + ((vec4(1, 1, 1) - F0) * powf(1 - cosa, 5));
 
    }
    
 
    Color getColor(Talalkozas talalkoz,Light** lights, int lightsize, int recursio) {
        Ray reflected;
        reflected.direction = reflect(talalkoz.ray.direction, talalkoz.normal);
        reflected.origin = talalkoz.position+EPSILON*reflected.direction;
        Color color = F(talalkoz.ray.direction, talalkoz.normal) * scene.trace(reflected, recursio + 1);
 
        return color + getSpecColor(talalkoz, lights, lightsize);
    }
 
    Color getSpecColor(Talalkozas talalkoz,Light** lights, int lightsize) {
        Color allcolor;
        for (int i = 0; i < lightsize; ++i) {
            Light* light = lights[i];
 
            if (light->onLight(talalkoz)) {
                allcolor = allcolor +light->getColorRefl(talalkoz, tukorColor, shininess,F0);
            }
        }
 
        return allcolor;
    }
};
 
 
//eloadasdiak
struct Water : public ReflectiveMaterial {
    float n;
 
    vec4 refract(vec4 inDir, vec4 normal) {
        float ior = n;
        float cosa = -dot(normal, inDir);
        if (cosa < 0) { cosa = -cosa; normal = -1 * normal; ior = 1 / n; }
        float disc = 1 - (1 - cosa * cosa) / ior / ior;
        if (disc < 0) return reflect(inDir, normal);
        return inDir / ior + normal * (cosa / ior - sqrt(disc));
    }
 
 
    Water(float n, Color k, Color tukorColor, float shininess)
        : ReflectiveMaterial(n, k, tukorColor, shininess), n(n)
    { }
 
    Color getColor(Talalkozas talalkoz,Light** lights, int lightsize, int recursio) {
        if (dot(talalkoz.ray.direction, talalkoz.normal) > 0) {
            talalkoz.normal = -1 * talalkoz.normal;
        }
 
        Ray reflected;
        reflected.direction = reflect(talalkoz.ray.direction, talalkoz.normal);
        reflected.origin = talalkoz.position + EPSILON*reflected.direction;
 
        Color reflectedColor, refractedColor;
 
 
        Ray refracted;
        refracted.direction = refract(talalkoz.ray.direction, talalkoz.normal);
        if ((refracted.direction.length())>EPSILON) {
            refracted.origin = talalkoz.position + EPSILON * refracted.direction;
            refracted.levegoben = !talalkoz.ray.levegoben;
            reflectedColor = F(talalkoz.ray.direction, talalkoz.normal) * scene.trace(reflected, recursio + 1)+ getSpecularHighlight(talalkoz, lights, lightsize);
            refractedColor = (vec4(1, 1, 1) - F(talalkoz.ray.direction, talalkoz.normal)) * scene.trace(refracted, recursio + 1);
        }
        else {
            reflectedColor = scene.trace(reflected, recursio + 1)
                + getSpecularHighlight(talalkoz, lights, lightsize);
        }
 
        return reflectedColor + refractedColor;
    }
 
 
    Color getSpecularHighlight(Talalkozas talalkoz, Light** lights, int lightsize) {
        Color allcolor;
        for (int i = 0; i < lightsize; ++i) {
            Light* light = lights[i];
 
            if (light->onLight(talalkoz)) {
                allcolor = allcolor + light->getColorReflFull(talalkoz, tukorColor, shininess, F0);
            }
 
        }
 
        return allcolor;
    }
};
 
 
 
struct Triangle : public Object {
    vec4 a, b, c, normal;
 
    // Az oramutato jarasaval ellentetes ha nem ugy adjuk meg a feny rosszul verodik vissza rola
    // Sunis konyv 171-172 oldal
    Triangle(Material* mat, vec4 a,  vec4 b, vec4 c)
        : Object(mat), a(a), b(b), c(c) {
        normal = cross((b - a).normalize(), (c - a).normalize()).normalize();
    }
 
    Talalkozas sugarTalalkozas(Ray ray) {
 
        float v = dot(ray.direction, normal);
        float s = dot(a - ray.origin, normal);
        if (v == 0) return Talalkozas();
        float t = s / v;
        if (t<0) return Talalkozas();
        vec4 x = ray.origin + t * ray.direction;
 
        vec4 ab = b - a;
        vec4 ax = x - a;
 
        vec4 bc = c - b;
        vec4 bx = x - b;
 
        vec4 ca = a - c;
        vec4 cx = x - c;
 
        if ((dot(cross(ab, ax), normal) >= 0) && (dot(cross(bc, bx), normal) >= 0) && (dot(cross(ca, cx), normal) >= 0)){
            return Talalkozas(ray, x, normal, true);
        }
        return Talalkozas();
    }
};
 
struct TriangleWater : public Object {
    vec4 a, b, c;
    vec4 normal;
    Triangle tri1;
    Triangle tri2;
    Triangle tri3;
    Triangle tri4;
 
    TriangleWater(Material* mat, vec4 a, vec4 b, vec4 c)
        : Object(mat), a(a), b(b), c(c),
 
        tri1(new DiffuseMaterial(Color(0.0f, 0.4f, 0.1f)), vec4(-500, a.v[1] + 2, +500), vec4(500, a.v[1] + 2,500), vec4(-500, a.v[1] + 2, -500)),    
        tri3(new DiffuseMaterial(Color(0.0f, 0.4f, 0.1f)), vec4(+500, a.v[1] + 2, -500), vec4(-500, a.v[1] + 2, -500), vec4(500, a.v[1] + 2, 500)),
        tri2(new DiffuseMaterial(Color(0.0f, 0.4f, 0.1f)), vec4(-500, a.v[1] - 2, +500), vec4(500, a.v[1] - 2, 500), vec4(-500, a.v[1] - 2, -500)),
        tri4(new DiffuseMaterial(Color(0.0f, 0.4f, 0.1f)), vec4(+500, a.v[1] - 2, -500), vec4(-500, a.v[1] - 2, -500), vec4(500, a.v[1] - 2, 500)) {
        normal = cross((b - a).normalize(), (c - a).normalize()).normalize();
    }
    float gorbe(vec4 v) {
        return cosf(sqrtf(((v.v[0] - 10)*(v.v[0] - 10) + v.v[2] * v.v[2])) * 2)
            + cosf(sqrtf(((v.v[0] + 10)*(v.v[0] + 10) + v.v[2] * v.v[2])) * 2) - v.v[1];
 
    }
 
    Talalkozas sugarTalalkozas(Ray ray) {
        Talalkozas tal1 = tri1.sugarTalalkozas(ray);
        Talalkozas tal2 = tri2.sugarTalalkozas(ray);
        Talalkozas tal3 = tri3.sugarTalalkozas(ray);
        Talalkozas tal4 = tri4.sugarTalalkozas(ray);
        
        if (tal1.letezik&&tal2.letezik || tal1.letezik&&tal4.letezik || tal3.letezik&&tal2.letezik || tal3.letezik&&tal4.letezik) {
            vec4 a1 = 0;
            vec4 b1 = 0;
            if (tal1.letezik&&tal2.letezik) {
                a1 = tal1.position;
                b1 = tal2.position;
            }
            else if (tal1.letezik&&tal4.letezik) {
                 a1 = tal1.position;
                 b1 = tal4.position;
            }
            else if (tal3.letezik&&tal4.letezik) {
                 a1 = tal3.position;
                 b1 = tal4.position;
            }
            else if (tal3.letezik&&tal2.letezik) {
                 a1 = tal3.position;
                 b1 = tal2.position;
            }
            //regula falsi
            vec4 c1 = b1 - gorbe(b1)*(b1 - a1) / (gorbe(b1) - gorbe(a1));
            int db = 0;
            while (fabs(gorbe(c1)) > EPSILON) {
                db++;
                if (db >= 5000) {
                    break;
                }
                if (gorbe(c1)  > 0)
                    b1 = c1;
                else  
                    a1 = c1;
 
                if (b1.v[0] == a1.v[0] && b1.v[1] == a1.v[1] && b1.v[2] == a1.v[2])
                    c1 = a1;
                else
                    c1 = b1 - gorbe(b1)*(b1 - a1) / (gorbe(b1) - gorbe(a1));
                    
            }
    
            a = vec4(a.v[0],c1.v[1],a.v[2]);
            b = vec4(b.v[0],c1.v[1],b.v[2]);
            c = vec4(c.v[0],c1.v[1],c.v[2]);
 
            vec4 ab = b - a;
            vec4 ax = c1 - a;
 
            vec4 bc = c - b;
            vec4 bx = c1 - b;
 
            vec4 ca = a - c;
            vec4 cx = c1 - c;
 
            if ((dot(cross(ab, ax), normal) >= 0) && (dot(cross(bc, bx), normal) >= 0) && (dot(cross(ca, cx), normal) >= 0)) {
 
                vec4 dx(1, -2*(c1.v[0] - 10)*sinf(2*sqrtf((c1.v[0] - 10)*(c1.v[0] - 10) + c1.v[2] * c1.v[2])) / sqrtf((c1.v[0] - 10)*(c1.v[0] - 10) + c1.v[2] * c1.v[2])
                    - 2*(c1.v[0] + 10)*sinf(2*sqrtf((c1.v[0] + 10)*(c1.v[0] + 10) + c1.v[2] * c1.v[2])) / sqrtf((c1.v[0] + 10)*(c1.v[0] + 10) + c1.v[2] * c1.v[2]), 0);
                vec4 dz(0,
                    -2*(c1.v[2])*sinf(2*sqrtf((c1.v[0] - 10)*(c1.v[0] - 10) + c1.v[2] * c1.v[2])) / sqrtf((c1.v[0] - 10)*(c1.v[0] - 10) + c1.v[2] * c1.v[2])
                    - 2*(c1.v[2])*sinf(2*sqrtf((c1.v[0] + 10)*(c1.v[0] + 10) + c1.v[2] * c1.v[2])) / sqrtf((c1.v[0] + 10)*(c1.v[0] + 10) + c1.v[2] * c1.v[2]), 1);
            
                vec4 normal = cross(dx,dz).normalize();
                return Talalkozas(ray, c1, normal, true);
            }
        }
        return Talalkozas();
    }
};
 
 
class FullScreenTexturedQuad {
    unsigned int vao, textureId;    // vertex array object id and texture id
public:
    void Create(vec4 image[windowWidth * windowHeight]) {
        glGenVertexArrays(1, &vao);    // create 1 vertex array object
        glBindVertexArray(vao);        // make it active
 
        unsigned int vbo;        // vertex buffer objects
        glGenBuffers(1, &vbo);    // Generate 1 vertex buffer objects
 
                                // vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        static float vertexCoords[] = { -1, -1,   1, -1,  -1, 1,
            1, -1,   1,  1,  -1, 1 };    // two triangles forming a quad
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);       // copy to that part of the memory which is not modified 
                                                                                               // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
 
                                                                      // Create objects by setting up their vertex data on the GPU
        glGenTextures(1, &textureId);                  // id generation
        glBindTexture(GL_TEXTURE_2D, textureId);    // binding
 
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, image); // To GPU
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // sampling
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
 
    void Draw() {
        glBindVertexArray(vao);    // make the vao and its vbos active playing the role of the data source
        int location = glGetUniformLocation(shaderProgram, "textureUnit");
        if (location >= 0) {
            glUniform1i(location, 0);        // texture sampling unit is TEXTURE0
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, textureId);    // connect the texture to the sampler
        }
        glDrawArrays(GL_TRIANGLES, 0, 6);    // draw two triangles forming a quad
    }
};
 
// The virtual world: single quad
FullScreenTexturedQuad fullScreenTexturedQuad;
 
// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    Ambient amb( vec4(), Color(0.6f, 0.7f, 0.9f));
    Directional direct(vec4(1.1f, 100.f, 0.f), Color(4.f, 4.f, 4.f));
    scene.AddLight(&amb);
    scene.AddLight(&direct);
 
    DiffuseMaterial green(Color(0.0f, 0.4f, 0.1f));
    
    SpecularMaterial blue(Color(0.0f, 0.4f, 1.0f), Color(60, 60, 60),1020);
    Water water(1.3f, Color(1, 1, 1), Color(1, 1, 1), 1020);
    ReflectiveMaterial silver(Color(0.14, 0.16, 0.13), Color(4.1, 2.3, 3.1), 1, 1020);
    ReflectiveMaterial gold(Color(0.17, 0.35, 1.6), Color(3.1, 2.7, 1.9), 1,1020);
 
    Ellipsoid ellipsoid =Ellipsoid(&gold,1.7,3.5,1.7);
    ellipsoid.translate(vec4(0.1f, +0.1f, 0.0f));
    scene.AddObject(&ellipsoid);
    
 
 
 
    float m = 3.f / 2.f;
    float s = 10.f / 2.f;
    float h = 50.f / 2.f;
 
    scene.AddObject(new TriangleWater(&water, vec4(-h, 0, +s), vec4(+h, 0, +s), vec4(-h, 0, -s)));
    
    scene.AddObject(new TriangleWater(&water, vec4(+h, 0, -s), vec4(-h, 0, -s), vec4(+h, 0, +s)));
 
    scene.AddObject(new Triangle(&green, vec4(+h, m, +s), vec4(-h, m, +s), vec4(h, m, s + 500)));
    scene.AddObject(new Triangle(&green, vec4(-h, m, +s), vec4(-h, m, s + 500), vec4(h, m, s + 500)));
 
    scene.AddObject(new Triangle(&green, vec4(h, m, s + 500), vec4(h + 500, m, s + 500), vec4(h, m, -s - 500)));
    scene.AddObject(new Triangle(&green, vec4(h, m, -s - 500), vec4(h + 500, m, +s + 500), vec4(h + 500, m, -s - 500)));
 
    scene.AddObject(new Triangle(&green, vec4(+h, m, -s), vec4(h, m, -s - 500), vec4(-h, m, -s)));
    scene.AddObject(new Triangle(&green, vec4(-h, m, -s), vec4(h, m, -s - 500), vec4(-h, m, -s - 500)));
 
    scene.AddObject(new Triangle(&green, vec4(-h, m, s + 500), vec4(-h, m, -s - 500), vec4(-h - 500, m, s + 500)));
    scene.AddObject(new Triangle(&green, vec4(-h, m, -s - 500), vec4(-h - 500, m, -s - 500), vec4(-h - 500, m, +s + 500)));
 
 
 
    scene.AddObject(new Triangle(&blue, vec4(+h, -m, -s), vec4(-h, -m, -s), vec4(-h, +m, -s)));
    scene.AddObject(new Triangle(&blue, vec4(-h, +m, -s), vec4(+h, +m, -s), vec4(+h, -m, -s)));
 
    scene.AddObject(new Triangle(&blue, vec4(+h, -m, +s), vec4(-h, -m, +s), vec4(-h, +m, +s)));
    scene.AddObject(new Triangle(&blue, vec4(-h, +m, +s), vec4(+h, +m, +s), vec4(+h, -m, +s)));
 
    scene.AddObject(new Triangle(&blue, vec4(+h, -m, -s), vec4(+h, -m, +s), vec4(+h, +m, +s)));
    scene.AddObject(new Triangle(&blue, vec4(+h, +m, +s), vec4(+h, +m, -s), vec4(+h, -m, -s)));
 
    scene.AddObject(new Triangle(&blue, vec4(-h, -m, -s), vec4(-h, -m, +s), vec4(-h, +m, +s)));
    scene.AddObject(new Triangle(&blue, vec4(-h, +m, +s), vec4(-h, +m, -s), vec4(-h, -m, -s)));
 
    scene.AddObject(new Triangle(&blue, vec4(-h, -m, +s), vec4(+h, -m, +s), vec4(-h, -m, -s)));
    scene.AddObject(new Triangle(&blue, vec4(+h, -m, -s), vec4(-h, -m, -s), vec4(+h, -m, +s)));
 
        
 
 
    float lastx = 0; float lasty = 0;
    float llastx = 0; float llasty = 0;
    float x = 0; float y = 0;
    bool first = true;
    int db = 0;
    for (int i = 0; i < 4; i++) {
        float angle = (2*(float)M_PI / 3.0f)*i;
        x= 0 +5.5 / 2.0f * sinf(angle);
        y = 0 +5.5 / 2.0f * cosf(angle);
        if (!first)
                scene.AddObject(new Triangle(&silver, vec4(+20, 0+2, 0+3), vec4(+20 + x,0-1, y+3), vec4(+20 + lastx, 0-1, lasty + 3)));
                scene.AddObject(new Triangle(&silver,vec4(+20+lastx,0+5.1, lasty+3), vec4(+20, 0 + 2, 0 + 3), vec4(+20+ x, 0+5.1, y+3)));
        first = false;
        llastx = lastx;
        llasty = lasty;
        lastx = x;
        lasty = y;
    }
    scene.AddObject(new Triangle(&silver, vec4(-20+x, 0-1, y+3), vec4(-20+lastx, 0-1, lasty+3), vec4(-20+llastx, 0-1, llasty+3)));
    scene.AddObject(new Triangle(&silver, vec4(-20+x, 0 + 5.1, y+3), vec4(-20 + llastx, 0 + 5.1, llasty + 3), vec4(-20+ lastx, 0 + 5.1, lasty+3)));
    
 
    
    
    for (int x = 0; x < windowWidth; x++) {
        for (int y = 0; y < windowHeight; y++) {
            camera.capturePixel(x, y);
            
        }
    }
    
    fullScreenTexturedQuad.Create(background);
 
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
    glBindAttribLocation(shaderProgram, 0, "vertexPosition"); // vertexPosition gets values from Attrib Array 0
 
                                                              // Connect the fragmentColor to the frame buffer memory
    glBindFragDataLocation(shaderProgram, 0, "fragmentColor");    // fragmentColor goes to the frame buffer memory
 
                                                                // program packaging
    glLinkProgram(shaderProgram);
    checkLinking(shaderProgram);
    // make this program run
    glUseProgram(shaderProgram);
}
 
void onExit() {
    glDeleteProgram(shaderProgram);
    printf("exit");
}
 
// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0, 0, 0, 0);                            // background color 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    fullScreenTexturedQuad.Draw();
    glutSwapBuffers();                                    // exchange the two buffers
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
    }
}
 
// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}
 
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
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