# grafika

#Első feladat

Twinkle, twinkle little star...

A 2D virtuális világ három, legalább 7-ágú forgó és pulzáló csillagot tartalmaz, amelyek színe különböző. A legfényesebb csillag az egérklikkek által kijelölt zárt, -0.8-as tenziójú Catmull-Rom spline-t követi (azaz egy kontrollpontban a sebesség az előző és következő szegmensek átlagsebességeinek a 0.9-de). A Catmull-Rom spline csomóértékei a gomblenyomáskori idők. A legutolsó és legelső kontrollpont között 0.5 sec telik el, majd a csillag periodikusan újra bejárja a görbét. A görbe mindenhol folytonosan deriválható, a legelső pontban is (azaz nem törik meg itt sem).

A másik két csillagot a legfényesebb csillag a Newton féle gravitációs erővel vonzza, azaz mozgatja, miközben sebességgel arányos surlódási erő fékezi őket. A gravitációs konstanst úgy kell megválasztani, hogy a mozgás élvezhető legyen, azaz a csillagok a képernyőn maradjanak.

SPACE lenyomására a virtuális kameránkat a fényes csillaghoz kapcsolhatjuk, egyébként a kamera statikus.

Motivált hallgatóknak: A mozgó csillagok a Doppler effektus szerint változtassák a színüket ("vörös eltolódás"), feltételezve, hogy a megfigyelő a világkoordinátarendszer origójában ül.

Beadási határidő: 2016. 03. 28. 23:59
Második feladat

A virtuális világ a budapesti (Budapest az északi szélesség 47 fokán van, a keringési sík és a föld forgástengelye 23 fokot zár be) kertjének észak-déli tájolású 50mx10mx3m-es úszómedencéjét tartalmazza.

A kertje füvesített, zöld színű, diffúz sík. Az úszómedence téglatest alakú, a medence halványkék, diffúz-spekuláris anyagú.

A medence optikailag sima felületű vízzel van felöltve (a víz törésmutatója 1.3 a látható tartományban, a kioltási tényező elhanyagolható). A víz felszínén, a medence két pontjából egy-egy körhullám indul ki, amelyek amplitúdója külön-külön állandó és nem verődik vissza a medence oldaláról (feszített víztükör). A hullámok terjedési sebessége 1 m/s, a frekvenciájuk 0.5 Hz, a két hullám szuperponálódik (össze kell adni két szinuszhullámot, ahol a fázisok a rezgések kezdőpontjától mért távolsággal arányosak). A sugár-vízfelület metszéspont számításhoz a "regula falsi" gyökkereső eljárás használandó.

A vízben két optikailag sima felületű tárgy úszik, az egyik aranyozott (n/k az r,g,b hullámhosszain: 0.17/3.1, 0.35/2.7, 1.5/1.9), felületének implicit egyenlete nem lineáris (pl. kvadratikus), a másik ezüstözött (n/k az r,g,b hullámhosszain: 0.14/4.1, 0.16/2.3, 0.13/3.1) poligonháló.

Az objektumokat a fehér Nap (sugara 7*10^8 m, a fény 8 perc alatt ér a földre, amelyekből kiszámítható az a határszög, amelynél ha kisebb a sugárirány és napirány közötti szög, akkor a nap sugársűrűségét kell használni), valamint a halványkék ég világítja meg. A rücskös felületek szempontjából a Nap irány, az ég pedig ambiens fényforrásnak tekinthető. A Napot eltaláló sima felületekről induló sugár sugársűrűsége százszorosa az eget eltaláló sugár sugársűrűségének. A Nap és az ég sugársűrűségét úgy kell beállítani, hogy kép pixelekre és hullámhosszakra vett átlagos sugársűrűsége 0.5 W/m^2/st legyen.

Jelenítse meg a színtér június 21.-én (nyári napforduló), 12.00 órakori állapotát CPU-n az onInitialization-ban végrehajtott rekurzív sugárkövetéssel, és az onDisplay-ben jelenítse meg a képet annak egy teljes képernyős négyszögre textúrázásával és kirajzolásával (példa program a sugárkövetés diák mellett)!

Beadási határidő: 2016. 04. 17. 23:59
Harmadik feladat

Tóruszba zárt spiderman

Egy procedurálisan textúrázott, diffúz-spekuláris tórusz belsejében egy ugyancsak procedurálisan textúrázott golyó gördül, egy cián és egy sárga fényű fényforrás labda pattog a tórusszal rugalmasan ütközve és a mechanikai energiáját megtartva, valamint spiderman avatárunk várja a sorsát, akinek a szemszögéből követjük az eseményeket és gyönyörködünk a Phong árnyalt színtérben. A golyó 3 másodperces periodiusidővel gördül végig a tóruszon. A tórusz rögzített, golyó antigravitációs készülékkel van ellátva, így nem szakad el a tórusz falától. A többiekre pedig hat a homogén nehézségi erőtér. A golyó pályája a tórusz falán periodikus és nem kör. Spiderman mindig a golyó irányába néz, mert szeretnénk elkerülni, hogy a golyó legázolja. Ha a tórusz belső falának egy pontjára mutatunk a bal egérgomb lenyomásával, akkor oda egy nem zérus nyugalmi hosszúságú gumikötelet lő ki, ami megnyúlás esetén a Hooke törvény és a dinamika alaptörvénye szerint magával rántja, így a közeledő golyó elől el tud ugrani. Minden újabb gumilövés a régit oldja.

Beadási határidő: 2016. 05. 17. 23:59
