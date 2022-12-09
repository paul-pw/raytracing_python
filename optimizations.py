import numpy as np
import numpy.typing as npt
from typing import NamedTuple, Tuple
from abc import ABC, abstractmethod
from typing import Optional
import matplotlib.pyplot as plt
import time

Vec = npt.NDArray[np.float64]
Color = npt.NDArray[np.float64]




def correctAndShowImage(img):
    m = np.max(img)
    i = np.power((img/m), 0.45)
    plt.imshow(i)


def randomOnUnitSphere() -> Vec:
    randNumber = np.random.randn(3)
    return randNumber / np.linalg.norm(randNumber)


def reflect(inputVector: Vec, normal: Vec) -> Vec:
    return inputVector - 2 * np.dot(inputVector, normal) * normal


def rayAt(ray, t):
    return ray[0] + t * ray[1]




class ImageSize(NamedTuple):
    width: int
    height: int


class Material(ABC):
    @abstractmethod
    def outRay(self, inRay, normal: Vec, hitPoint: Vec):
        pass

# ======================
def normalizeVector(vec: Vec, useAxis=None):
    return vec/np.linalg.norm(vec, axis=useAxis, keepdims=True)

def raysAt(rays, t):
    return rays[:, 0] + t[:, None] * rays[:, 1]

class Camera:
    def __init__(self, imageSize: ImageSize, position: Vec, cameraDirection: Vec, upDirection: Vec) -> None:
        assert 0 < imageSize.width and 0 < imageSize.height, f"image cant have negative Dimestions, was {imageSize}"
        # setup
        self.imageSize = imageSize
        # sensor plane is 1 unit diagonal calculate the size of a pixel within 1 unit diagonal sensor
        self.unitPerPixel = np.sqrt(
            self.imageSize.width ** 2 + self.imageSize.height ** 2)
        self.aperturePosition = position
        self.sensorPosition = position+cameraDirection
        self.up = normalizeVector(upDirection)
        self.left = normalizeVector(np.cross(cameraDirection, self.up))

    def _getRay(self, pixelX: int, pixelY: int):
        assert 0 <= pixelX < self.imageSize.width, f"pixelX has to be in Image, was {pixelX}"
        assert 0 <= pixelY < self.imageSize.height, f"pixelY has to be in Image, was {pixelY}"

        # set pixel position from 2d to 3d and add random offset within pixel (for anti aliasing)
        x = -self.imageSize.width/2 + pixelX + np.random.rand()
        y = -self.imageSize.height/2 + pixelY + np.random.rand()
        pixelPos = self.sensorPosition - self.left * \
            (x/self.unitPerPixel) - self.up*(y/self.unitPerPixel)

        # calculate ray direction
        rayDirection = normalizeVector(pixelPos - self.aperturePosition)

        # return Ray
        return np.array([self.aperturePosition, rayDirection, np.array([1, 1, 1])])

# ===============
class Object(ABC):
    @abstractmethod
    def hit(self, rays):
        pass

def elemwiseDot(a, b):
    return np.sum(a*b, axis=1)

class Circle(Object):
    def __init__(self, center: Vec, radius: float, material: Material) -> None:
        self.center = center
        self.radius = radius
        self.material = material
        super().__init__()


    def hit(self, rays):
        locations = rays[:, 0]
        directions = rays[:, 1]

        # calculate b, c and delta
        a = 1
        b = 2 * elemwiseDot(directions, (locations - self.center))
        c = np.linalg.norm(locations - self.center,
                           axis=1) ** 2 - self.radius ** 2
        delta = b ** 2 - 4*a*c
        filter = np.full(c.shape, False)
        filter[delta > 0] = True
        delta = delta[filter]
        b = b[filter]
        c = c[filter]

        # calculate t1 and t2
        t1 = (-b + np.sqrt(delta))/(2*a)
        t2 = (-b - np.sqrt(delta))/(2*a)

        # take the correct t
        tn = np.minimum(t1, t2)
        tn = np.where(tn > 0.0, tn, np.maximum(t1, t2))

        # init output variables
        t = np.full(rays.shape[0], -1.0)
        hitPoints = np.zeros((rays.shape[0], 3))
        normals = np.zeros((rays.shape[0], 3))
        material = np.full((rays.shape[0]), np.array(
            [(self.material)], dtype="object"))

        # set output variables
        hitPoints[filter] = raysAt(rays[filter], tn)
        normals[filter] = normalizeVector(hitPoints[filter] - self.center, 1)
        t[filter] = tn
        return (t, normals, hitPoints, material)
# ============

class Triangle(Object):
    def __init__(self, abc, material: Material) -> None:
        self.a = abc[0]
        self.b = abc[1]
        self.c = abc[2]
        self.ab = self.b - self.a
        self.bc = self.c - self.b
        self.ca = self.a - self.c
        ac = self.c - self.a
        self.normal = normalizeVector(np.cross(self.ab, ac))
        self.material = material
        super().__init__()

    def hit(self, rays):
        locations = rays[:, 0]
        directions = rays[:, 1]
        filter = np.full(rays.shape[0],True)
        t = np.full(rays.shape[0], -1.0)
        hitPoints = np.zeros((rays.shape[0], 3))
        normals = np.full((rays.shape[0], 3), self.normal)
        material = np.full((rays.shape[0]), np.array(
            [(self.material)], dtype="object"))


        dn = elemwiseDot(directions, self.normal)
        
        # Filter out all values where there is no Intersection with the plane
        filter[dn==0] = False
        
        # calculate all Values for t
        print(self.a,self.normal, directions,filter, self.normal)
        t[filter] = elemwiseDot(self.a - locations[filter], self.normal)/dn
        # calculate the hit point
        hitPoints[filter] = raysAt(rays[filter], t)

        # check if the hitpoint is in the triangle
        abn = elemwiseDot(np.cross(self.ab, hitPoints[filter] - self.a), self.normal)
        filter[filter] &= (abn > 0) 
        bcn = elemwiseDot(np.cross(self.bc, hitPoints[filter] - self.b), self.normal)
        filter[filter] &= (bcn > 0)
        can = elemwiseDot(
            np.cross(self.ca, hitPoints[filter] - self.a), self.normal)
        filter[filter] &= (can > 0)
        
        # filter out all invalid entrys
        t[np.invert(filter)] = -1

        return (t, normals, hitPoints, material)



class Lambert(Material):
    def __init__(self, diffuseColor,  specularColor=np.array([0, 0, 0]), diffuseVsSpecular=1, specularIndex=0) -> None:
        self.diffuseColor = diffuseColor
        self.specularColor = specularColor
        self.diffuseVsSpecular = diffuseVsSpecular
        self.specularIndex = specularIndex
        super().__init__()

    def outRay(self, inRay, normal: Vec, hitPoint: Vec):
        rayDirection = inRay[1]
        rayColor = inRay[2]

        # Auswahl diffus oder specular (zahl zwischen [0,1])
        select = np.random.rand(1)

        # Ray werte inizialisieren
        location = hitPoint
        color = np.zeros((3))
        direction = np.zeros((3))

        # Werte die sonnst genutzt werden inizialisieren/normieren
        randomOnSphere = randomOnUnitSphere()
        normal = normalizeVector(normal)

        # Reflektionsanteil berechnen
        if (select > self.diffuseVsSpecular):
            reflected = reflect(normalizeVector(rayDirection), normal)
            direction = reflected + self.specularIndex * randomOnSphere
            color = rayColor * self.specularColor
            if (np.dot(direction, normal) < 0):
                direction = reflected

        # diffusen anteil berechnen
        if (select <= self.diffuseVsSpecular):
            point = randomOnSphere + normal + hitPoint
            direction = point - hitPoint
            color = rayColor * self.diffuseColor

        # Werte normieren
        direction = normalizeVector(direction)

        # neuen Ray zurückgeben
        return np.array([location, direction, color])


class Emissive(Material):
    def __init__(self, color) -> None:
        self.color = color
        super().__init__()

    def outRay(self, inRay, normal: Vec, hitPoint: Vec):
        rayColor = inRay[2]

        direction = np.array([0, 0, 0])
        location = np.array([0, 0, 0])
        color = rayColor * self.color
        return np.array([location, direction, color])

# =============


def generateRays(camera):
    width = camera.imageSize.width
    height = camera.imageSize.height
    rays = np.zeros((height*width, 3, 3))
    for x in range(width):
        for y in range(height):
            rays[x+y*width] = camera._getRay(x, y)
    return rays


def hitObject(obj, rays):
    t, normals, hitPoints, material = obj.hit(rays)
    hits = np.array([(tn, n, m) for tn, n, m in zip(t, normals, material)],
                    dtype="float64, object, object")
    return hits

def filterMin(a, b):
    if (a[0] > 0):
        if (b[0] > 0):
            if (a[0] < b[0]):
                return a
            return b
        return a
    return b


def rayFromHit(ray, hit):
    return hit[2].outRay(ray, hit[1], rayAt(ray, hit[0]))


def bounce(rays, objects):
    start = time.time()
    rayFilter = np.full(rays.shape[0], True)
    hits = np.full((rays.shape[0]), np.array(
        [(-1, None, None)], dtype="float64, object, object"))
    end = time.time()
    print("bounce Init: ", end - start)

    start = time.time()
    for obj in objects:
        s1 = time.time()
        newHits = hitObject(obj, rays)
        e1 = time.time()
        print("hit obj: ", e1 - s1)

        s1 = time.time()
        hits = np.array([filterMin(h, nh)
                         for h, nh in zip(hits, newHits)])
        e1 = time.time()
        print("hit filterMin: ", e1 - s1)
    end = time.time()
    print("bounce hit: ", end - start)
    start = time.time()
    rayFilter &= np.array([h[0] > 0 for h in hits])
    
    # if everything is filtered out return
    if(not np.any(rayFilter)):
        return (rays, rayFilter)

    end = time.time()
    print("bounce filter1: ", end - start)

    start = time.time()
    print(hits[rayFilter], rays[rayFilter])
    rays[rayFilter] = np.array([rayFromHit(ray, hit)
                                for hit, ray in zip(hits[rayFilter], rays[rayFilter])])
    end = time.time()
    print("bounce reflect: ", end - start)

    start = time.time()
    rayFilter[rayFilter] &= np.invert(
        np.all(rays[rayFilter][:, 1] == 0, axis=1))
    end = time.time()
    print("bounce filter2: ", end - start)

    return (rays, rayFilter)


def render(camera: Camera, objects: list[Object], bounces: int = 5, itteration: int = 1):
    img = np.zeros((camera.imageSize.width, camera.imageSize.height, 3))

    width = camera.imageSize.width
    height = camera.imageSize.height

    for j in range(itteration):
        start = time.time()
        rays = generateRays(camera)
        rayFilter = np.full(rays.shape[0], True)
        end = time.time()
        print("Init: ", end - start)

        s1 = time.time()
        for i in range(bounces):
            start = time.time()
            rays[rayFilter], rayFilter[rayFilter] = bounce(
                rays[rayFilter], objects)
            # if everything is filtered out break the loop
            if (not np.any(rayFilter)):
                break
            end = time.time()
            print("bounce: ", i, end - start)
        end = time.time()
        print("bounces: ", end - s1)

        start = time.time()
        for x in range(width):
            for y in range(height):
                img[x, y] += rays[x+y*width, 2]
        end = time.time()
        print("output: ", end - start)
        # print(rays)
    return img


objects1: list[Object] = [Circle(np.array([75, 0, 0]), 10, Lambert([0.5, 0.7, 0.5], 1)),
                          Circle(np.array([75, 0, -20]), 10,
                                 Lambert([0, 0, 0], [0.5, 0.7, 0.5], 0, 0)),
                          Circle(np.array([75, 0, 20]), 10, Lambert(
                              [0, 0, 0], [0.5, 0.7, 0.5], 0, 0.5)),
                          Circle(np.array([0, 5000, 0]),
                                 4990, Lambert([0.5, 0.5, 0.7])),
                          Circle(np.array([0, 0, 0]), 5000, Emissive([5, 5, 5]))]
cubeMaterial = Lambert([0.5, 0.7, 0.5])
cube = [
    # front
    Triangle(np.array([[50, 5, -5], [50, 5, 5], [50, -5, -5]]), cubeMaterial),
    Triangle(np.array([[50, 5, 5], [50, -5, 5], [50, -5, -5]]), cubeMaterial),

    # back
    Triangle(np.array([[60, 5, -5], [60, 5, 5], [60, -5, -5]]), cubeMaterial),
    Triangle(np.array([[60, 5, 5], [60, -5, 5], [60, -5, -5]]), cubeMaterial),

    # left
    Triangle(
        np.array([[50, 5, -5], [50, -5, -5], [60, -5, -5]]), cubeMaterial),
    Triangle(np.array([[60, -5, -5], [60, 5, -5], [50, 5, -5]]), cubeMaterial),

    # right
    Triangle(np.array([[50, 5, 5], [50, -5, 5], [60, -5, 5]]), cubeMaterial),
    Triangle(np.array([[60, -5, 5], [60, 5, 5], [50, 5, 5]]), cubeMaterial),

    # top
    Triangle(
        np.array([[50, -5, -5], [50, -5, 5], [60, -5, -5]]), cubeMaterial),
    Triangle(np.array([[50, -5, 5], [60, -5, 5], [60, -5, -5]]), cubeMaterial),

    # bottom
    Triangle(np.array([[50, 5, -5], [50, 5, 5], [60, 5, -5]]), cubeMaterial),
    Triangle(np.array([[50, 5, 5], [60, 5, 5], [60, 5, -5]]), cubeMaterial),


]

objects2: list[Object] = [Circle(np.array([0, 5000, 0]), 4990, Lambert([0.5, 0.5, 0.7])),
                          Circle(np.array([0, 0, 0]), 5000,
                                 Emissive([5, 5, 5])),
                          *cube
                          ]

#camera = Camera(ImageSize(128, 256), np.array([1, 0, 0]), np.array([1, 0, 0]))
#start = time.time()
#img = render(camera, objects2, 5, 5)
#end = time.time()
#print("Time consumed in working: ", end - start)
#correctAndShowImage(img)
#plt.pause(60)
# img = render(camera,objects, 2)
# correctAndShowImage(img)

camera = Camera(ImageSize(128, 128), np.array([278,273,-800]), np.array([0,0,1]), np.array([0,1,0]))
white = Lambert([0.8, 0.8, 0.8])
light = Emissive([500, 500, 500])
green = Lambert([0.5, 1, 0.5])
red = Lambert([1,0.5,0.5])

floor = [
    Triangle(np.array([
        [552.8, 000.0, 000.0],
        [000.0, 000.0, 000.0],
        [000.0, 000.0, 559.2]
    ]), white),
    Triangle(np.array([
        [000.0, 000.0, 559.2],
        [549.6, 000.0, 559.2],
        [552.8, 000.0, 000.0]
    ]), white)
]

light = [
    Triangle(np.array([
        [343.0, 548.8, 227.0],
        [343.0, 548.8, 332.0],
        [213.0, 548.8, 332.0]
    ]), light),
    Triangle(np.array([
        [213.0, 548.8, 332.0],
        [213.0, 548.8, 227.0],
        [343.0, 548.8, 227.0]
    ]), light)
]

ceiling = [
    Triangle(np.array([
        [556.0, 548.8, 000.0],
        [556.0, 548.8, 559.2],
        [000.0, 548.8, 559.2]
    ]), white),
    Triangle(np.array([
        [000.0, 548.8, 559.2],
        [000.0, 548.8, 000.0],
        [556.0, 548.8, 000.0]
    ]), white)
]

backWall = [
    Triangle(np.array([
        [549.6, 000.0, 559.2],
        [000.0, 000.0, 559.2],
        [000.0, 548.8, 559.2]
    ]), white),
    Triangle(np.array([
        [000.0, 548.8, 559.2],
        [556.0, 548.8, 559.2],
        [549.6, 000.0, 559.2]
    ]), white)
]

rightWall = [
    Triangle(np.array([
        [000.0, 000.0, 559.2],
        [000.0, 000.0, 000.0],
        [000.0, 548.8, 000.0]
    ]), green),
    Triangle(np.array([
        [000.0, 548.8, 000.0],
        [000.0, 548.8, 559.2],
        [000.0, 000.0, 559.2]
    ]), green)
]

leftWall = [
    Triangle(np.array([
        [552.8, 000.0, 000.0],
        [549.6, 000.0, 559.2],
        [556.0, 548.8, 559.2]
    ]), red),
    Triangle(np.array([
        [556.0, 548.8, 559.2],
        [556.0, 548.8, 000.0],
        [552.8, 000.0, 000.0]
    ]), red)
]


objects: list[Object] = [*rightWall]

img = render(camera,objects, 2)