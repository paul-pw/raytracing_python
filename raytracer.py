# %%

import numpy as np
import numpy.typing as npt
from typing import NamedTuple, Tuple
from abc import ABC, abstractmethod
from typing import Optional
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import time

Vec = npt.NDArray[np.float64]
Color = npt.NDArray[np.float64]


class ImageSize(NamedTuple):
    width: int
    height: int


def randomOnUnitSphere() -> Vec:
    randNumber = np.random.randn(3)
    return randNumber / np.linalg.norm(randNumber)


def reflect(inputVector: Vec, normal: Vec) -> Vec:
    return inputVector - 2 * np.dot(inputVector, normal) * normal


def correctAndShowImage(img):
    m = np.max(img)
    i = np.power((img/m), 0.45)
    plt.imshow(i)


def rayAt(ray, t):
    return ray[0] + t * ray[1]


class Camera:
    def __init__(
        self, imageSize: ImageSize,
        # set sane defaults
        position: Vec = np.array([0, 0, 0]), direction: Vec = np.array([1, 0, 0]), up: Vec = np.array([0, 0, 1])
    ) -> None:
        assert 0 < imageSize.width and 0 < imageSize.height, f"image cant have negative Dimestions, was {imageSize}"
        # setup
        self.imageSize = imageSize
        # sensor plane is 1 unit diagonal calculate the size of a pixel within 1 unit diagonal sensor
        self.unitPerPixel = np.sqrt(
            self.imageSize.width ** 2 + self.imageSize.height ** 2)

        # set sensor and aperture position (sensor is in front of aperture with direction)
        self.sensorPosition = position+direction
        self.aperturePosition = position

        # set the up and left values of the sensor (where the top and the left side of the sensor is)
        self.up = normalizeVector(up)
        self.left = normalizeVector(np.cross(direction, self.up))

    def _getRay(self, pixelX: int, pixelY: int):
        assert 0 <= pixelX < self.imageSize.width, f"pixelX has to be in Image, was {pixelX}"
        assert 0 <= pixelY < self.imageSize.height, f"pixelY has to be in Image, was {pixelY}"

        # set pixel position from 2d to 3d and add random offset within pixel (for anti aliasing)
        x = -self.imageSize.width/2 + pixelX + np.random.rand()
        y = -self.imageSize.height/2 + pixelY + np.random.rand()

        # set the Pixel Position in 3D Space using the left and the up vectors
        pixelPos = self.sensorPosition - self.left * \
            (x/self.unitPerPixel) - self.up*(y/self.unitPerPixel)

        # calculate ray direction
        rayDirection = normalizeVector(pixelPos - self.aperturePosition)

        # return Ray
        return np.array([self.aperturePosition, rayDirection, np.array([1, 1, 1])])


class Material(ABC):
    @abstractmethod
    def outRay(self, inRay, normal: Vec, hitPoint: Vec):
        pass


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


def elemwiseDot(a, b):
    return np.sum(a*b, axis=1)

# normalize multiple vectors


def normalizeVector(vec: Vec, useAxis=None):
    return vec/np.linalg.norm(vec, axis=useAxis, keepdims=True)

# rayAt but for more Rays


def raysAt(rays, t):
    return rays[:, 0] + t[:, None] * rays[:, 1]


class Object(ABC):
    @abstractmethod
    def hit(self, rays):
        pass


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
        # initialize values
        locations = rays[:, 0]
        directions = rays[:, 1]
        filter = np.full(rays.shape[0], True)
        t = np.full(rays.shape[0], -1.0)
        hitPoints = np.zeros((rays.shape[0], 3))
        normals = np.full((rays.shape[0], 3), self.normal)
        material = np.full((rays.shape[0]), np.array(
            [(self.material)], dtype="object"))

        # calculate dn
        dn = elemwiseDot(directions, self.normal)

        # Filter out all values where there is no Intersection with the plane
        filter[dn == 0] = False

        # calculate all Values for t
        t[filter] = elemwiseDot(self.a - locations[filter], self.normal)/dn
        # calculate the hit point
        hitPoints[filter] = raysAt(rays[filter], t)

        # check if the hitpoint is in the triangle
        abn = elemwiseDot(
            np.cross(self.ab, hitPoints[filter] - self.a), self.normal)
        filter[filter] &= (abn > 0)
        bcn = elemwiseDot(
            np.cross(self.bc, hitPoints[filter] - self.b), self.normal)
        filter[filter] &= (bcn > 0)
        can = elemwiseDot(
            np.cross(self.ca, hitPoints[filter] - self.a), self.normal)
        filter[filter] &= (can > 0)

        # filter out all invalid entrys
        t[np.invert(filter)] = -1

        return (t, normals, hitPoints, material)


# generates all the Rays at once
def generateRays(camera, itterations):
    width = camera.imageSize.width
    height = camera.imageSize.height
    rays = np.zeros((height*width*itterations, 3, 3))
    for x in range(width):
        for y in range(height):
            for j in range(itterations):
                rays[x+y*width+j*width*height] = camera._getRay(x, y)
    return rays

# create an image from all the rays


def createImage(rays, width, height, itterations):
    img = np.zeros((width, height, 3))
    for x in range(width):
        for y in range(height):
            for j in range(itterations):
                img[x, y] += rays[x+y*width+j*width*height, 2]
    return img

# lets all Rays hit a specific Object


def hitObject(obj, rays):
    t, normals, hitPoints, material = obj.hit(rays)
    hits = np.array([(tn, n, m) for tn, n, m in zip(t, normals, material)],
                    dtype="float64, object, object")
    return hits

# return the minimal but still >0 value of a[0] or b[0]


def filterMin(a, b):
    if (a[0] > 0):
        if (b[0] > 0):
            if (a[0] < b[0]):
                return a
            return b
        return a
    return b

# generate a new Ray with the Material in the Hit


def rayFromHit(ray, hit):
    return hit[2].outRay(ray, hit[1], rayAt(ray, hit[0]))

# Bounce all Rays once


def bounce(rays, objects):
    # Intialize the Ray filter and the hits
    rayFilter = np.full(rays.shape[0], True)
    hits = np.full((rays.shape[0]), np.array(
        [(-1, None, None)], dtype="float64, object, object"))

    # hit all Objects
    for obj in objects:
        # Hit the current Object with all the Rays
        newHits = hitObject(obj, rays)

        # find the first Hit of all the Rays and update the Hits accordingly
        hits = np.array([filterMin(h, nh)
                         for h, nh in zip(hits, newHits)])

    # update the Ray filter with all rays that didn't hit anything
    rayFilter &= np.array([h[0] > 0 for h in hits])

    # if everything is filtered out return
    if (not np.any(rayFilter)):
        return (rays, rayFilter)

    # let all Rays interact with the Material of the hit
    rays[rayFilter] = np.array([rayFromHit(ray, hit)
                                for hit, ray in zip(hits[rayFilter], rays[rayFilter])])

    # Filter all the Rays that hit Emissive Objects out
    rayFilter[rayFilter] &= np.invert(
        np.all(rays[rayFilter][:, 1] == 0, axis=1))

    # Return all the rays and the RayFilter update
    return (rays, rayFilter)


def render(camera, objects, bounces: int = 5, itterations: int = 1):
    width = camera.imageSize.width
    height = camera.imageSize.height

    # generate all the Rays
    rays = generateRays(camera, itterations)
    rayFilter = np.full(rays.shape[0], True)

    # bounce the Rays
    for i in range(bounces):
        print('bounce {}/{}       '.format(i+1, bounces), end='\r')
        # bounce all the Rays that are allowed to bounce once and update the filter
        rays[rayFilter], rayFilter[rayFilter] = bounce(
            rays[rayFilter], objects)
        # if everything is filtered out break the loop
        if (not np.any(rayFilter)):
            break

    # Convert the Color Values of the Rays to an Image
    img = createImage(rays, width, height, itterations)

    return img


def Plane(origin, directions, material):
    p1 = origin
    p2 = origin + directions[0]
    p3 = origin + directions[1]
    p4 = p2 + directions[1]
    #point3 = points[2] + (points[1] - points[0])
    return [
        Triangle(np.array([
            np.array(p2),
            np.array(p1),
            np.array(p3),
        ]), material),
        Triangle(np.array([
            np.array(p2),
            np.array(p3),
            np.array(p4),
        ]), material),
    ]


def Block(origin, directions, material):
    # bottom
    p1 = origin
    p2 = p1 + directions[0]
    p3 = p1 + directions[1]
    p4 = p1 + directions[2]

    return [
        *Plane(p1, np.array([
            directions[0], directions[1]
        ]), material),
        *Plane(p1, np.array([
            directions[1], directions[2]
        ]), material),
        *Plane(p1, np.array([
            directions[0], directions[2]
        ]), material),
        *Plane(p2, np.array([
            directions[1], directions[2]
        ]), material),
        *Plane(p3, np.array([
            directions[0], directions[2]
        ]), material),
        *Plane(p4, np.array([
            directions[0], directions[1]
        ]), material)
    ]


white = Lambert([0.8, 0.8, 0.8])
light = Emissive([500, 500, 500])
blue = Lambert([0.5, 0.5, 1])
red = Lambert([1, 0.5, 0.5])
yellow = Lambert([1, 1, 0.5])

floor = Plane(np.array([0.0, 0.0, 0.0]),
              np.array([
                  [552.8, 0.0, 0.0],
                  [0.0, 0.0, 559.2]
              ]), white)

light = Plane(np.array([343.0, 548.7, 332.0]),
              np.array([
                  [0.0, 0.0, -105.0],
                  [-130.0, 0.0, 0.0]
              ]), light)

ceiling = Plane(np.array([556.0, 548.8, 559.2]),
                np.array([
                    [0.0, 0.0, -559.2],
                    [-556.0, 0.0, 0.0]
                ]), white)

backWall = Plane(np.array([0.0, 0.0, 559.2]),
                 np.array([
                     [549.6,   0.0, 0.0],
                     [0.0, 548.8, 0.0]
                 ]), white)

rightWall = Plane(np.array([0.0, 0.0, 0.0]),
                  np.array([
                      [0.0, 0.0, 559.2],
                      [0.0, 548.8, 0.0]
                  ]), blue)

leftWall = Plane(np.array([549.6,   0.0, 559.2]),
                 np.array([
                     [3.2,   0.0,   -559.2],
                     [6.4, 548.8, 0.0]
                 ]), red)

shortBlock = Block(np.array([500.0, 165.0, 200.0]),
                   np.array([
                       [0.0, -165.0, 0.0],
                       [-160.0, 0.0, -49.0],
                       [-50.0, 0.0, 158.0]
                   ]), yellow)

tallBlock = Block(np.array([290.0, 330.0, 406.0]),
                  np.array([
                      [-49.0, 0.0, -159.0],
                      [-158.0, 0.0, 50.0],
                      [0.0, -330.0, 0.0]
                  ]), blue)

objects: list[Object] = [*floor, *light, *ceiling, *
                         backWall, *rightWall, *leftWall, *shortBlock, *tallBlock]


width = int(input("Width: "))
height = int(input("Height: "))
bounces = int(input("Bounces: "))
itterations = int(input("Ray Multiplier: "))

camera = Camera(ImageSize(width, height), np.array(
    [278, 273, -800]), np.array([0, 0, 1]), np.array([0, 1, 0]))

# ====================== RENDER...
itteration = 0
img = np.zeros((width, height, 3))
now = int(time.time())

try:
    while (True):
        print('itteration: {}'.format(itteration))
        itteration += 1
        img += render(camera, objects, bounces, itterations)

        # save data
        np.savez_compressed(
            '{}-cornell_box_linear_data'.format(now), image=img)

        i = ndimage.rotate(img, -90)
        m = np.max(img)/5
        i = np.power((i/m), 0.45)
        i = np.clip(i, 0, 1)
        # save plot
        plt.imsave('{}-cornell_box.png'.format(now), i)
        plt.imshow(i)
except KeyboardInterrupt:
    print("keyboard interrupt, halting program")
    np.savez_compressed(
        '{}-cornell_box_linear_data'.format(now), image=img)
