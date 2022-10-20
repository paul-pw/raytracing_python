from concurrent.futures import thread
from tokenize import Floatnumber, Number
from typing import Tuple
from winreg import DisableReflectionKey
import numpy as np
import random

# Typedefs
Vec3 = np.array(np.float32)
Vec2 = np.array(np.float32)


def normalize_vector(vec):
    return vec/np.linalg.norm(vec)


def degToVec(phi, theta):
    return np.array[
        np.cos(phi)*np.sin(theta),
        np.sin(phi)*np.sin(theta),
        np.cos(theta)
    ]


class Ray:
    """this class represents a Light Ray"""

    def __init__(self, startPoint: Vec3, direction: Vec3, pixel, lightValue: Vec3) -> None:
        self.startPoint = startPoint
        self.direction = direction
        self.lightValue: Vec3 = lightValue
        self._ended = False

    def setEnded(self):
        self._ended = True

    def hasEnded(self):
        return self._ended


class Material():
    # TODO find a better name.
    def reflection(self, incomming: Vec3) -> Tuple(Vec3, Vec3):
        pass


class Object:
    """this represents an Object"""

    def __init__(self, material: Material) -> None:
        self.material = material


class DefaultMaterial(Material):
    def __init__(self, color: Vec3) -> None:
        assert all(map(lambda x: 1 >= x >= 0, color)
                   ), f"color has to be between 0 and 1, got {color}"
        self.color = color
        super().__init__()

    def reflection(self, incomming: Vec3, normal: Vec3) -> Tuple(Vec3, Vec3):
        reflect = (2*np.dot(normal, incomming))*normal-incomming
        return (reflect, self.color)


class Camera:
    """this reperesents a Camera"""

    def __init__(self, objects, focalLength, resolution: Tuple(Number, Number), position: Vec3, orientation: Vec2) -> None:
        # setup the scene
        self.objects = objects
        self.position = position
        self.orientation = orientation

        # store the Resolution
        self.resolution = resolution

        # calc fov in °
        self.fov = 2 * np.arctan(35/(2*focalLength))

        # degreesPerPixel
        self.degreesPerPixel = self.fov / \
            np.sqrt(self.resolution[0]**2+self.resolution[1]
                    ** 2)  # fov/diagonalPixelCount

        pass

    def render(self, samples: Number):

        output = np.zeros((self.resolution[0], self.resolution[1], 3))

        for x in range(0, self.resolution[0]):
            for y in range(0, self.resolution[1]):
                ray = self._pixeltoRay(x, y)  # generate a Ray
                ray = self._shootRay(ray, samples)  # shoot that Ray
                # clip lower bounds to zero
                output[x, y] += max(0, ray.lightValue)

    def _pixelToRay(self, x, y):
        [top, left] = self.resolution/2-self.resolution

        # get the x and y position in degrees (one could add some variance in the x,y position here for anti aliasing)
        xInDegrees = (left + x) * self.degreesPerPixel
        yInDegrees = (top + y) * self.degreesPerPixel

        theta = self.orientation[0] + xInDegrees
        phi = self.orientation[1] + yInDegrees

        # to start out, the ray has negative values. Light sources also have negative values, so a ray only has a positive value if it interacted with a light source.
        return Ray(self.position, degToVec(phi, theta), [-1, -1, -1])

    def _shootRay(self, ray: Ray, samples: Number) -> Ray:
        workingRay = ray
        for i in range(samples):  # TODO hier vielleicht nicht while sondern for
            [IntersectionPoint, normal,
                intersectedObject] = calcIntersection(workingRay, self.objects)
            workingRay = calcReflection(workingRay, IntersectionPoint,
                                        normal, intersectedObject.material)
            if (workingRay.hasEnded()):
                break
        return workingRay


def calcIntersection(ray: Ray, objects: list(Object)) -> Tuple(Vec3, Vec3, Object):
    """this calculates the first intersection of the ray and any of the Objects"""
    """returns (inf,inf,inf) for intersection Point and (0,0,0) for Normal Vector"""


def calcReflection(ray: Ray, point: Vec3, normal: Vec3, material: Material) -> Ray:
    """calculates the Repflection of the Object with the incomming ray"""
    [rayDirection, color] = material.reflection(ray.direction, normal)
    newRay = Ray(point, rayDirection, ray.LightValue*color)

    # TODO find better stop condition and do better stop handling!
    if any(map(lambda x:  x > 1, color)) or all(map(lambda x:  x == 0, color)):
        newRay.setEnded()

    return newRay


def main():


    # Using the special variable
    # __name__
if __name__ == "__main__":
    main()
