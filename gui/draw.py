import pygame


def draw_road(surface, center: (float, float), radius: float, width: float):
    pygame.draw.circle(
        surface=surface, color=(0, 255, 0), center=center, radius=radius, width=width
    )
