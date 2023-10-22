import pygame

display_size = (920, 720)
display_color = "black"
vis_rect = (0, 0, 720, 720)
vis_color = (30, 30, 30)
vis_center = (vis_rect[2] / 2, vis_rect[3] / 2)

road_radius = 300
road_width = 30
road_color = (255, 255, 255)

car_image = pygame.image.load("assets/car_1.png")

car_image_rect = car_image.get_rect()
